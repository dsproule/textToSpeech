# Speech to Text Model
import time
import glob
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from CONSTS import *
import nnUtils

class Model(nn.Module):
	'''
	Model()
	Transformer architecture defined for speech to text. Will take spectrograms padded to equal length 
	and convert them to sequential character string.

	X: Batches of spectrograms padded
	y: (labels for each value, length) 
		labels: padded to max sequence length w/ blanks so they look like [<start> chars.. <end> <space padding>]
		length: last index with data before padding commences
'''
	def __init__(self, device, idNum=EMPTY_ID):
		super().__init__()

		# General Config
		self.device = device
		self.idNum = idNum

		# --- Model Layer Config ---
		self.dropout = nn.Dropout(p=.5)

		# Spectrogram Embedder --
		BASE_CONF_1 = 84
		BASE_CONF_2 = 128

		self.conv1 = nn.Conv2d(1, BASE_CONF_1, kernel_size=3, padding="same")
		self.pooler = nn.MaxPool2d(kernel_size=2)
		self.conv2 = nn.Conv2d(BASE_CONF_1, BASE_CONF_2, kernel_size=3)
		self.norm1 = nn.BatchNorm2d(BASE_CONF_2)		

		self.conv3 = nn.Conv2d(BASE_CONF_2, BASE_CONF_2, kernel_size=3)
		self.conv4 = nn.Conv2d(BASE_CONF_2, BASE_CONF_1 + BASE_CONF_2, kernel_size=3)
		self.norm2 = nn.BatchNorm2d(BASE_CONF_1 + BASE_CONF_2)

		self.dense1 = nn.Linear(59, BASE_CONF_1)
		self.dense2 = nn.Linear(BASE_CONF_1, BASE_CONF_1)
		self.dense3 = nn.Linear(BASE_CONF_1, BASE_CONF_1)
		self.dense4 = nn.Linear(BASE_CONF_1, EMBEDDING_SIZE) 	# per pooling layer (/ 2). per cnn w/o padding (- 2) (128 -> 31)

		self.norm3 = nn.BatchNorm1d(26)		# Frames (26)
		self.norm4 = nn.BatchNorm1d(26)

		# Transformer --
		encoderLayerConfig = nn.TransformerEncoderLayer(EMBEDDING_SIZE, nhead=4)
		self.encoder = nn.TransformerEncoder(encoderLayerConfig, num_layers=2)

		self.decoderEmbedder = nn.Embedding(30, EMBEDDING_SIZE)		# 30 for alphabet, space, end, start, blank
		self.decoderPosEmbedder = nnUtils.PositionalEncoder(TRAINING_TARGET_LEN, EMBEDDING_SIZE)
		self.decoderPosEmbedder.posEncoding = self.decoderPosEmbedder.posEncoding.to(device)

		self.targetMask = torch.tril(torch.ones(TRAINING_TARGET_LEN, TRAINING_TARGET_LEN)).to(device)

		decoderLayerConfig = nn.TransformerDecoderLayer(EMBEDDING_SIZE, nhead=4)
		self.decoder = nn.TransformerDecoder(decoderLayerConfig, num_layers=2)

		self.decoderTolabels = nn.Linear(EMBEDDING_SIZE, 30) # Embedding -> labels layer
		self.to(device)

	# forward()
	# defines the forward pass of the model used for training. Requires target labels
	# and inputs 
	def forward(self, X, y):
		context = self.encoder_pass(X)

		# Converts labels to embeddings and generating final output
		y = self.decoderEmbedder(y)
		y = self.decoderPosEmbedder.pos_encode(y)
		y = y.transpose(0, 1)

		y = self.decoder(y, context, tgt_mask=self.targetMask)
		y = self.decoderTolabels(y)
		
		return F.log_softmax(y, dim=2)
	
	# encoder_pass()
	# Takes in a spectrogram or batch of spectrograms and returns the encoder context
	def encoder_pass(self, X):
		# Performs 'embedding' of spectrograms
		X = self.conv1(X)
		X = self.conv2(X)
		X = self.norm1(X)
		X = self.dropout(X)
		X = self.pooler(X)

		X = self.conv3(X)
		X = self.conv4(X)
		X = self.norm2(X)

		X = torch.mean(X, dim=1)		# Combines the results of all the filters
		X = X.transpose(1, 2)

		X = self.dense1(X)	
		X = self.dropout(X)
		X = self.dense2(X)
		X = self.norm3(X)

		X = self.dense3(X)
		X = self.dense4(X)
		X = self.dropout(X)
		X = self.norm4(X)
		X = X.transpose(0, 1)		# now in (time step, batch, feature) format
		
		# Generates encoder attention
		return self.encoder(X)

	# fit_data()
	# Attempts to fix a dataset to a neural network model
	# Output is expected in () shape
	def fit_data(self, dataset, epochs, lr=3e-4, mom=.01, dec=.01, clipRate=.5, modelName='default'):
		print("\n-----------------------------------------------------------------------")
		print(f"Beginning training on model {self.idNum}. lr={lr}, mom={mom}, dec={dec}")

		torch.autograd.set_detect_anomaly(True)

		sched = nnUtils.Scheduler(stagBuffSize=300)
		optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=mom, weight_decay=dec)
		lossFunction = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
		for epoch in range(epochs):
			self.train()
			for labels, specLens, targetLens, specs in self.fetch_dataset(dataset, 'train'):
				self.zero_grad()
				output = sched.train(self, specs, labels, style=GREEDY_SEARCH)

				loss = lossFunction(output, labels, input_lengths=specLens, target_lengths=targetLens)
				loss.backward()
				
				schedCode = sched.sched_check(loss.item(), printFreq=50)
				if schedCode != SUCCESS:
					if schedCode >= SOFT_ERROR:
						if schedCode == OPTIM_SHIFT:
							optimizer.param_groups[0]['lr'] = sched.get_optim_update()
						break
					elif schedCode == FATAL_ERROR_CODE:
						return FATAL_ERROR_CODE
				elif schedCode == SUCCESS:
					if len(glob.glob(f"models\\autosave-{modelName}-model-{self.idNum}-*")) > 0:
						os.remove(glob.glob(f"models\\autosave-{modelName}-model-{self.idNum}-*")[0])
					self.save_model(f"models\\autosave-{modelName}-model-{self.idNum}-{round(loss.item(), 3)}")

				torch.nn.utils.clip_grad_norm_(self.parameters(), clipRate)
				optimizer.step()

			self.eval()
			valLoss, lossBatches = 0, 0
			print("Calculating validation...")
			with torch.no_grad():
				for labels, specLens, targetLens, specs in self.fetch_dataset(dataset, 'val'):				
					output = self.greedy_search_inference(specs)
					valLoss += lossFunction(output, labels, input_lengths=specLens, target_lengths=targetLens).item()
					lossBatches += 1

			valLoss /= lossBatches
			
			print(f"Epoch ({epoch}), last val loss: {round(valLoss, 4)}, best training loss: {round(sched.bestTrainLoss, 4)}, prev best val loss: {round(sched.bestValLoss, 4)}")
			if sched.check_validation(valLoss) == SUCCESS:
				autosave = glob.glob(f"models\\autosave-{modelName}-model-{self.idNum}-*")[0]
				save = glob.glob(f"models\\{modelName}-model-{epoch}-{self.idNum}-*")

				os.remove(autosave)
				if len(save) > 0:
					os.remove(save[0])

				self.save_model(f"models\\{modelName}-model-{epoch}-{self.idNum}-{round(valLoss, 3)}")

			# if sched.shift_triggered():
			# 	sched.make_shift_adjustments()

		return loss

	# fetch_dataset()
	# Takes a dataset path and iterates through each chunk, batch size at a time to return
	# stored results
	# IS A GENERATOR SO MUST BE ITERATED THROUGH
	# returns labels, specLens, targetLens, specs
	def fetch_dataset(self, path, mode, start=0, end=None, step=1):
		if mode != 'train' and mode != 'val':
			raise ValueError(f"Mode: (\'{mode}\') is an invalid option. Use \'train\' or \'val\'")
		
		for file in glob.glob(f"{path}\\labels\\{mode}\\*")[start:end:step]:
			id_ = file[file.rfind("\\")+1:file.rfind("_")]

			labels = np.load(f"{path}\\labels\\{mode}\\{id_}_labels.npy")
			specLens = np.load(f"{path}\\specLens\\{mode}\\{id_}_specLens.npy")
			tLens = np.load(f"{path}\\tLens\\{mode}\\{id_}_tLens.npy")
			specs = np.load(f"{path}\\specs\\{mode}\\{id_}_specs.npy")

			fileLen = labels.shape[0]
			for batch in range(0, fileLen, BATCH_SIZE):
				if batch+BATCH_SIZE > fileLen:
					continue
				yield torch.tensor(labels[batch:batch+BATCH_SIZE], dtype=torch.int, device=self.device), \
						torch.tensor(specLens[batch:batch+BATCH_SIZE], device=self.device), \
						torch.tensor(tLens[batch:batch+BATCH_SIZE], device=self.device), \
						torch.tensor(specs[batch:batch+BATCH_SIZE], dtype=torch.float, device=self.device).view(BATCH_SIZE, 1, specs.shape[1], specs.shape[2])
	# save_model()
	# saves the model as a state_dict
	def save_model(self, name):
		torch.save(self.state_dict(), name)

	# load_model()
	# loads in a state_dict to set parameters of current model
	def load_model(self, modelPath):
		# modelPath is path to a saved state_dict
		self.zero_grad()
		self.load_state_dict(torch.load(modelPath))

	# beam_search_inference()
	# Uses beam search
	def beam_search_inference(self, spec, beamWidth=5):
		with torch.no_grad():
			self.eval()

			X = spec.view(1, 1, spec.shape[2], spec.shape[3])
			context = self.encoder_pass(X)

			seqs = [[START_ID]]
			scores = [0.0]
			for _ in range(TRAINING_TARGET_LEN):
				potSeqs = []
				potScores = []

				# Goes through every sequence and generates next possible tokens
				for i in range(len(seqs)):
					y = torch.tensor(seqs[i][-1], dtype=torch.int, device=self.device)

					y = self.decoderEmbedder(y)
					y = y.view(-1, 1, EMBEDDING_SIZE)

					decoderOutput = self.decoder(y, context)
					decoderOutput = self.decoderTolabels(decoderOutput)
					decoderOutput = F.log_softmax(decoderOutput, dim=-1)

					# top 5 tokens/scores are kept
					topProbs, topLabels = torch.topk(decoderOutput, k=beamWidth, dim=-1)
					topLabels, topProbs = topLabels.squeeze(), topProbs.squeeze()
					
					# each of these are appended to the current sequence along with their scores saved
					for j in range(beamWidth):
						potSeqs.append(seqs[i] + [topLabels[j].item()])
						potScores.append(scores[i] + topProbs[j].item())
					
				# Gets the most likely sequences based off their scores and saves them
				sortedInds = torch.argsort(torch.tensor(potScores), descending=True)
				seqs = [potSeqs[i] for i in sortedInds[:beamWidth]]
				scores = [potScores[i] for i in sortedInds[:beamWidth]]

				for seq in seqs:
					if seq[-1] == END_ID:
						return seqs[torch.argmax(torch.tensor(scores))]
		return seqs[torch.argmax(torch.tensor(scores))]

	# greedy_search_inference()
	def greedy_search_inference(self, X):
		context = self.encoder_pass(X)

		seqData = torch.zeros((TRAINING_TARGET_LEN, X.shape[0], 30), dtype=torch.float, device=self.device)

		for timeStep in range(TRAINING_TARGET_LEN):
			y = torch.argmax(seqData[:timeStep+1], dim=-1).view(X.shape[0], -1) # Extracts labels from softmax data

			y = self.decoderEmbedder(y)
			y = self.decoderPosEmbedder.sing_encode(y, timeStep)
			y = y.transpose(0, 1)

			decoderOutput = self.decoder(y, context)
			decoderOutput = self.decoderTolabels(decoderOutput)
			decoderOutput = F.log_softmax(decoderOutput, dim=-1).squeeze()

			seqData[timeStep] = decoderOutput[timeStep]

		return seqData

	# wip_inf
	def wip_inf(self, X):
		context = self.encoder_pass(X)

		seqData = torch.zeros((TRAINING_TARGET_LEN, X.shape[0], 30), dtype=torch.float, device=self.device)

		seqInit = torch.zeros(30, dtype=torch.float)
		seqInit[START_ID] = 1
		seqData[0, :] = torch.log_softmax(seqInit, dim=-1)


		for timeStep in range(TRAINING_TARGET_LEN - 1):
			y = torch.argmax(seqData[:timeStep], dim=-1).view(X.shape[0], -1) # Extracts labels from softmax data

			y = self.decoderEmbedder(y)
			y = self.decoderPosEmbedder.inf_encode(y)
			y = y.transpose(0, 1)

			decoderOutput = self.decoder(y, context)
			decoderOutput = self.decoderTolabels(decoderOutput)
			decoderOutput = F.log_softmax(decoderOutput, dim=-1).squeeze()

			seqData = decoderOutput
		
		return seqData

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else None)
	if device == None:
		raise Exception("GPU NOT AVAILABLE")

	# Basic Training -------------------
	# modelId = 0
	# clipRates = [4.0, 2.0]
	# lrs = [1e-2, 3e-4, 2e-3]
	# dec = .01
	# moms = [.02, .1]
	# for clipRate in clipRates:
	# 	for mom in moms:
	# 		for lr in lrs:
	# 			net = Model(device, modelId)
	# 			net.fit_data('AudioMNIST', 6, lr=lr, mom=mom, dec=dec, clipRate=clipRate, modelName="MNIST-GREED")
	# 			modelId += 1

	# Continued Training -------------------
	# modelId = 3
	# mom, dec, clipRate = .02, .01, 4.0
	# lrs = [2e-2, 3e-4]
	# for lr in lrs:
	# 	for file in glob.glob("models\\*"):
	# 		net = Model(device, modelId)
	# 		net.load_model(file)
	# 		net.fit_data('AudioMNIST', 4, lr=lr, mom=mom, dec=dec, clipRate=clipRate, modelName="MNIST-GREED")
	# 		modelId += 1


	# # Inference Test -------------------
	net = Model(device, idNum=12)

	label = torch.tensor(np.load("AudioMNIST\\labels\\val\\8_labels.npy")[6], dtype=torch.int, device=net.device).view(1, 65)
	spec = np.load("AudioMNIST\\specs\\val\\8_specs.npy")[6]
	spec = torch.tensor(spec, dtype=torch.float, device=device).view(1, 1, spec.shape[0], spec.shape[1])
	net.load_model("models\\MNIST-GREED-model-1-0-1.316")

	# fp = torch.argmax(net.forward(spec, label), dim=-1)

	# label = [28, 15, 14, 5, 29]
	# fp = net.greedy_search_inference(spec)
	# fp = net.wip_inf(spec)
	# print(torch.argmax(fp, dim=-1))
	# net.greedy_search_inference(spec)
	# print(torch.argmax(net.greedy_search_inference(spec), dim=-1))