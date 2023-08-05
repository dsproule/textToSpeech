# Speech to Text Model
import time
import glob
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from CONSTS import *

class PositionalEncoder:
	'''
	PositionalEncoder()
	Creates positional encoder used to embed the decoder sequence.

	seqLen: Max sequence length 
	embeddingDim: The number of features per embedded word
	'''
	def __init__(self, seqLen, embeddingDim):
		self.posEncoding = torch.zeros(seqLen, embeddingDim)
		for pos in range(seqLen):
			for i in range(0, embeddingDim, 2):
				self.posEncoding[pos, i] = np.sin(pos / (1e4)**((2 * i) / embeddingDim))
				self.posEncoding[pos, i + 1] = np.cos(pos / (1e4)**((2 * i) / embeddingDim))

		self.posEncoding.unsqueeze(0)

	def pos_encode(self, target):
		return target + self.posEncoding

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

		# Spectrogram Embedder --
		self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding="same")
		self.pooler = nn.MaxPool2d(kernel_size=2)
		self.conv2 = nn.Conv2d(256, 256, kernel_size=3)

		self.conv3 = nn.Conv2d(256, 128, kernel_size=3)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
		self.norm1 = nn.BatchNorm2d(256)
		self.norm2 = nn.BatchNorm2d(256)

		self.dense1 = nn.Linear(59, 128)
		self.dense2 = nn.Linear(128, 512)
		self.dense3 = nn.Linear(512, 256)
		self.dense4 = nn.Linear(256, EMBEDDING_SIZE) 	# per pooling layer (/ 2). per cnn w/o padding (- 2) (128 -> 31)

		self.norm3 = nn.BatchNorm1d(42)		# Frames (42)
		self.norm4 = nn.BatchNorm1d(42)

		# Transformer --
		encoderLayerConfig = nn.TransformerEncoderLayer(EMBEDDING_SIZE, nhead=8)
		self.encoder = nn.TransformerEncoder(encoderLayerConfig, num_layers=5)

		self.decoderEmbedder = nn.Embedding(30, EMBEDDING_SIZE)		# 30 for alphabet, space, end, start, blank
		self.decoderPosEmbedder = PositionalEncoder(TRAINING_TARGET_LEN, EMBEDDING_SIZE)
		self.decoderPosEmbedder.posEncoding = self.decoderPosEmbedder.posEncoding.to(device)

		self.targetMask = torch.tril(torch.ones(TRAINING_TARGET_LEN, TRAINING_TARGET_LEN)).to(device)

		decoderLayerConfig = nn.TransformerDecoderLayer(EMBEDDING_SIZE, nhead=8)
		self.decoder = nn.TransformerDecoder(decoderLayerConfig, num_layers=5)

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
		
		return y
		

	# save_model()
	# saves the model as a state_dict
	def save_model(self, name):
		torch.save(self.state_dict(), name)

	# load_model()
	# loads in a state_dict to set parameters of current model
	def load_model(self, modelPath):
		# modelPath is path to a saved state_dict
		self.load_state_dict(torch.load(modelPath))

	# fit_data()
	# Attempts to fix a dataset to a neural network model
	# Output is expected in () shape
	def fit_data(self, dataset, epochs, lr=3e-4, mom=.01, dec=.01, clipRate=.5):
		print("\n-----------------------------------------------------------------------")
		print(f"Beginning training on model {self.idNum}. lr={lr}, mom={mom}, dec={dec}")

		# torch.autograd.set_detect_anomaly(True)

		strikes = 0
		curBestTrainLoss, curBestValLoss = float("inf"), float("inf")

		optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=mom, weight_decay=dec)
		lossFunction = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
		trigger, trigCount, noProgCount = False, 0, 0
		for epoch in range(epochs):
			startTime = time.time()
			self.train()
			curMax = float("inf")
			for labels, specLens, targetLens, specs in self.fetch_dataset(dataset, 'train'):
				output = self.forward(specs, labels)

				loss = lossFunction(F.log_softmax(output, dim=2), labels, input_lengths=specLens, target_lengths=targetLens)
				loss.backward()
				
				if trigger:
					if trigCount > 5:
						return loss
					print("Trigger set: ", loss)
					trigCount += 1

				if noProgCount > NO_PROGRESS_LIM:
					print("Model stopped progressing. Last recorded loss was", loss)
					return loss

				if loss < curBestTrainLoss:
					curBestTrainLoss = loss
					noProgCount = 0
					if len(glob.glob(f"models\\autosave-MNIST-model-{self.idNum}-*")) > 0:
						os.remove(glob.glob(f"models\\autosave-MNIST-model-{self.idNum}-*")[0])
					self.save_model(f"models\\autosave-MNIST-model-{self.idNum}-{round(loss.item(), 3)}")
				elif loss.isnan():
					# return loss
					trigger = True
				elif loss == 0:
					np.save("lossDebug\\specs.npy", specs.cpu().detach().numpy())
					np.save("lossDebug\\output.npy", output.cpu().detach().numpy())
					np.save("lossDebug\\labels.npy", labels.cpu().detach().numpy())
					np.save("lossDebug\\specLens.npy", specLens.cpu().detach().numpy())
					np.save("lossDebug\\targetLens.npy", targetLens.cpu().detach().numpy())
					print("Loss went to 0")
					exit()
				else:
					noProgCount += 1

				torch.nn.utils.clip_grad_norm_(self.parameters(), clipRate)
				optimizer.step()

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

	# encoder_pass()
	# Takes in a spectrogram or batch of spectrograms and returns the encoder context
	def encoder_pass(self, X):
		# Performs 'embedding' of spectrograms
		X = self.conv1(X)
		X = self.conv2(X)
		X = self.norm1(X)
		X = self.pooler(X)

		X = self.conv3(X)
		X = self.conv4(X)
		X = self.norm2(X)

		X = torch.mean(X, dim=1)		# Combines the results of all the filters
		X = X.transpose(1, 2)

		X = self.dense1(X)	
		X = self.dense2(X)
		X = self.norm3(X)

		X = self.dense3(X)
		X = self.dense4(X)
		X = self.norm4(X)
		X = X.transpose(0, 1)		# now in (time step, batch, feature) format
		
		# Generates encoder attention
		return self.encoder(X)

	# predict()
	def predict(self, spec):
		with torch.no_grad():
			self.eval()
			
			X = spec.view(1, 1, spec.shape[2], spec.shape[3])
			context = self.encoder_pass(X)

			outputSeq = torch.zeros((1, TRAINING_TARGET_LEN), dtype=torch.int, device=self.device)
			outputSeq[0, 0] = START_ID
			for step in range(1, TRAINING_TARGET_LEN):
				y = self.decoderEmbedder(outputSeq)
				y = self.decoderPosEmbedder.pos_encode(y)
				y = y.transpose(0, 1)

				decoderOutput = self.decoder(y, context, tgt_mask=self.targetMask)
				decoderOutput = self.decoderTolabels(decoderOutput)

				outputSeq[0, step] = torch.argmax(decoderOutput, dim=-1)[step]
			return outputSeq[0]


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else None)
	if device == None:
		raise Exception("GPU NOT AVAILABLE")

	modelId = 0
	clipRates = [4.0, 1.0]
	lrs = [3e-4, 1e-2, 2e-3]
	dec = .01
	moms = [.02, .1]
	for clipRate in clipRates:
		for mom in moms:
			for lr in lrs:
				net = Model(device, modelId)
				net.fit_data('AudioMNIST', 10, lr=lr, mom=mom, dec=dec, clipRate=clipRate)
				modelId += 1