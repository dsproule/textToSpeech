# Speech to Text Model
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
	def __init__(self, device):
		super().__init__()

		# General Config
		self.device = device


		# --- Model Layer Config ---

		# Spectrogram Embedder --
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
		self.pooler = nn.MaxPool2d(kernel_size=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

		self.dense1 = nn.Linear(31, EMBEDDING_SIZE) 	# per pooling layer (/ 2). per cnn w/o padding (- 2) (128 -> 31)

		# Transformer --
		encoderLayerConfig = nn.TransformerEncoderLayer(EMBEDDING_SIZE, nhead=8)
		self.encoder = nn.TransformerEncoder(encoderLayerConfig, num_layers=6)

		self.decoderEmbedder = nn.Embedding(30, EMBEDDING_SIZE)		# 30 for alphabet, space, end, start, blank
		self.decoderPosEmbedder = PositionalEncoder(MAX_TARGET_LEN, EMBEDDING_SIZE)
		self.targetMask = torch.tril(torch.ones(MAX_TARGET_LEN, MAX_TARGET_LEN)).bool()

		decoderLayerConfig = nn.TransformerDecoderLayer(EMBEDDING_SIZE, nhead=8)
		self.decoder = nn.TransformerDecoder(decoderLayerConfig, num_layers=6)

		self.dense2 = nn.Linear(EMBEDDING_SIZE, 30) # Embedding -> labels layer

	# forward()
	# defines the forward pass of the model
	def forward(self, X, y):
		# Performs 'embedding' of spectrograms
		X = self.conv1(X)
		X = self.pooler(X)
		X = self.conv2(X)
		X = self.pooler(X)

		X = torch.sum(X, dim=1)		# Combines the results of all the filters
		X = F.normalize(X, dim=1)
		X = X.transpose(1, 2)

		X = self.dense1(X)		
		X = X.transpose(0, 1)		# now in (time step, batch, feature) format
		
		# Generates encoder attention
		context = self.encoder(X)

		# Converts labels to embeddings and generating final output
		y = self.decoderEmbedder(y)
		y = self.decoderPosEmbedder.pos_encode(y)
		y = y.transpose(0, 1)
		y = self.decoder(y, context, tgt_mask=self.targetMask)

		y = self.dense2(y)
		return F.log_softmax(y, dim=2)
		

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
	def fit_data(self, dataset, epochs, lr=3e-4, mom=.01, dec = .01):
		# Convert dataset into batches
		mockX = torch.ones((BATCH_SIZE, 1, 128, 313))
		mockLabels = torch.ones((BATCH_SIZE, MAX_TARGET_LEN), dtype=torch.int32)
		mockLens = torch.ones((BATCH_SIZE), dtype=torch.int32)
		# ----------------------------

		optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=mom, weight_decay=dec)
		lossFunction = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
		for epoch in range(epochs):
			self.train()
			for batch in trainBatches:
				output = self.forward(mockX, mockLabels, mockLens)

				loss = lossFunction(output, targets, input_lengths=self._conv_output_dims(inputLens), target_lengths=targetLens)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
				optimizer.step()

			self.eval()
			for batch in valBatches:
				output = self.forward(mockX, mockLabels, mockLens)
				loss = lossFunction(output, targets, input_lengths=self._conv_output_dims(inputLens), target_lengths=targetLens)

		return loss

	# _conv_output_dims()
	# Will take in the original input sequence and trim down based on model transformations
	def _conv_output_dims(self, outputLens):
		outputLens -= 0 # conv1
		outputLens //= 2 # pooler
		outputLens -= 2 # conv2
		outputLens //= 2 # pooler

		return outputLens

if __name__ == '__main__':
	import MelModule
	import librosa

	melHandler = MelModule.Mel(clipDur=5)
	model = Model(0)
	audioFiles = [librosa.load(f"testFiles/0_08_{i}.wav")[0] for i in range(0, 3)]
	specs = np.stack([spec for spec, length in melHandler.conv_batch(audioFiles)])
	specs = torch.Tensor(specs).view(-1, 1, 128, 313)
	assert len(specs.shape) == 4 				# Used to reassure I remembered to use view()

	mockX = torch.ones((BATCH_SIZE, 1, 128, 313))
	mockLabels = torch.ones((BATCH_SIZE, MAX_TARGET_LEN), dtype=torch.int32)
	y = model(mockX, mockLabels)
	# print(model.fit_data('', 1).shape)
	
