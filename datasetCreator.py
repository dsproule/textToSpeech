from CONSTS import *
import CasterModule
import MelModule

import numpy as np
import os
import librosa
import shutil

class DatasetProcessor:
	'''
	DatasetProcessor()
	Takes in a dataset and converts them to desired values ready for network.

	Specifically will create directories for: Spectrogram w/ Lengths, TargetLabels w/ Lengths  
		Will also create a metadata file for record keeping and defaults
	'''
	def __init__(self, targetDir, trainSplit=.9):
		self.targetDir = targetDir
		self.trainSplit = trainSplit

		if targetDir not in os.listdir('.'):
			os.mkdir(targetDir)

			os.mkdir(f"{targetDir}/specs/")
			os.mkdir(f"{targetDir}/specs/train")
			os.mkdir(f"{targetDir}/specs/val")

			os.mkdir(f"{targetDir}/labels/")
			os.mkdir(f"{targetDir}/labels/train")
			os.mkdir(f"{targetDir}/labels/val")

			os.mkdir(f"{targetDir}/seqLens/")
			os.mkdir(f"{targetDir}/seqLens/train")
			os.mkdir(f"{targetDir}/seqLens/val")

	# process()
	# Will take datasetBuffer and create three associated npy files
	# 	first npy file will have spectrogram attached
	# 	second npy file will have label attached
	# 	third npy file will have seqLens in specLen, targetLen
	# respective files will be saved as chunkID_label.npy
	def process(self, datasetBuffer, chunkID):
		caster = CasterModule.Caster()
		melConv = MelModule.Mel(sr=8000, clipDur=10)	# Can be changed if need be

		specs, labels, seqLens = [], [], []
		for file, label in datasetBuffer:
			labelPair = caster.padded_str_to_map(label, MAX_TARGET_LEN)
			specPair = melConv.conv(librosa.load(file)[0])

			labels.append(labelPair[0])
			specs.append(specPair[0])
			seqLens.append([specPair[1], labelPair[1]])

		np.save(f"{self.targetDir}/labels/train/{chunkID}_labels.npy", np.stack(labels))
		np.save(f"{self.targetDir}/specs/train/{chunkID}_specs.npy", np.stack(specs))
		np.save(f"{self.targetDir}/seqLens/train/{chunkID}_seqLens.npy", np.stack(seqLens))

	# create_val()
	# All files generated with process() will be pushed to the train set. 
	# This function will transport the last chunks to the validation set
	def create_val(self, chunkCount):
		splitInd = int(self.trainSplit * chunkCount) 
		categories = ['labels', 'specs', 'seqLens']

		for category in categories:
			files = os.listdir(f"{self.targetDir}/{category}/train/")
			files.sort()

			for file in files[splitInd:]:
				shutil.move(f"{self.targetDir}/{category}/train/{file}", f"{self.targetDir}/{category}/val/")
		

if __name__ == '__main__':
	# Specifically only used to test process functionality
	dataset = []
	chunkID = 0
	dp = DatasetProcessor('testDatasetCreator')
	for i in range(0, 5):
		dataset.append((f"testFiles/0_08_{i}.wav", "zero"))
		if len(dataset) >= BUFF_LIM:
			dp.process(dataset, chunkID)
			dataset = []
			chunkID += 1

	dp.create_val(chunkID)