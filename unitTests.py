# File used to test functionality of every module created in this
import MelModule
import CasterModule
import DatasetCreator
from CONSTS import *

import pandas as pd
import librosa
import numpy as np
import os
import glob

# Constants for unit tests
TESTS = 30
BUFF_LIM = 20

"""
datasetGenerator.py
	- DO ALL TESTS FOR COMMONVOICE AND LIBRISPEECH
	- test to see if buffer gets filled properly
	- confirm process is able to handle .flac and mp3

"""

# Test definitions ========================================================

# All functions return codes depending on success
	# 0 --> successful on all tests
	# num --> failed on this test

def buff_fill(dataset):
	if len(dataset) > BUFF_LIM:		# dataset doesn't have to be full because if files were ejected it won't be
		return 2
	for file, label in dataset:
		if os.path.exists(file) == False:
			return 1
	return 0

def dataset_processor(dataset):
	caster = CasterModule.Caster()
	melConv = MelModule.Mel(sr=SR, clipDur=MAX_CLIP_DUR)

	specs, labels, seqLens = [], [], []
	for file, label in dataset:
		try:
			# If file is too long or label is too long it gets ejected
			specPair = melConv.conv(librosa.load(file, sr=SR)[0])
			labelPair = caster.padded_str_to_map(label, MAX_TARGET_LEN)
		except OverflowError:
			continue

		labels.append(labelPair[0])
		specs.append(specPair[0])
		seqLens.append([specPair[1], labelPair[1]])

	if (len(labels) == len(specs) == len(seqLens)) == False:
		return 2
	if len(labels) == 0:
		return 1
	return 0

def load_compat(dataset):
	try:
		for file, _ in dataset:
			librosa.load(file, sr=SR)[0].shape[0] <= 0
	except:
		return 1
	return 0

def mel_normalize_clip_test(signal):
	paddedSig = melConv._pad_sig(audioFiles[0])
	if paddedSig[0].shape[0] != (SR * MAX_CLIP_DUR):					# Improper length padded
		return 3
	if (paddedSig[0][:signal.shape[0]] == signal).min() == False:	# Verify's data was unaltered
		return 2
	if (paddedSig[0][paddedSig[1] + 1:] == 0).min() == False:		# If any of the padded is not 0
		return 1
	return 0

def mel_convert_test(signal):
	conv = melConv.conv(signal)
	if conv[0].shape[0] != melConv.bands:
		return 2

	i = conv[1] + 1
	while (conv[0][:, i] != 0).max() != False:
		i += 1

		if i != conv[1]:
			return 3
	return 0

def singluar_mel_normalize(spectrogram):
	melNorm = melConv.normalize_spec(spectrogram)
	if melNorm.min() < 0:							# Proper max bound
		return 3
	if melNorm.max() > 1:							# Proper min bound
		return 2
	if spectrogram.shape != melNorm.shape:			# Unaltered shape
		return 1
	return 0

def mel_convert_batch(signalBatch):
	batch = melConv.conv_batch(signalBatch)
	for pair in batch:
		spec, seqLen = pair
		if spec.max() > 1:
			return 3
		if spec.min() < 0:
			return 2
		if (spec[:, seqLen + 1] == 0).min() == 0:
			return 1
	return 0

def cast_stm():
	regStr = "hello"
	capStr = "HELLO"
	invStr = "he--llo-"

	idealMap = [START_ID, 8, 5, 12, 12, 15, END_ID]

	if (c.str_to_map(regStr) != idealMap).max() == True:
		return 3
	if (c.str_to_map(capStr) != idealMap).max() == True:
		return 2
	if (c.str_to_map(invStr) != idealMap).max() == True:
		return 1

	return 0

def ctc_test():
	notClean = [4, 4, 4, 4, BLANK_ID, 2, 3, 1, 1, BLANK_ID, 3, 4 ,7, BLANK_ID, SPACE_ID, BLANK_ID, BLANK_ID, 1]
	
	clean = [4, 2, 3, 1, 3, 4, 7, SPACE_ID, 1]

	if (c.ctc_strip(notClean) != clean).max() == True:
		return 1
	return 0

def cast_mts():
	regMap = [START_ID, 8, 5, 12, BLANK_ID, 12, 15, END_ID]
	notCleanMap = [START_ID, 8, 8, 8, 5, BLANK_ID, BLANK_ID, BLANK_ID, 12, BLANK_ID, 12, 15, END_ID]

	idealStr = list("hello")
	idealStr.insert(0, "<s>")
	idealStr.append("<e>")

	if (c.map_to_str(regMap) != idealStr).max() == True:
		return 2
	conv = c.ctc_strip(notCleanMap)
	conv = c.map_to_str(conv)
	if (conv != idealStr).max() == True:
		print(conv)
		print(idealStr)
		return 1 
	return 0

def cast_compat(string):
	conv = c.str_to_map(string)
	conv = c.map_to_str(conv)
	if ''.join(conv[1:-1]) != string.lower():
		return 1
	return 0

# result_str()
# Test String representations ========================================================
def result_str(label, code):
	if code == 0:
		result = "SUCCEEDED"
	else:
		result = f"FAILED on test with code {code}"

	return f"{label} {result}"

# Tests ====================================================================
if __name__ == '__main__':
	correct = 0
	audioFiles = [librosa.load(f"testFiles/0_08_{i}.wav")[0] for i in range(0, 5)]

	# Mel tests	===================================================================
	melConv = MelModule.Mel(bands=128, sr=SR, frameLen=512, hopLen=256, clipDur=MAX_CLIP_DUR)
	
	result = mel_normalize_clip_test(audioFiles[0])
	correct += 3 - result
	print(result_str("mel_pad", result))

	result = mel_convert_test(audioFiles[0])
	correct += 2 - result
	print(result_str("mel_conv", result))

	spec = librosa.power_to_db(librosa.feature.melspectrogram(y=melConv._pad_sig(audioFiles[0])[0], sr=melConv.sr, n_fft=melConv.frameLen, hop_length=melConv.hopLen, n_mels=melConv.bands))
	result = singluar_mel_normalize(spec)
	correct += 3 - result
	print(result_str("mel_norm_sing", result))

	result = mel_convert_batch(audioFiles)
	correct += 3 - result
	print(result_str("mel_conv_batch", result))
	del melConv

	# Cast tests ===================================================================
	c = CasterModule.Caster()

	result = cast_stm()
	correct += 3 - result
	print(result_str("cast_stm", result))

	result = ctc_test()
	correct += 1 - result
	print(result_str("ctc_strip", result))

	result = cast_mts()
	correct += 2 - result
	print(result_str("cast_mts", result))

	result = cast_compat("You are the worst")
	correct += 1 - result
	print(result_str("cast_compat_1", result))

	result = cast_compat("theWOrstt         Bro")
	correct += 1 - result
	print(result_str("cast_compat_2", result))

	result = cast_compat("Im giving the test a super long string to work with hopign THAT it WIL L break some thing but WE DONT know what will hapend")
	correct += 1 - result
	print(result_str("cast_compat_3", result))
	del c

	# Dataset generator tests ===================================================================

	dataset = []

	chapter = "dataset\\LibriSpeech\\dev-clean\\1272\\128104\\"
	readerID, chapterID = chapter[:-1].split("\\")[-2:]	# extracts READER_ID and CHAPTER_ID from 'dataset\LibriSpeech\dir\READER_ID\CHAPTER_ID'		
	transcript = f"{chapter}{readerID}-{chapterID}.trans.txt"

	with open(transcript, 'r') as f:
		delimLength = 6 + len(readerID) + len(chapterID)	# lines begin with 'readerID-chapterID-numberID'. numberID is always XXXX and 2 more for dashes
		for line in f.readlines():
			file = chapter + line[:delimLength] + ".flac"
			label = line[delimLength + 1:-1] 					# clips from file to before the \n
			dataset.append((file, label))
			if len(dataset) >= BUFF_LIM:
				break

	result = buff_fill(dataset)
	correct += 2 - result
	print(result_str("ls_buff_fill", result))

	result = dataset_processor(dataset)
	correct += 2 - result
	print(result_str("ls_dataset_processor", result))

	result = load_compat(dataset)
	correct += 1 - result
	print(result_str("ls_load_compat", result))


	df = pd.read_table('Dataset\\validated.tsv')	
	for path, sent in zip(df["path"], df["sentence"]):
		if type(sent) == type("string"):
			file = "Dataset\\Processed_CV_Wavs\\" + path[:-3] + "wav"
			if os.path.exists(file):
				label = sent
				dataset.append((file, label))
				if len(dataset) >= BUFF_LIM:
					break

	result = buff_fill(dataset)
	correct += 2 - result
	print(result_str("cv_buff_fill", result))

	result = dataset_processor(dataset)
	correct += 2 - result
	print(result_str("cv_dataset_processor", result))

	result = load_compat(dataset)
	correct += 1 - result
	print(result_str("cv_load_compat", result))



	print(f"\n===============================================================\n({correct} / {TESTS}) tests were correct")
