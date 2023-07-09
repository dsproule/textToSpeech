# File used to test functionality of every module created in this
import MelModule
import CasterModule
import librosa
import numpy as np
from CONSTS import *

# Constants for unit tests
CLIP_DUR = 3
SR = 8000
TESTS = 20

"""
datasetGenerator.py
	- DO ALL TESTS FOR COMMONVOICE AND LIBRISPEECH
	- test to see if buffer gets filled properly
	- confirm process is able to handle .flac and mp3
	- confirm files are properly split even if some had to be ditched (eg. Over buffer)

"""

# Test definitions ========================================================

# All functions return codes depending on success
	# 0 --> successful on all tests
	# num --> failed on this test

def common_voice_buffer_fill():
	enPath = 'dataset\\validated.tsv'
	df = pd.read_table(enPath)	
	for path, sent in zip(df["path"], df["sentence"]):
		if type(sent) == type("string"):
			file = "dataset\\clips\\" + path
			label = sent
			dataset.append((file, label))
			if len(dataset) >= BUFF_LIM:
				pass
				
def mel_normalize_clip_test(signal):
	paddedSig = melConv._pad_sig(audioFiles[0])
	if paddedSig[0].shape[0] != (SR * CLIP_DUR):					# Improper length padded
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

	if c.str_to_map(regStr) != idealMap:
		return 3
	if c.str_to_map(capStr) != idealMap:
		return 2
	if c.str_to_map(invStr) != idealMap:
		return 1

	return 0

def ctc_test():
	notClean = [4, 4, 4, 4, BLANK_ID, 2, 3, 1, 1, BLANK_ID, 3, 4 ,7, BLANK_ID, SPACE_ID, BLANK_ID, BLANK_ID, 1]
	
	clean = [4, 2, 3, 1, 3, 4, 7, SPACE_ID, 1]

	if c.ctc_strip(notClean) != clean:
		return 1
	return 0

def cast_mts():
	regMap = [START_ID, 8, 5, 12, BLANK_ID, 12, 15, END_ID]
	notCleanMap = [START_ID, 8, 8, 8, 5, BLANK_ID, BLANK_ID, BLANK_ID, 12, BLANK_ID, 12, 15, END_ID]

	idealStr = list("hello")
	idealStr.insert(0, "<s>")
	idealStr.append("<e>")

	if c.map_to_str(regMap) != idealStr:
		return 2
	conv = c.ctc_strip(notCleanMap)
	conv = c.map_to_str(conv)
	if conv != idealStr:
		print(conv)
		print(idealStr)
		return 1 
	return 0

def cast_compat(string):
	conv = c.str_to_map(string)
	conv = c.map_to_str(conv)[1:-1]
	if ''.join(conv) != string.lower():
		print(''.join(conv))
		print(string.lower())
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
	melConv = MelModule.Mel(bands=128, sr=SR, frameLen=512, hopLen=256, clipDur=CLIP_DUR)
	
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


	print(f"\n===============================================================\n({correct} / {TESTS}) tests were correct")
