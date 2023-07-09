import glob
import pandas as pd
from CONSTS import * 
import DatasetCreator

dataset = []		# [(filePath, sentence)]
dp = DatasetCreator.DatasetProcessor('testDatasetCreator')
chunkID = 0
print("Beginning work on dataset...\n\nLoading LibriSpeech files into set...")

for chapter in glob.glob("dataset\\LibriSpeech\\*\\*\\*\\"):
	readerID, chapterID = chapter[:-1].split("\\")[-2:]	# extracts READER_ID and CHAPTER_ID from 'dataset\LibriSpeech\dir\READER_ID\CHAPTER_ID'		
	transcript = f"{chapter}{readerID}-{chapterID}.trans.txt"
	
	with open(transcript, 'r') as f:
		delimLength = 6 + len(readerID) + len(chapterID)	# lines begin with 'readerID-chapterID-numberID'. numberID is always XXXX and 2 more for dashes
		for line in f.readlines():
			file = chapter + line[:delimLength] + ".flac"
			label = line[delimLength + 1:-1] 					# clips from file to before the \n
			dataset.append((file, label))
			if len(dataset) >= BUFF_LIM:
				dp.process(dataset, chunkID)
				dataset = []
				chunkID += 1
				exit()

# print("Loading CommonVoice files into set...\n")
# # entire useful dataset but too much to proecess right now ['dataset\\en\\test.tsv', 'dataset\\en\\train.tsv', 'dataset\\en\\validated.tsv']
# enPath = 'dataset\\validated.tsv'
# df = pd.read_table(enPath)	
# for path, sent in zip(df["path"], df["sentence"]):
# 	if type(sent) == type("string"):
# 		file = "dataset\\clips\\" + path
# 		label = sent
# 		dataset.append((file, label))
# 		if len(dataset) >= BUFF_LIM:
# 			dp.process(dataset, chunkID)
# 			dataset = []
# 			chunkID += 1

print("Separating training data from validation data...")
dp.create_val(chunkID)