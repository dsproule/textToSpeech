import matplotlib.pyplot as plt
import model
import glob
import torch
import torch.nn as nn
from CONSTS import *
import numpy as np

# Go thru the models and run validation sets
# Save values and make a graph
# Print highest average
# Show Trends

TEST_LEN = 60
device = torch.device('cuda' if torch.cuda.is_available() else None)
if device == None:
	raise Exception("GPU NOT AVAILABLE")

points = list()
trainedModels = glob.glob("models\\*")
for i in range(20):
	j = 0
	modelParams = trainedModels[i]
	print(f"Loading model {modelParams[7:]}...")

	net = model.Model(device, 300).to(device)

	lossFunction = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
	net.eval()
	points.append([])
	for labels, specLens, targetLens, specs in net.fetch_dataset('ProcessedDataset', 'train', start=0, end=TEST_LEN // 2, step=1):
		output = net(specs, labels)
		print(output.shape, labels[0])
		loss = lossFunction(output, labels, input_lengths=specLens, target_lengths=targetLens)
		exit()
		points[i].append(loss.item())
		j += 1

	for labels, specLens, targetLens, specs in net.fetch_dataset('ProcessedDataset', 'val', start=399, end=399-(TEST_LEN // 2), step=-1):
		output = net(specs, labels)
		loss = lossFunction(output, labels, input_lengths=specLens, target_lengths=targetLens)
		points[i].append(loss.item())
		j += 1

points = np.array(points)
points = np.reshape(points, (20, -1))

np.save("models\\val-loss-data.npy", points)

meanPairs = {np.mean(points[i]): i for i in range(points.shape[0])}
sortedMeans = sorted(meanPairs)

i, j = 1, 0
f = open("models\\val-loss-order.txt", "w")
for mean in sortedMeans:
	plt.subplot(2, 2, i)
	plt.plot(points[meanPairs[mean]])
	f.write(trainedModels[meanPairs[mean]] + ', ' + str(mean) + '\n')
	j += 1
	if j % 5 == 0:
		i += 1

f.close()
plt.show()

