# main files
model.py						- Contains the transformer architecture
datasetConverter.py				- Contains the mel spectrogram converter and caster
CONSTS.py						- Global constant definitions for syncronization across filse

# dataset files
datasetCreator.py				- Used to create dataset
DatasetInfo 					- Details dataset metadata

# test files
unitTests.py					- Used to test functionality of custom modules

To Do
- Make Dataset creator
- Probably have to work CTC issues (specifically input lengths, target lengths and cuttoff)