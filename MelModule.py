import librosa
import numpy as np

class Mel:
	'''
		Mel()
		converts wav and mp3 files to mel spectrograms. Also will pad to a constant size across the dataset
	'''
	def __init__(self, bands=128, sr=16000, frameLen=512, hopLen=256, clipDur=3):
		super().__init__()
		
		# Mel spectrogram parameters
		self.bands = bands
		self.sampleLimit = sr * clipDur
		self.sr = sr
		self.clipDur = clipDur
		self.frameLen = frameLen
		self.hopLen = hopLen

	# _pad_sig()
	# private function used to internally pad spectrograms.
	# returns: (padded signal, original length)
	def _pad_sig(self, signal):
		orig_len = signal.shape[0]
		if signal.shape[0] < self.sampleLimit:
			padded = np.zeros((self.sampleLimit))
			padded[:signal.shape[0]] = signal
			signal = padded
		elif signal.shape[0] > self.sampleLimit:
			raise OverflowError("Input file too long to be padded up.")

		return signal, orig_len

	# mel_normalizer()
	# normalizes an entire batch at once so the values range from 0 to 1
	def normalize_spec(self, spec):
		spec += abs(np.amin(spec))
		return spec / np.amax(spec)

	# conv()
	# Converts the loaded audio signal to a spectrogram
	# returns: (spectrogram padded and normalized, length)
	def conv(self, signal):
		signal = self._pad_sig(signal)

		spec = librosa.feature.melspectrogram(y=signal[0], sr=self.sr, n_fft=self.frameLen, hop_length=self.hopLen, n_mels=self.bands)
		spec = librosa.power_to_db(spec)
		spec = self.normalize_spec(spec)

		seqLen = (signal[1] // self.hopLen) + 1
		assert (spec[:, seqLen+1] == 0).max()

		return spec, seqLen

	# conv_batch()
	# Converts a batch of audio signals to spectrograms
	def conv_batch(self, signalBatch):
		y = []
		for signal in signalBatch:
			spec, seqLen = self.conv(signal)
			y.append(np.array((spec, seqLen), dtype=object))
		return np.stack(y)