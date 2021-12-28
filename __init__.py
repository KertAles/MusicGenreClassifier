
from scipy import signal
from scipy.io import wavfile

import numpy as np
import librosa



path='./dataset/Data/genres_original/blues/blues.00000.wav'


sample_rate, samples = wavfile.read(path)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate)



spectrogram = spectrogram[10:120, :]

meanvect = np.mean(spectrogram, axis=1)
covmat = np.cov(spectrogram)


vect = meanvect

for idx, row in enumerate(covmat):
    vect = np.concatenate((vect, row[idx:]))
    



from loadData import load_split_data, load_split_data_spect

data_train, label_train, data_test, label_test = load_split_data_spect(split=0.8, shuffle=True)
