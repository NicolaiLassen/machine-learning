from scipy.io import wavfile
import numpy as np


def wav_to_matrix(filename):
    fs, data = wavfile.read(filename + ".wav")
    np_wav = np.array(data, dtype=np.float32) / 32767.0
    return fs, np_wav


def matrix_to_wav(X, sample_rate, filename):
    wav = X * 32767.0
    wav = wav.astype(np.int16)
    wavfile.write(filename + ".wav", sample_rate, wav)
    return
