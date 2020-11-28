import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from librosa import display
import librosa

sample_rate = 11025

# separate into bins
n_bins = 254
# hearing range (herz)
log_min = np.log2(1)
log_max = np.log2(4096)
# generate log spaced bins
bin_limits = np.logspace(log_min, log_max, num=n_bins, base=2.0)

# parameters for fft
framelength = 4096
superpose = int(2*framelength/3) # (2/3)

def get_sfft(filename, sample_rate=sample_rate, duration=None,
             n_fft=framelength, hop_length=superpose):
    song, sr = librosa.load(filename, sr=sample_rate, duration=duration)
    # matrix has shape [n_frecuency_bins x time_frames]
    x = librosa.stft(song, n_fft=framelength, hop_length=superpose)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=framelength)
    return x, freqs

def get_maxs(matrix, freqs, ranges=bin_limits):
    # get indices of max values at each time
    # (max values per column)
    idx = np.argmax(matrix, axis=0)
    # get frequencies corresponding to index
    max_freqs = [freqs[i] for i in idx]
    # get bin id that contains frequency
    max_bins = [np.argmax(freq < ranges) for freq in max_freqs]
    return max_bins

def get_feature_vector(filename, sample_rate=sample_rate,
                       duration=None, n_fft=framelength,
                       hop_length=superpose, ranges=bin_limits):
    x, freqs = get_sfft(filename, sample_rate=sample_rate, duration=duration,
                        n_fft=framelength, hop_length=superpose)
    features = get_maxs(x, freqs, ranges=ranges)
    return np.array(features, dtype=int)

def load_db(path):
    db = np.load(path, allow_pickle=True)
    files = db['files']
    features = [np.array(row, dtype=int) for row in db['features']]
    return files, features


