import numpy as np
from librosa import display
import librosa

sample_rate = 11025

# separate into bins
n_bins = 128
# hearing range (herz)
log_min = np.log2(1)
log_max = np.log2(4096)
# generate log spaced bins
bin_limits = np.logspace(log_min, log_max, num=n_bins, base=2.0)
bin_limits = bin_limits[1:]

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

def make_new_bins(x, x_bins, new_bins=bin_limits):
    _, n_cols = np.shape(x)
    # see which frequencies belong to which bin
    x_classes = np.digitize(x_bins, new_bins)
    # get max amplitude in each bin 
    new_x = np.zeros((len(new_bins), n_cols))
    for i in range(len(new_bins)):
        # get index of first class in bin
        first = np.argmax(x_classes == i)
        # get indices for classes in bin
        idx = x_classes == i
        if any(idx):
            # index of max per column
            max_idx = np.argmax(x[idx], axis=0)
            # vector of classes with max values
            new_x[i] = x_bins[max_idx + first]
    return new_x

def get_feature_vector(filename, sample_rate=sample_rate,
                       duration=None, n_fft=framelength,
                       hop_length=superpose, ranges=bin_limits):
    x, freqs = get_sfft(filename, sample_rate=sample_rate, duration=duration,
                        n_fft=framelength, hop_length=superpose)
    features = make_new_bins(x, freqs)
    return np.array(features)

def load_db(path):
    return np.load(path)

