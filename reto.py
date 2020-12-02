import numpy as np
import os
import glob
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

# define similarity metric (l1 norm)
def get_similarity(v1, v2):
    dif = np.cumsum(np.abs(v1 - v2))
    return np.sum(dif)

# define convolution
def convo(f, v1, v2, window=None, step=None, should_stop_early=True):
    # time length of v1
    _, n_t1 = np.shape(v1)
    # get time length of other matrix
    _, n_t2 = np.shape(v2)

    # default
    if not window:
        window = int(n_t1 / 20)
    # fraction of v1 length 
    elif window < 1:
        window = int(n_t1 * window)

    # default
    if not step:
        step = int(window * 2/3)
    # fraction of window
    elif step < 1:
        step = int(window * step)

    # perform convolution
    i = 0
    value = 0
    same_count = 0
    while (window + i*step < n_t1) and (same_count < 5):
        # get columns corresponding to time window
        v1_i = v1[:, i*step : window + i*step]
        # initialize counter for the other matrix
        j = 0
        while (window + j*step < n_t2) and (same_count < 5):
            v2_j = v2[:, j*step : window + j*step]
            j += 1
            dif = f(v1_i, v2_j)
            if dif == 0:
                i += 1
                v1_i = v1[:, i*step : window + i*step]
                if should_stop_early:
                    same_count += 1
            else:
                same_count = 0
                value += dif
        # step counter
        i += 1
    stopped_early = same_count > 4
    return value, stopped_early

# get first match
def get_first_match(sample, db, should_stop_early=True):
    scores = np.zeros(len(db))
    for i, features in zip(range(len(db)), list(db.values())):
        score, stopped_early = convo(get_similarity,
                                     sample,
                                     features,
                                     should_stop_early=should_stop_early)
        if should_stop_early and stopped_early:
            return score, list(db)[i]
        scores[i] = score
    best_i = np.argmin(scores)
    return list(db)[best_i], scores[best_i]

# compare with all in database
def compare_all(sample, db, should_stop_early=True):
    scores = {}
    for file, features in db.items():
        score, _ = convo(get_similarity,
                         sample,
                         features,
                         should_stop_early=should_stop_early)
        scores[file] = score
    return scores

# get best match from dict
def get_best_match(scores_dict):
    idx = np.argmin(list(scores_dict.values()))
    key = list(scores_dict.keys())[idx]
    return key, scores_dict[key]

# generate db
def generate_db(dir_path, out_path):
    # read wav files in directory
    files = glob.glob(os.path.join(dir_path, '*.wav'))
    features = [0] * len(files)
    for i in range(len(files)):
        print('Procesando {}/{}: {}...'.format(i + 1, len(files), files[i]))
        features[i] = get_feature_vector(files[i])
    db_dict = dict(zip(files, features))
    # save filenames and features to binary file 
    np.savez(out_path, **db_dict)

