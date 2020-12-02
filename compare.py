from reto import get_feature_vector, load_db
import numpy as np

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
    for i, features in zip(range(len(db)), db.values()):
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
    idx = np.argmin(scores_dict.values())
    key = scores_dict.keys()[idx]
    return key, scores_dict[key]


