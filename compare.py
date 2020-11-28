from reto import get_feature_vector, load_db
import numpy as np

song_path = 'starman.wav'
db_path = 'db.npz'

# load database
filenames, db = load_db(db_path)
# shuffle
rng = np.random.default_rng()
idx = np.arange(len(filenames))
rng.shuffle(idx)
filenames = filenames[idx]
db = [db[i] for i in idx]

# get features for song
features = get_feature_vector(song_path)

# compare to database

# define similarity metric (l1 norm)
def get_similarity(v1, v2):
    dif = np.cumsum(np.abs(v1 - v2))
    return np.sum(dif)

# define convolution
def convo(f, v1, v2, window=None, step=None):
    # default
    if not window:
        window = int(len(v1) / 20)
    # fraction of v1 length 
    elif window < 1:
        window = int(len(v1) * window)

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
    while (window + i*step < len(v1)) and (same_count < 10):
        v1_i = v1[i*step : window + i*step]
        i += 1
        j = 0
        dif = 1
        while (window + j*step < len(v2)) and dif:
            v2_j = v2[j*step : window + j*step]
            j += 1
            dif = f(v1_i, v2_j)
            if dif == 0:
                same_count += 1
            else:
                same_count = 0
                value += dif
    stopped_early = same_count > 9

    return value, stopped_early

# compare with databse
def compare_db(sample, db):
    scores = np.zeros(len(db))
    for i in range(len(db)):
        score, stopped_early = convo(get_similarity, sample, db[i])
        if stopped_early:
            print('returning')
            return i
        scores[i] = score
    return np.argmin(scores)

# test
idx = compare_db(features, db)
print(idx)
print('match is: ', filenames[idx])

