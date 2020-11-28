from reto import get_feature_vector
import numpy as np
import os
import glob

path = './db/'

# read wav files in directory
files = glob.glob(os.path.join(path, '*.wav'))
features = [0] * len(files)
for i in range(len(files)):
    print(files[i])
    features[i] = np.array(get_feature_vector(files[i]), dtype=int)
features = features

# save filenames and features to binary file 
np.savez('db', files=files, features=features)

