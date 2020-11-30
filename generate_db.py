from reto import get_feature_vector
import numpy as np
import os
import glob

path = './db/'

# read wav files in directory
files = glob.glob(os.path.join(path, '*.wav'))
features = [0] * len(files)
for i in range(len(files)):
    features[i] = get_feature_vector(files[i])
    print(files[i])
db_dict = dict(zip(files, features))

# save filenames and features to binary file 
np.savez('db', **db_dict)

