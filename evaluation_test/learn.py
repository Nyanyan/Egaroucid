from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout, Lambda, LeakyReLU, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import randrange, sample, shuffle
import subprocess
import datetime
import os
from math import tanh, log
from copy import deepcopy
import sys

n_dense_pattern = int(sys.argv[1])
n_dense_additional = int(sys.argv[2])

use_phase = int(sys.argv[3])
ply_d = 2

input_files = sys.argv[4:]

n_epochs = 100

inf = 10000000.0

ply_d = 2

test_ratio = 0.2

n_patterns = 16
n_additional_features = 4
n_mobility = 4

n_raw_data = 86


pattern_nums = [
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12, 12, 12, 12,
    13, 13, 13, 13,
    14, 14, 14, 14,
    15, 15, 15, 15
]

n_raw_patterns = 62
n_raw_mobility = 16

feature_sizes = [8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10, 0, 0, 0, 0, 8, 8, 8, 8]
n_pattern_varieties = [4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4]

additional_feature_mul = [100, 50, 65, 65]






# create model
x = [None for _ in range(n_raw_data)]
y_before_add = []
names = ['line2', 'line3', 'line4', 'diagonal5', 'diagonal6', 'diagonal7', 'diagonal8', 'edge2X', 'triangle', 'edgeblock', 'cross', 'corner9', 'edge2Y', 'narrow_triangle', 'fish', 'kite']
idx = 0
for i in range(n_patterns):
    layers = []
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense0'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense1'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(1, name=names[i] + '_out'))
    add_elems = []
    for j in range(n_pattern_varieties[i]):
        x[idx] = Input(shape=feature_sizes[i] * 2, name=names[i] + '_in_' + str(j))
        tmp = x[idx]
        for layer in layers:
            tmp = layer(tmp)
        add_elems.append(tmp)
        idx += 1
    y_before_add.extend(add_elems)
add_names = ['surround', 'canput', 'stability', 'num']
for i in range(n_additional_features):
    x[idx] = Input(shape=2, name='')
    y_add = Dense(n_dense_additional, name=add_names[i] + '_dense0')(x[idx])
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(n_dense_additional, name=add_names[i] + '_dense1')(y_add)
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(n_dense_additional, name=add_names[i] + '_out')(y_add)
    y_before_add.add(y_add)
    idx += 1
mobility_names = ['mobility1', 'mobility2', 'mobility3', 'mobility4']
for i in range(n_mobility):
    layers = []
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense0'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense1'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(1, name=names[i] + '_out'))
    add_elems = []
    for j in range(4):
        x[idx] = Input(shape=8 * 2, name=mobility_names[i] + '_in_' + str(j))
        tmp = x[idx]
        for layer in layers:
            tmp = layer(tmp)
        add_elems.append(tmp)
        idx += 1
    y_before_add.extend(add_elems)
y = Add()(y_before_add)
model = Model(inputs=x, outputs=y)

#model = load_model('learned_data/bef_' + str(stone_strt) + '_' + str(stone_end) + '.h5')

#model.summary()
#plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='mse', metrics='mae', optimizer='adam')






# input index data
all_data_idx = [[] for _ in range(n_raw_data)]
all_labels_raw = []
for file in input_files:
    with open(file, 'rb') as f:
        while True:
            if len(all_labels_raw) % 10000 == 0:
                print(len(all_labels_raw))
            try:
                b = f.read(4)
                n_stones = int.from_bytes(b, byteorder='big', signed=True)
                phase = (n_stones - 4) // ply_d
                if phase == use_phase:
                    for i in range(n_raw_data):
                        b = f.read(4)
                        all_data_idx[i].append(int.from_bytes(b, byteorder='big', signed=True))
                    b = f.read(4)
                    all_labels_raw.append(int.from_bytes(b, byteorder='big', signed=True))
                else:
                    f.read(4 * (n_raw_data + 1))
            except:
                break











# translate to input array
def idx2pattern(pattern_idx, idx):
    pattern_size = feature_sizes[pattern_nums[pattern_idx]]
    pattern_unzipped = [0.0 for _ in range(pattern_size * 2)]
    for cell in range(feature_sizes[pattern_idx]):
        elem = idx % 3
        if elem == 0:
            pattern_unzipped[cell] = 1.0
        elif elem == 1:
            pattern_unzipped[pattern_size + cell] = 1.0
        elem //= 3

def idx2mobility(idx):
    pattern_unzipped = [0.0 for _ in range(8 * 2)]
    for cell in range(8 * 2):
        elem = idx % 2
        if elem == 1:
            pattern_unzipped[cell] = 1.0
        elem //= 2

all_data = [[] for _ in range(n_raw_data)]
all_labels = []
for data_idx in range(len(all_labels_raw)):
    for pattern_idx in range(n_raw_patterns):
        idx = all_data_idx[pattern_idx][data_idx]
        pattern_unzipped = idx2pattern(pattern_idx, idx)
        all_data[pattern_idx].append(pattern_unzipped)
    for additional_feature_idx in range(n_additional_features):
        additional_feature = [all_data_idx[n_raw_patterns + additional_feature_idx * 2], all_data_idx[n_raw_patterns + additional_feature_idx * 2 + 1]]
        all_data[n_raw_patterns + additional_feature_idx].append(additional_feature)
    for mobility_idx in range(n_raw_mobility):
        idx = all_data_idx[n_raw_patterns + n_additional_features * 2 + mobility_idx]
        pattern_unzipped = idx2mobility(idx)
        all_data[n_raw_patterns + n_additional_features + mobility_idx].append(idx)
    score = all_labels_raw[data_idx] / 64
    all_labels.append(score)





# learn
len_data = len(all_labels)
print('len_data', len_data)

all_data = [np.array(arr) for arr in all_data]
all_labels = np.array(all_labels)
print('converted to np arr')

n_train_data = int(len_data * (1.0 - test_ratio))
n_test_data = int(len_data * test_ratio)
print('train', n_train_data, 'test', n_test_data)

test_idxes = set(sample(range(len_data), n_test_data))
train_idxes = set(range(len_data)) - test_idxes
test_idxes = list(test_idxes)
train_idxes = list(train_idxes)


train_data = [arr[train_idxes] for arr in all_data]
train_labels = all_labels[train_idxes]
test_data = [arr[test_idxes] for arr in all_data]
test_labels = all_labels[test_idxes]

early_stop = EarlyStopping(monitor='val_loss', patience=5)
#model_checkpoint = ModelCheckpoint(filepath=os.path.join('learned_data/' + str(phase), 'model_{epoch:02d}_{val_loss:.5f}_{val_mae:.5f}.h5'), monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001)
history = model.fit(train_data, train_labels, epochs=n_epochs, validation_data=(test_data, test_labels), callbacks=[early_stop, reduce_lr])

#now = datetime.datetime.today()
#print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
model.save('learned_data/' + str(phase) + '.h5')

for key in ['loss', 'val_loss']:
    plt.plot(history.history[key], label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('graph/loss_' + str(phase) + '.png')
    plt.clf()



# pre calculation
