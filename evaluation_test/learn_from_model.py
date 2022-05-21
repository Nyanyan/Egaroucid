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
from random import random, randrange, sample, shuffle
import subprocess
import datetime
import os
from math import tanh, log
from copy import deepcopy
import sys
import datetime

# original code: https://qiita.com/rhene/items/459c2f6b07d5e851efc0
class DisplayCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.last_mae, self.last_loss = None, None
        self.now_batch, self.now_epoch = None, None

    def print_progress(self):
        epoch = self.now_epoch
        epochs = self.epochs
        print("\rEpoch %d/%d -- mae: %f loss: %f" % (epoch+1, epochs, self.last_mae, self.last_loss), end='')

    def on_train_begin(self, logs={}):
        print('##### Train Start ##### ' + str(datetime.datetime.now()))
        self.epochs = self.params['epochs']
        self.params['verbose'] = 0

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_begin(self, epoch, log={}):
        self.now_epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        self.last_mae = logs.get('mae') if logs.get('mae') else 0.0
        self.last_loss = logs.get('loss') if logs.get('loss') else 0.0
        self.print_progress()

    def on_train_end(self, logs={}):
        print('\n##### Train Complete ##### ' + str(datetime.datetime.now()))

#use_phase = int(sys.argv[3])
ply_d = 2

n_epochs = 10000

inf = 10000000.0

ply_d = 2

test_ratio = 0.2

n_patterns = 16
n_additional_features = 4
n_mobility = 4

n_raw_data = 82
n_raw_data_input = 86


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
n_pattern_varieties = [4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4]

additional_feature_mul = [100, 50, 65, 65]

feature_actual_sizes = [
    3 ** 8, 3 ** 8, 3 ** 8, 3 ** 5, 3 ** 6, 3 ** 7, 3 ** 8, 3 ** 10, 3 ** 10, 3 ** 10, 3 ** 10, 3 ** 9, 3 ** 10, 3 ** 10, 3 ** 10, 3 ** 10, 
    100 * 100, 50 * 50, 65 * 65, 65 * 65,
    2 ** 16, 2 ** 16, 2 ** 16, 2 ** 16
]

feature_idxes = [
    0, 4, 8, 12, 16, 20, 24, 26, 30, 34, 38, 42, 46, 50, 54, 58, 
    62, 63, 64, 65, 
    66, 70, 74, 78, 
    82
]

input_sizes = [16, 16, 16, 10, 12, 14, 16, 20, 20, 20, 20, 18, 20, 20, 20, 20, 2, 2, 2, 2, 16, 16, 16, 16]

input_sizes_raw = []
for i in range(len(input_sizes)):
    for _ in range(n_pattern_varieties[i]):
        input_sizes_raw.append(input_sizes[i])


def idx2pattern(pattern_idx, idx):
    pattern_size = feature_sizes[pattern_nums[pattern_idx]]
    pattern_unzipped = [0.0 for _ in range(pattern_size * 2)]
    for cell in range(pattern_size):
        elem = idx % 3
        if elem == 0:
            pattern_unzipped[cell] = 1.0
        elif elem == 1:
            pattern_unzipped[pattern_size + cell] = 1.0
        idx //= 3
    return pattern_unzipped

def idx2pattern2(pattern_idx, idx):
    pattern_size = feature_sizes[pattern_idx]
    pattern_unzipped = [0.0 for _ in range(pattern_size * 2)]
    for cell in range(pattern_size):
        elem = idx % 3
        if elem == 0:
            pattern_unzipped[cell] = 1.0
        elif elem == 1:
            pattern_unzipped[pattern_size + cell] = 1.0
        idx //= 3
    return pattern_unzipped

def idx2mobility(idx):
    pattern_unzipped = [0.0 for _ in range(8 * 2)]
    for cell in range(8 * 2):
        elem = idx % 2
        if elem == 1:
            pattern_unzipped[cell] = 1.0
        idx //= 2
    return pattern_unzipped

def create_input_feature(feature_idx, idx):
    if feature_idx < 16:
        return idx2pattern2(feature_idx, idx)
    elif feature_idx < 20:
        return [idx // additional_feature_mul[feature_idx - 16], idx % additional_feature_mul[feature_idx - 16]]
    else:
        return idx2mobility(idx)

n_params = 0

n_denses = [
    64, 64, 64, 64, 
    64, 64, 64, 256, 
    256, 128, 128, 128, 
    128, 128, 128, 128,
    16, 16, 16, 16, 
    128, 128, 128, 128
]

for feature_idx in range(24):
    x = Input(shape=input_sizes[feature_idx], name='')
    y = Dense(n_denses[feature_idx], name='dense0')(x)
    y = LeakyReLU(alpha=0.01)(y)
    y = Dense(n_denses[feature_idx], name='dense1')(y)
    y = LeakyReLU(alpha=0.01)(y)
    y = Dense(1, name='out')(y)
    model = Model(inputs=x, outputs=y)
    n_params += model.count_params()

print('n_params', n_params, n_params * 30)

for use_phase in reversed(range(23)):
    with open('data/' + str(use_phase) + '.txt', 'r') as f:
        all_labels = [int(elem) for elem in f.read().splitlines()]

    with open('data/' + str(use_phase) + '_count.txt', 'r') as f:
        all_weights = [int(elem) for elem in f.read().splitlines()]

    data_strt_idx = 0

    for feature_idx in range(24):
        print('phase', use_phase, 'feature', feature_idx)
        x = Input(shape=input_sizes[feature_idx], name='')
        y = Dense(n_denses[feature_idx], name='dense0')(x)
        y = LeakyReLU(alpha=0.01)(y)
        y = Dense(n_denses[feature_idx], name='dense1')(y)
        y = LeakyReLU(alpha=0.01)(y)
        y = Dense(1, name='out')(y)
        model = Model(inputs=x, outputs=y)
        model.compile(loss='mse', metrics='mae', optimizer='adam')
        train_data_tmp = []
        train_labels_tmp = []
        train_weights_tmp = []
        for i in range(data_strt_idx, data_strt_idx + feature_actual_sizes[feature_idx]):
            if all_weights[i] > 0:
                train_data_tmp.append(i - data_strt_idx)
                train_labels_tmp.append(all_labels[i])
                train_weights_tmp.append(all_weights[i])
        train_data = np.zeros((len(train_data_tmp), input_sizes[feature_idx]))
        train_labels = np.zeros(len(train_data_tmp))
        train_weights = np.zeros(len(train_data_tmp))
        max_train_weight = max(train_weights_tmp)
        for i in range(len(train_data_tmp)):
            train_data[i] = np.array(create_input_feature(feature_idx, train_data_tmp[i]))
            train_labels[i] = train_labels_tmp[i]
            train_weights[i] = max(0.001, train_weights_tmp[i] / max_train_weight)
        
        early_stop = EarlyStopping(monitor='loss', patience=100)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        cbDisplay = DisplayCallBack()
        history = model.fit(train_data, train_labels, sample_weight=train_weights, epochs=n_epochs, batch_size=16384, validation_split=0.0, verbose=0, callbacks=[early_stop, reduce_lr, cbDisplay])
        #history = model.fit(train_data, train_labels, epochs=n_epochs, batch_size=16384, validation_split=0.0, verbose=0, callbacks=[early_stop, reduce_lr, cbDisplay])
        #with open('learned_data/log.txt', 'a') as f:
        #    f.write(str(model.evaluate(train_data, train_labels)) + '\n')
        #model.save('learned_data/' + str(use_phase) + '_' + str(n_dense_pattern) + '_' + str(feature_idx) + '.h5')
        
        predict_data = np.zeros((feature_actual_sizes[feature_idx], input_sizes[feature_idx]))
        for i in range(feature_actual_sizes[feature_idx]):
            predict_data[i] = np.array(create_input_feature(feature_idx, i))
        prediction = model.predict(predict_data, batch_size=8192)
        print(prediction.shape)
        with open('learned_data/' + str(use_phase) + '_model.txt', 'a') as f:
            for i in range(feature_actual_sizes[feature_idx]):
                f.write(str(round(prediction[i][0])) + '\n')
        
        data_strt_idx += feature_actual_sizes[feature_idx]
        print('')

    print('')