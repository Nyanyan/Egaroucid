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
from random import randrange, shuffle
import subprocess
import datetime
import os
from math import tanh, log
from copy import deepcopy


n_patterns = 11
pattern_sizes = [8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10]
max_pattern_size = max(pattern_sizes)
max_predict_num = 152561
n_dense2 = 2
n_add_dense1 = 8
max_canput = 40
max_surround = 60
pattern_in_sizes = [8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8]
all_input_size = 85

def get_layer_index(model, layer_name, not_found=None):
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    return not_found

all_data = [[] for _ in range(all_input_size)]

all_idx = 0
for pattern_idx in range(n_patterns):
    for pattern_elem in range(pow(3, pattern_sizes[pattern_idx])):
        arr = [-1 for _ in range(pattern_sizes[pattern_idx] * 2)]
        for i in range(pattern_sizes[pattern_idx]):
            digit = (pattern_elem // int(pow(3, pattern_sizes[pattern_idx] - 1 - i))) % 3
            if digit == 0:
                arr[i] = 1.0
                arr[pattern_sizes[pattern_idx] + i] = 0.0
            elif digit == 1:
                arr[i] = 0.0
                arr[pattern_sizes[pattern_idx] + i] = 1.0
            else:
                arr[i] = 0.0
                arr[pattern_sizes[pattern_idx] + i] = 0.0
        all_data[all_idx].append(arr)
    for pattern_elem in range(pow(3, pattern_sizes[pattern_idx]), max_predict_num):
        all_data[all_idx].append([0 for _ in range(pattern_sizes[pattern_idx] * 2)])
    for i in range(1, pattern_in_sizes[pattern_idx]):
        for pattern_elem in range(max_predict_num):
            all_data[all_idx + i].append([elem for elem in all_data[all_idx][pattern_elem]])
    all_idx += pattern_in_sizes[pattern_idx]
for canput in range(max_canput + 1):
    for sur0 in range(max_surround + 1):
        for sur1 in range(max_surround + 1):
            all_data[all_idx].append([(canput - 15.0) / 15.0, (sur0 - 15.0) / 15.0, (sur1 - 15.0) / 15.0])

print([len(arr) for arr in all_data])

n_input_cols = len(all_data)
#all_data = [[np.array(all_data[j][i]) for j in range(n_input_cols)] for i in range(max_predict_num)]
all_data = [np.array(arr) for arr in all_data]

names = ['line2', 'line3', 'line4', 'diagonal5', 'diagonal6', 'diagonal7', 'diagonal8', 'edge2X', 'triangle', 'edgeblock', 'cross']

for stone_strt in [0, 10, 20, 30, 40, 50]:
    stone_end = stone_strt + 10
    for black_white in range(2):
        model = load_model('learned_data/' + str(black_white) + '_' + str(stone_strt) + '_' + str(stone_end) + '.h5')

        for pattern_idx in range(len(names)):

            pre_evaluation = Model(inputs=model.input, outputs=model.get_layer(names[pattern_idx] + '_pre_prediction').output)

            prediction = pre_evaluation.predict(all_data)
            
            print(prediction.shape)
            
            print(len(set([prediction[i][0] for i in range(len(prediction))])))
            
            with open('param.txt', 'a') as f:
                for pattern_elem in range(pow(3, pattern_sizes[pattern_idx])):
                    for dense_idx in range(n_dense2):
                        f.write(str(prediction[pattern_elem][dense_idx] / pattern_in_sizes[pattern_idx]) + '\n')

        pre_evaluation = Model(inputs=model.input, outputs=model.get_layer('pre_prediction').output)
        prediction = pre_evaluation.predict(all_data)
        with open('param.txt', 'a') as f:
            all_idx = 0
            for canput in range(max_canput + 1):
                for sur0 in range(max_surround + 1):
                    for sur1 in range(max_surround + 1):
                        for dense_idx in range(n_add_dense1):
                            f.write(str(prediction[all_idx][n_patterns * n_dense2 + dense_idx]) + '\n')
                        all_idx += 1
            
            i = get_layer_index(model, 'all_dense0')
            j = 0
            print(model.layers[i].weights[j].shape)
            for ii in range(model.layers[i].weights[j].shape[0]):
                for jj in range(model.layers[i].weights[j].shape[1]):
                    f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
            j = 1
            print(model.layers[i].weights[j].shape)
            for ii in range(model.layers[i].weights[j].shape[0]):
                f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
