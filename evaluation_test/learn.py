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

n_dense_pattern = int(sys.argv[1])
n_dense_additional = int(sys.argv[2])

use_phase = int(sys.argv[3])
ply_d = 2

print('phase', use_phase)

input_files = sys.argv[4:]

n_epochs = 400

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
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense2'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=names[i] + '_dense3'))
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
    x[idx] = Input(shape=2, name=add_names[i] + '_in')
    y_add = Dense(n_dense_additional, name=add_names[i] + '_dense0')(x[idx])
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(n_dense_additional, name=add_names[i] + '_dense1')(y_add)
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(1, name=add_names[i] + '_out')(y_add)
    y_before_add.append(y_add)
    idx += 1
mobility_names = ['mobility1', 'mobility2', 'mobility3', 'mobility4']
for i in range(n_mobility):
    layers = []
    layers.append(Dense(n_dense_pattern, name=mobility_names[i] + '_dense0'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=mobility_names[i] + '_dense1'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=mobility_names[i] + '_dense2'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(n_dense_pattern, name=mobility_names[i] + '_dense3'))
    layers.append(LeakyReLU(alpha=0.01))
    layers.append(Dense(1, name=mobility_names[i] + '_out'))
    add_elems = []
    for j in range(4):
        x[idx] = Input(shape=8 * 2, name=mobility_names[i] + '_in_' + str(j))
        tmp = x[idx]
        for layer in layers:
            tmp = layer(tmp)
        add_elems.append(tmp)
        idx += 1
    y_before_add.extend(add_elems)
y = Add(name='last_layer')(y_before_add)
model = Model(inputs=x, outputs=y)

#model = load_model('learned_data/' + str(use_phase) + '_' + str(n_dense_pattern) + '.h5')

#model.summary()
#exit()


#model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
print('n_params', model.count_params())

model.compile(loss='mse', metrics='mae', optimizer='adam')





# input index data
all_data_idx = [[] for _ in range(n_raw_data_input)]
all_labels_raw = []
score_arr = [0 for _ in range(129)]
for file in input_files:
    with open(file, 'rb') as f:
        while True:
            if len(all_labels_raw) % 10000 == 0:
                print('\r', len(all_labels_raw), end='')
            if len(all_labels_raw) >= 1000000:
                break
            b = f.read(4)
            if not b:
                break
            n_stones = int.from_bytes(b, byteorder='little', signed=False)
            #print(n_stones)
            phase = (n_stones - 4) // ply_d
            b = f.read(4)
            if phase == use_phase:
                for i in range(n_raw_data_input):
                    b = f.read(4)
                    all_data_idx[i].append(int.from_bytes(b, byteorder='little', signed=False))
                b = f.read(4)
                score = int.from_bytes(b, byteorder='little', signed=True)
                all_labels_raw.append(score)
                score_arr[score + 64] += 1
            else:
                f.read(4 * (n_raw_data_input + 1))

#print(all_data_idx)
#print(all_labels_raw)

#plt.plot(range(-64, 65), score_arr)
#plt.show()
#plt.clf()








# translate to input array
all_data = [np.zeros((len(all_labels_raw), input_sizes_raw[i])) for i in range(n_raw_data)]
all_labels = np.zeros(len(all_labels_raw))
for data_idx in trange(len(all_labels_raw)):
    for pattern_idx in range(n_raw_patterns):
        idx = all_data_idx[pattern_idx][data_idx]
        pattern_unzipped = idx2pattern(pattern_idx, idx)
        #if pattern_idx == 0 and idx == 0:
        #    print(pattern_unzipped)
        all_data[pattern_idx][data_idx] = np.array(pattern_unzipped)
    for additional_feature_idx in range(n_additional_features):
        additional_feature = [all_data_idx[n_raw_patterns + additional_feature_idx * 2][data_idx], all_data_idx[n_raw_patterns + additional_feature_idx * 2 + 1][data_idx]]
        all_data[n_raw_patterns + additional_feature_idx][data_idx] = np.array(additional_feature)
    for mobility_idx in range(n_raw_mobility):
        idx = all_data_idx[n_raw_patterns + n_additional_features * 2 + mobility_idx][data_idx]
        pattern_unzipped = idx2mobility(idx)
        all_data[n_raw_patterns + n_additional_features + mobility_idx][data_idx] = np.array(pattern_unzipped)
    all_labels[data_idx] = all_labels_raw[data_idx] * 256

#print(all_data)
#print(all_labels)

#exit()

#print([all_data[i].shape for i in range(n_raw_data)], all_labels.shape)
#for i in range(n_raw_data):
#    print(i, all_data[i][0])
#exit()


# learn
len_data = len(all_labels)
print('len_data', len_data)

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

early_stop = EarlyStopping(monitor='val_loss', patience=10)
#model_checkpoint = ModelCheckpoint(filepath=os.path.join('learned_data/' + str(phase), 'model_{epoch:02d}_{val_loss:.5f}_{val_mae:.5f}.h5'), monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.0001)
history = model.fit(train_data, train_labels, epochs=n_epochs, batch_size=16384, validation_data=(test_data, test_labels), callbacks=[early_stop, reduce_lr])

#now = datetime.datetime.today()
#print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
model.save('learned_data/' + str(use_phase) + '_' + str(n_dense_pattern) + '.h5')

plt.xlabel('epoch')
plt.ylabel('loss')
for key in ['loss', 'val_loss']:
    plt.plot(history.history[key], label=key)
    plt.legend(loc='best')
plt.savefig('graph/loss_' + str(use_phase) + '_' + str(n_dense_pattern) + '.png')
plt.clf()

plt.xlabel('epoch')
plt.ylabel('mae')
for key in ['mae', 'val_mae']:
    plt.plot(history.history[key], label=key)
    plt.legend(loc='best')
plt.savefig('graph/mae_' + str(use_phase) + '_' + str(n_dense_pattern) + '.png')
plt.clf()

with open('learned_data/learn_log.txt', 'a') as f:
    f.write(str(history.history['loss'][len(history.history['loss']) - 1]))
    f.write('\t')
    f.write(str(history.history['mae'][len(history.history['mae']) - 1]))
    f.write('\n')







# pre calculation
pre_calc_data = [np.zeros((65536, input_sizes_raw[i]), float) for i in range(n_raw_data)]
for idx in trange(65536):
    for pattern_idx in range(n_patterns):
        if idx < feature_actual_sizes[pattern_idx]:
            pattern_unzipped = idx2pattern(feature_idxes[pattern_idx], idx)
            pre_calc_data[feature_idxes[pattern_idx]][idx] = np.array(pattern_unzipped)
    for additional_feature in range(n_additional_features):
        if idx < feature_actual_sizes[n_patterns + additional_feature]:
            pre_calc_data[feature_idxes[n_patterns + additional_feature]][idx] = np.array([idx // additional_feature_mul[additional_feature], idx % additional_feature_mul[additional_feature]])
    for mobility_idx in range(n_mobility):
        pattern_unzipped = idx2mobility(idx)
        pre_calc_data[feature_idxes[n_patterns + n_additional_features + mobility_idx]][idx] = np.array(pattern_unzipped)

last_layer_model = Model(inputs=model.input, outputs=model.get_layer('last_layer').input)
predictions = last_layer_model.predict(pre_calc_data, batch_size=16384)
print(len(predictions), len(predictions[0]))
print([predictions[feature_idxes[pattern_idx]][101][0] / 256 for pattern_idx in range(24)])
#print([predictions[i][0][0] * 64 for i in range(len(predictions))])
step = 256
n_lines = 0
plus_elem = 0
with open('learned_data/' + str(use_phase) + '_' + str(n_dense_pattern) + '.txt', 'w') as f:
    for pattern_idx in range(24):
        for i in range(feature_actual_sizes[pattern_idx]):
            val = round(predictions[feature_idxes[pattern_idx]][i][0])
            if predictions[feature_idxes[pattern_idx]][i][0] > 0.0:
                plus_elem += 1
            f.write(str(val) + '\n')
            n_lines += 1
print('done', n_lines, plus_elem)


'''
# test prediction
pred_in_idx = [[5874], [1718], [4085], [5921], [3168], [4748], [3510], [3161], [2433], [2231], [2538], [2957], [120], [91], [107], [189], [486], [334], [284], [678], [2], [1714], [1820], [1809], [6110], [6128], [54323], [57959], [59048], [56870], [45130], [55593], [58968], [58765], [55813], [49276], [59016], [39537], [54697], [55242], [58198], [53748], [15241], [18415], [19400], [19602], [14956], [38275], [39363], [17502], [44941], [55701], [59022], [58954], [45234], [54282], [56584], [58402], [45234], [54273], [56574], [58456], [11], [14], [11], [7], [0], [0], [21], [23], [1628], [8194], [0], [64], [592], [17090], [17344], [16451], [8224], [0], [257], [33796], [0], [0], [1], [3]]
pred_in_idx = [[arr[0]] for arr in pred_in_idx]
all_data = [np.zeros((len(pred_in_idx[0]), input_sizes_raw[i])) for i in range(n_raw_data)]
for data_idx in range(len(pred_in_idx[0])):
    for pattern_idx in range(n_raw_patterns):
        idx = pred_in_idx[pattern_idx][data_idx]
        pattern_unzipped = idx2pattern(pattern_idx, idx)
        #if pattern_idx == 0 and idx == 0:
        #    print(pattern_unzipped)
        all_data[pattern_idx][data_idx] = np.array(pattern_unzipped)
    for additional_feature_idx in range(n_additional_features):
        additional_feature = [pred_in_idx[n_raw_patterns + additional_feature_idx * 2][data_idx] / additional_feature_mul[additional_feature_idx], pred_in_idx[n_raw_patterns + additional_feature_idx * 2 + 1][data_idx] / additional_feature_mul[additional_feature_idx]]
        all_data[n_raw_patterns + additional_feature_idx][data_idx] = np.array(additional_feature)
    for mobility_idx in range(n_raw_mobility):
        idx = pred_in_idx[n_raw_patterns + n_additional_features * 2 + mobility_idx][data_idx]
        pattern_unzipped = idx2mobility(idx)
        all_data[n_raw_patterns + n_additional_features + mobility_idx][data_idx] = np.array(pattern_unzipped)

pred_test = model.predict(all_data)
print(pred_test, pred_test[0][0] * 64)

for data_idx in range(len(pred_in_idx[0])):
    score = 0
    for pattern_idx in range(n_raw_patterns):
        idx = pred_in_idx[pattern_idx][data_idx]
        score += predictions[feature_idxes[pattern_nums[pattern_idx]]][idx][0]
        #print(predictions[feature_idxes[pattern_nums[pattern_idx]]][idx][0] * 64, end=' ')
    for additional_feature_idx in range(n_additional_features):
        idx = pred_in_idx[n_raw_patterns + additional_feature_idx * 2][data_idx] * additional_feature_mul[additional_feature_idx] + pred_in_idx[n_raw_patterns + additional_feature_idx * 2 + 1][data_idx]
        score += predictions[feature_idxes[n_patterns + additional_feature_idx]][idx][0]
        #print(predictions[feature_idxes[n_patterns + additional_feature_idx]][idx][0] * 64, end=' ')
    for mobility_idx in range(n_raw_mobility):
        idx = pred_in_idx[n_raw_patterns + n_additional_features * 2 + mobility_idx][data_idx]
        score += predictions[feature_idxes[n_patterns + n_additional_features + mobility_idx // 4]][idx][0]
        #print(predictions[feature_idxes[n_patterns + n_additional_features + mobility_idx // 4]][idx][0] * 64, end=' ')
    print('')
    print(score, score * 64)
'''