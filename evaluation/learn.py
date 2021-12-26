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

inf = 10000000.0

test_ratio = 0.1
n_epochs = 300


line2_idx = [[8, 9, 10, 11, 12, 13, 14, 15], [1, 9, 17, 25, 33, 41, 49, 57], [6, 14, 22, 30, 38, 46, 54, 62], [48, 49, 50, 51, 52, 53, 54, 55]] # line2
for pattern in deepcopy(line2_idx):
    line2_idx.append(list(reversed(pattern)))

line3_idx = [[16, 17, 18, 19, 20, 21, 22, 23], [2, 10, 18, 26, 34, 42, 50, 58], [5, 13, 21, 29, 37, 45, 53, 61], [40, 41, 42, 43, 44, 45, 46, 47]]
for pattern in deepcopy(line3_idx):
    line3_idx.append(list(reversed(pattern)))

line4_idx = [[24, 25, 26, 27, 28, 29, 30, 31], [3, 11, 19, 27, 35, 43, 51, 59], [4, 12, 20, 28, 36, 44, 52, 60], [32, 33, 34, 35, 36, 37, 38, 39]]
for pattern in deepcopy(line4_idx):
    line4_idx.append(list(reversed(pattern)))

diagonal5_idx = [[4, 11, 18, 25, 32], [24, 33, 42, 51, 60], [59, 52, 45, 38, 31], [39, 30, 21, 12, 3]]
for pattern in deepcopy(diagonal5_idx):
    diagonal5_idx.append(list(reversed(pattern)))

diagonal6_idx = [[5, 12, 19, 26, 33, 40], [16, 25, 34, 43, 52, 61], [58, 51, 44, 37, 30, 23], [47, 38, 29, 20, 11, 2]]
for pattern in deepcopy(diagonal6_idx):
    diagonal6_idx.append(list(reversed(pattern)))

diagonal7_idx = [[1, 10, 19, 28, 37, 46, 55], [48, 41, 34, 27, 20, 13, 6], [62, 53, 44, 35, 26, 17, 8], [15, 22, 29, 36, 43, 50, 57]]
for pattern in deepcopy(diagonal7_idx):
    diagonal7_idx.append(list(reversed(pattern)))

diagonal8_idx = [[0, 9, 18, 27, 36, 45, 54, 63], [7, 14, 21, 28, 35, 42, 49, 56]]
for pattern in deepcopy(diagonal8_idx):
    diagonal8_idx.append(list(reversed(pattern)))

edge_2x_idx = [[9, 0, 1, 2, 3, 4, 5, 6, 7, 14], [9, 0, 8, 16, 24, 32, 40, 48, 56, 49], [49, 56, 57, 58, 59, 60, 61, 62, 63, 54], [54, 63, 55, 47, 39, 31, 23, 15, 7, 14]]
for pattern in deepcopy(edge_2x_idx):
    edge_2x_idx.append(list(reversed(pattern)))

triangle_idx = [
    [0, 1, 2, 3, 8, 9, 10, 16, 17, 24], 
    [7, 6, 5, 4, 15, 14, 13, 23, 22, 31], 
    [63, 62, 61, 60, 55, 54, 53, 47, 46, 39], 
    [56, 57, 58, 59, 48, 49, 50, 40, 41, 32], 
    [0, 8, 16, 24, 1, 9, 17, 2, 10, 3], 
    [7, 15, 23, 31, 6, 14, 22, 5, 13, 4], 
    [63, 55, 47, 39, 62, 54, 46, 61, 53, 60],
    [56, 48, 40, 32, 57, 49, 41, 58, 50, 59]
]

corner25_idx = [
    [0, 1, 2, 3, 4, 8, 9, 10, 11, 12],[0, 8, 16, 24, 32, 1, 9, 17, 25, 33],
    [7, 6, 5, 4, 3, 15, 14, 13, 12, 11],[7, 15, 23, 31, 39, 6, 14, 22, 30, 38],
    [56, 57, 58, 59, 60, 48, 49, 50, 51, 52],[56, 48, 40, 32, 24, 57, 49, 41, 33, 25],
    [63, 62, 61, 60, 59, 55, 54, 53, 52, 51],[63, 55, 47, 39, 31, 62, 54, 46, 38, 30]
]

center16_idx = [
    [18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45],
    [21, 20, 19, 18, 29, 28, 27, 26, 37, 36, 35, 34, 45, 44, 43, 42],
    [18, 26, 34, 42, 19, 27, 35, 43, 20, 28, 36, 44, 21, 29, 37, 45],
    [21, 29, 37, 45, 20, 28, 36, 44, 19, 27, 35, 43, 18, 26, 34, 42]
]
for pattern in deepcopy(center16_idx):
    center16_idx.append(list(reversed(pattern)))

corner9_idx = [
    [0, 1, 2, 8, 9, 10, 16, 17, 18], 
    [7, 6, 5, 15, 14, 13, 23, 22, 21], 
    [56, 57, 58, 48, 49, 50, 40, 41, 42], 
    [63, 62, 61, 55, 54, 53, 47, 46, 45], 
    [0, 8, 16, 1, 9, 17, 2, 10, 18],
    [7, 15, 23, 6, 14, 22, 5, 13, 21],
    [63, 55, 47, 62, 54, 46, 61, 53, 45]
]

edge_block = [
    [0, 2, 3, 4, 5, 7, 10, 11, 12, 13], 
    [0, 16, 24, 32, 40, 56, 17, 25, 33, 41], 
    [56, 58, 59, 60, 61, 63, 50, 51, 52, 53], 
    [7, 23, 31, 39, 47, 63, 22, 30, 38, 46], 
    [7, 5, 4, 3, 2, 0, 13, 12, 11, 10],
    [56, 40, 32, 24, 16, 0, 41, 33, 25, 17],
    [63, 61, 60, 59, 58, 56, 53, 52, 51, 50],
    [63, 47, 39, 31, 23, 7, 46, 38, 30, 22]
]

cross_idx = [
    [0, 9, 18, 27, 1, 10, 19, 8, 17, 26], 
    [7, 14, 21, 28, 6, 13, 20, 15, 22, 29], 
    [56, 49, 42, 35, 57, 50, 43, 48, 41, 34], 
    [63, 54, 45, 36, 62, 53, 44, 55, 46, 37], 
    [0, 9, 18, 27, 8, 17, 26, 1, 10, 19],
    [7, 14, 21, 28, 15, 22, 29, 6, 13, 20],
    [56, 49, 42, 35, 48, 41, 34, 57, 50, 43],
    [63, 54, 45, 36, 55, 46, 37, 62, 53, 44]
]

edge_2y_idx = [[10, 0, 1, 2, 3, 4, 5, 6, 7, 13], [17, 0, 8, 16, 24, 32, 40, 48, 56, 41], [50, 56, 57, 58, 59, 60, 61, 62, 63, 53], [46, 63, 55, 47, 39, 31, 23, 15, 7, 22]]
for pattern in deepcopy(edge_2y_idx):
    edge_2y_idx.append(list(reversed(pattern)))

narrow_triangle_idx = [
    [0, 1, 2, 3, 4, 8, 9, 16, 24, 32], 
    [7, 6, 5, 4, 3, 15, 14, 23, 31, 39], 
    [63, 62, 61, 60, 59, 55, 54, 47, 39, 31], 
    [56, 57, 58, 59, 60, 48, 49, 40, 32, 24], 
    [0, 8, 16, 24, 32, 1, 9, 2, 3, 4],
    [7, 15, 23, 31, 39, 6, 14, 5, 4, 3],
    [63, 55, 47, 39, 31, 62, 54, 61, 60, 59],
    [56, 48, 40, 32, 24, 57, 49, 58, 59, 60]
]

middle_idx = [
    [42, 43, 44, 45, 49, 50, 51, 52, 53, 54],
    [21, 29, 37, 45, 14, 22, 30, 38, 46, 54],
    [18, 19, 20, 21, 9, 10, 11, 12, 13, 14],
    [18, 26, 34, 42, 9, 17, 25, 33, 41, 49],
    [45, 44, 43, 42, 54, 53, 52, 51, 50, 49],
    [45, 37, 29, 21, 54, 46, 38, 30, 22, 14],
    [21, 20, 19, 18, 14, 13, 12, 11, 10, 9],
    [42, 34, 26, 18, 49, 41, 33, 25, 17, 9]
]

pattern_idx = [line2_idx, line3_idx, line4_idx, diagonal5_idx, diagonal6_idx, diagonal7_idx, diagonal8_idx, edge_2x_idx, triangle_idx, edge_block, cross_idx, corner9_idx, edge_2y_idx]
ln_in = sum([len(elem) for elem in pattern_idx]) + 1

# [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
# [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
# [0, 10, 20, 30, 40, 50]

for stone_strt in reversed([0, 4, 8, 12, 16, 20, 24, 28, 32, 36]):
    stone_end = stone_strt + 4

    min_n_stones = 4 + stone_strt
    max_n_stones = 4 + stone_end

    all_data = [[] for _ in range(ln_in)]
    all_labels = []

    def make_lines(board, patterns, player):
        res = []
        for pattern in patterns:
            tmp = []
            black_num = 0
            white_num = 0
            for elem in pattern:
                tmp.append(1.0 if board[elem] == str(player) else 0.0)
                black_num += 1.0 if board[elem] == str(player) else 0.0
            for elem in pattern:
                tmp.append(1.0 if board[elem] == str(1 - player) else 0.0)
                white_num += 1.0 if board[elem] == str(1 - player) else 0.0
            #tmp.append((black_num - 5) / 5)
            #tmp.append((white_num - 5) / 5)
            #tmp.append((black_num + white_num - 10) / 10)
            res.append(tmp)
        return res

    def digit(n, r):
        n = str(n)
        l = len(n)
        for i in range(r - l):
            n = '0' + n
        return n

    def calc_n_stones(board):
        res = 0
        for elem in board:
            if elem != '.':
                res += 1
        return res

    def collect_data(directory, num):
        global all_data, all_labels
        try:
            with open('data/' + directory + '/' + digit(num, 7) + '.txt', 'r') as f:
                data = list(f.read().splitlines())
        except:
            print('cannot open')
            return
        for datum in data:
            board, player, v1, v2, v3, result = datum.split()
            player = int(player)
            n_stones = calc_n_stones(board)
            if min_n_stones <= n_stones < max_n_stones: # and player == black_white:
                v1 = float(v1)
                v2 = float(v2)
                v3 = float(v3)
                result = float(result)
                result = result / 64
                idx = 0
                for i in range(len(pattern_idx)):
                    lines = make_lines(board, pattern_idx[i], 0)
                    for line in lines:
                        all_data[idx].append(line)
                        idx += 1
                all_data[idx].append([(v2 - 20) / 20, (v3 - 20) / 20])
                all_labels.append(result)

    x = [None for _ in range(ln_in)]
    ys = []
    names = ['line2', 'line3', 'line4', 'diagonal5', 'diagonal6', 'diagonal7', 'diagonal8', 'edge2X', 'triangle', 'edgeblock', 'cross', 'corner9', 'edge2Y']
    idx = 0
    for i in range(len(pattern_idx)):
        layers = []
        layers.append(Dense(256, name=names[i] + '_dense0'))
        layers.append(LeakyReLU(alpha=0.01))
        layers.append(Dense(128, name=names[i] + '_dense1'))
        layers.append(LeakyReLU(alpha=0.01))
        layers.append(Dense(128, name=names[i] + '_dense2'))
        layers.append(LeakyReLU(alpha=0.01))
        layers.append(Dense(1, name=names[i] + '_out'))
        layers.append(Activation('tanh'))
        add_elems = []
        for j in range(len(pattern_idx[i])):
            x[idx] = Input(shape=len(pattern_idx[i][0]) * 2, name=names[i] + '_in_' + str(j))
            tmp = x[idx]
            for layer in layers:
                tmp = layer(tmp)
            add_elems.append(tmp)
            idx += 1
        #tmp = Add(name=names[i] + '_pre_prediction')(add_elems)
        #tmp = LeakyReLU(alpha=0.01)(tmp)
        #ys.append(tmp)
        ys.append(Add(name=names[i] + '_pre_prediction')(add_elems))
    x[idx] = Input(shape=2, name='additional_input')
    y_add = Dense(8, name='add_dense0')(x[idx])
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(8, name='add_dense1')(y_add)
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(8, name='add_dense2')(y_add)
    y_add = LeakyReLU(alpha=0.01)(y_add)
    y_add = Dense(1, name='add_dense3')(y_add)
    y_add = Activation('tanh')(y_add)
    #y_add = LeakyReLU(alpha=0.01)(y_add)
    ys.append(y_add)
    y_all = Add()(ys)
    '''
    y_all = Concatenate(axis=-1)(ys)
    y_all = Dense(16, name='final_dense0')(y_all)
    y_all = LeakyReLU(alpha=0.01)(y_all)
    y_all = Dense(1, name='final_dense1')(y_all)
    '''
    model = Model(inputs=x, outputs=y_all)

    #model = load_model('learned_data/f_' + str(stone_strt) + '_' + str(stone_end) + '.h5')

    #model.summary()
    plot_model(model, to_file='learned_data/model.png', show_shapes=True)

    model.compile(loss='mse', metrics='mae', optimizer='adam')

    for i in trange(263):
        collect_data('records3', i)
    len_data = len(all_labels)
    print(len_data)
    
    all_data = [np.array(arr) for arr in all_data]
    all_labels = np.array(all_labels)
    print('converted to numpy arr')
    
    p = np.random.permutation(len_data)
    all_data = [arr[p] for arr in all_data]
    all_labels = all_labels[p]

    n_train_data = int(len_data * (1.0 - test_ratio))
    n_test_data = int(len_data * test_ratio)

    train_data = [arr[0:n_train_data] for arr in all_data]
    train_labels = all_labels[0:n_train_data]
    test_data = [arr[n_train_data:len_data] for arr in all_data]
    test_labels = all_labels[n_train_data:len_data]


    #print(model.evaluate(all_data, all_labels))
    early_stop = EarlyStopping(monitor='val_loss', patience=15)
    model_checkpoint = ModelCheckpoint(filepath=os.path.join('learned_data/' + str(stone_strt) + '_' + str(stone_end), 'model_{epoch:02d}_{val_loss:.5f}_{val_mae:.5f}.h5'), monitor='val_loss', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    history = model.fit(train_data, train_labels, epochs=n_epochs, batch_size=2048, validation_data=(test_data, test_labels), callbacks=[early_stop, reduce_lr])

    now = datetime.datetime.today()
    print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
    model.save('learned_data/' + str(stone_strt) + '_' + str(stone_end) + '.h5')

    for key in ['loss', 'val_loss']:
        plt.plot([history.history[key][i] for i in range(1, len(history.history[key]))], label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('learned_data/loss_' + str(stone_strt) + '_' + str(stone_end) + '.png')
    plt.clf()
    
    for key in ['mae', 'val_mae']:
        plt.plot([history.history[key][i] for i in range(1, len(history.history[key]))], label=key)
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.legend(loc='best')
    plt.savefig('learned_data/mae_' + str(stone_strt) + '_' + str(stone_end) + '.png')
    plt.clf()
    
    with open('learned_data/result.txt', 'a') as f:
        f.write(str(stone_strt) + '_' + str(stone_end) + '\t' + str(history.history['loss'][-1]) + '\t' + str(history.history['val_loss'][-1]) + '\t' + str(history.history['mae'][-1]) + '\t' + str(history.history['val_mae'][-1]) + '\n')
