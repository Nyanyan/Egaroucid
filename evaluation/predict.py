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
from tqdm import trange, tqdm
from random import randrange, shuffle
import subprocess
import datetime
import os
from math import tanh, log
from copy import deepcopy

inf = 10000000.0

test_ratio = 0.1
n_epochs = 50

pow3 = [1]
for i in range(11):
    pow3.append(pow3[-1] * 3)

p31 = 3
p32 = 9
p33 = 27
p34 = 81
p35 = 243
p36 = 729
p37 = 2187
p38 = 6561
p39 = 19683
p310 = 59049


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
    [56, 48, 40, 57, 49, 41, 58, 50, 42], 
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

pattern_idx = [line2_idx, line3_idx, line4_idx, diagonal5_idx, diagonal6_idx, diagonal7_idx, diagonal8_idx, edge_2x_idx, triangle_idx, edge_block, cross_idx, corner9_idx, edge_2y_idx]
ln_in = 104 #sum([len(elem) for elem in pattern_idx]) + 1

pattern_sizes = [
    8, 8, 8, 8, 
    8, 8, 8, 8, 
    8, 8, 8, 8, 
    5, 5, 5, 5, 
    6, 6, 6, 6, 
    7, 7, 7, 7, 
    8, 8, 
    10, 10, 10, 10, 
    10, 10, 10, 10, 
    10, 10, 10, 10, 
    10, 10, 10, 10, 
    9, 9, 9, 9, 
    10, 10, 10, 10
]

pattern_idxes = [
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
    12, 12, 12, 12
]

# [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
# [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
# [0, 10, 20, 30, 40, 50]

ml_phase = 0
black_white = 0

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
        res += int(elem != '.')
    return res

def calc_pop(a, b, s):
    return (a // pow3[s - 1 - b]) % 3

def calc_rev_idx(pattern_idx, pattern_size, idx):
    res = 0
    if pattern_idx <= 7 or pattern_idx == 12:
        for i in range(pattern_size):
            res += pow3[i] * calc_pop(idx, i, pattern_size)
    elif pattern_idx == 8:
        res += p39 * calc_pop(idx, 0, pattern_size)
        res += p38 * calc_pop(idx, 4, pattern_size)
        res += p37 * calc_pop(idx, 7, pattern_size)
        res += p36 * calc_pop(idx, 9, pattern_size)
        res += p35 * calc_pop(idx, 1, pattern_size)
        res += p34 * calc_pop(idx, 5, pattern_size)
        res += p33 * calc_pop(idx, 8, pattern_size)
        res += p32 * calc_pop(idx, 2, pattern_size)
        res += p31 * calc_pop(idx, 6, pattern_size)
        res += calc_pop(idx, 3, pattern_size)
    elif pattern_idx == 9:
        res += p39 * calc_pop(idx, 5, pattern_size)
        res += p38 * calc_pop(idx, 4, pattern_size)
        res += p37 * calc_pop(idx, 3, pattern_size)
        res += p36 * calc_pop(idx, 2, pattern_size)
        res += p35 * calc_pop(idx, 1, pattern_size)
        res += p34 * calc_pop(idx, 0, pattern_size)
        res += p33 * calc_pop(idx, 9, pattern_size)
        res += p32 * calc_pop(idx, 8, pattern_size)
        res += p31 * calc_pop(idx, 7, pattern_size)
        res += calc_pop(idx, 6, pattern_size)
    elif pattern_idx == 10:
        res += p39 * calc_pop(idx, 0, pattern_size)
        res += p38 * calc_pop(idx, 1, pattern_size)
        res += p37 * calc_pop(idx, 2, pattern_size)
        res += p36 * calc_pop(idx, 3, pattern_size)
        res += p35 * calc_pop(idx, 7, pattern_size)
        res += p34 * calc_pop(idx, 8, pattern_size)
        res += p33 * calc_pop(idx, 9, pattern_size)
        res += p32 * calc_pop(idx, 4, pattern_size)
        res += p31 * calc_pop(idx, 5, pattern_size)
        res += calc_pop(idx, 6, pattern_size)
    elif pattern_idx == 11:
        res += p38 * calc_pop(idx, 0, pattern_size)
        res += p37 * calc_pop(idx, 3, pattern_size)
        res += p36 * calc_pop(idx, 6, pattern_size)
        res += p35 * calc_pop(idx, 1, pattern_size)
        res += p34 * calc_pop(idx, 4, pattern_size)
        res += p33 * calc_pop(idx, 7, pattern_size)
        res += p32 * calc_pop(idx, 2, pattern_size)
        res += p31 * calc_pop(idx, 5, pattern_size)
        res += calc_pop(idx, 8, pattern_size)
    return res


def make_lines_idx(num, siz, pattern_idx):
    rev_num = calc_rev_idx(pattern_idx, siz, num)
    res = [-1 for _ in range(siz * 2)]
    for i in range(siz):
        dig = num % 3
        if dig == 0:
            res[i] = 1
            res[i + siz] = 0
        elif dig == 1:
            res[i] = 0
            res[i + siz] = 1
        else:
            res[i] = 0
            res[i + siz] = 0
        num //= 3
    num = rev_num
    res2 = [-1 for _ in range(siz * 2)]
    for i in range(siz):
        dig = num % 3
        if dig == 0:
            res2[i] = 1
            res2[i + siz] = 0
        elif dig == 1:
            res2[i] = 0
            res2[i + siz] = 1
        else:
            res2[i] = 0
            res2[i + siz] = 0
        num //= 3
    return [res, res2]
    

def collect_data():
    global all_data, all_labels
    with open('big_data.txt', 'r') as f:
        for _ in trange(10):
            try:
                datum = [int(elem) for elem in f.readline().split()]
                phase = datum[0]
                player = datum[1]
                if phase == ml_phase and player == black_white:
                    idx = 0
                    for i in range(50):
                        num = datum[2 + i]
                        for arr in make_lines_idx(num, pattern_sizes[i], pattern_idxes[i]):
                            all_data[idx].append(arr)
                            idx += 1
                    #for _ in range(4):
                    all_data[100].append([(datum[52] - 15) / 15, (datum[53] - 15) / 15])
                    all_data[101].append([(datum[54] - 15) / 15, (datum[55] - 15) / 15])
                    all_data[102].append([(datum[56] - 15) / 15, (datum[57] - 15) / 15])
                    all_data[103].append([(datum[58] - 15) / 15, (datum[59] - 15) / 15])
                    score = datum[60] / 64
                    all_labels.append(score)
            except:
                break

model = load_model('learned_data/0_0.h5')
before_adding = Model(inputs=model.input, outputs=model.get_layer('line2_pre_prediction').output)

collect_data()
len_data = len(all_labels)
print(len_data)

all_data = [np.array(arr) for arr in all_data]
all_labels = np.array(all_labels)
print('converted to numpy arr')

pred = model.predict(all_data)
print(pred[0])