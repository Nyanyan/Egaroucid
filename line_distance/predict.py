from re import L
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import comb, exp
import datetime

model = load_model('learned_data/model.h5')
gap = Model(inputs=model.input, outputs=model.get_layer('leaky_re_lu').output)

hw = 8

board = '''..111...
....00.1
11111011
.1111011
0.111001
...11011
...11101
..1111..
'''
player = 0

board = board.replace('\n', '')

board_arr = [[[0.0 for _ in range(2)] for _ in range(hw)] for _ in range(hw)]
for y in range(hw):
    for x in range(hw):
        if board[y * hw + x] == str(player):
            board_arr[y][x][0] = 1.0
        elif board[y * hw + x] == str(1 - player):
            board_arr[y][x][1] = 1.0

for i in range(2):
    for y in range(hw):
        for x in range(hw):
            print(int(board_arr[y][x][i]), end=' ')
        print('')
    print('')

test_data = np.array([board_arr])
prediction = model.predict(test_data)
for y in range(hw):
    for x in range(hw):
        print(prediction[0][y * hw + x], end=' ')
    print('')

gap_prediction = gap.predict(test_data)
print(gap_prediction.shape)
for y in range(hw):
    for x in range(hw):
        print(gap_prediction[0][y][x][0], end=' ')
    print('')