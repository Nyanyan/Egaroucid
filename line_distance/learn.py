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

hw = 8
hw2 = 64

n_epochs = 10
test_ratio = 0.1
n_boards = 2

kernel_size = 3
n_kernels = 16
n_residual = 3

all_data = []
all_labels = []

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def collect_data(file_num):
    with open('data/' + digit(file_num, 7) + '.txt', 'r') as f:
        data = f.read().splitlines()
    for datum in data:
        raw_board, player, policy = datum.split()
        player = int(player)
        policy = int(policy)
        board = [[[0.0 for _ in range(2)] for _ in range(hw)] for _ in range(hw)]
        for y in range(hw):
            for x in range(hw):
                if raw_board[y * hw + x] == str(player):
                    board[y][x][0] = 1.0
                elif raw_board[y * hw + x] == str(1 - player):
                    board[y][x][1] = 1.0
        all_data.append(board)
        label = [0.0 for _ in range(hw2)]
        label[policy] = 1.0
        all_labels.append(label)

inputs = Input(shape=(hw, hw, n_boards,))
x1 = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(inputs)
x1 = LeakyReLU(alpha=0.01)(x1)
for _ in range(n_residual):
    sc = x1
    x1 = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(x1)
    x1 = Add()([x1, sc])
    x1 = LeakyReLU(alpha=0.01)(x1)
x1 = GlobalAveragePooling2D()(x1)
yp = Activation('tanh')(x1)
yp = Dense(hw2)(yp)
yp = Activation('softmax', name='policy')(yp)

model = Model(inputs=inputs, outputs=yp)
model.summary()

for file in trange(5):
    collect_data(file)

n_data = len(all_labels)
print(n_data)
all_data = np.array(all_data)
all_labels = np.array(all_labels)
print('converted to numpy arr')
p = np.random.permutation(n_data)
all_data = all_data[p]
all_labels = all_labels[p]

train_num = int((1.0 - test_ratio) * n_data)
train_data = all_data[0:train_num]
train_labels = all_labels[0:train_num]
test_data = all_data[train_num:n_data]
test_labels = all_labels[train_num:n_data]

model.compile(loss='categorical_crossentropy', optimizer='adam')
#print(model.evaluate([train_board], [train_policies, train_value]))
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_data, train_labels, epochs=n_epochs, batch_size=2048, validation_data=(test_data, test_labels), callbacks=[early_stop])

now = datetime.datetime.today()
print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
model.save('learned_data/model.h5')

for key in ['loss', 'val_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('policy loss')
plt.legend(loc='best')
plt.savefig('learned_data/loss.png')
plt.clf()
