import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp
import datetime

hw = 8
hw2 = 64

all_data = {}

n_epochs = 1000
game_num = 1000
test_ratio = 0.1
n_boards = 3

kernel_size = 3
n_kernels = 64
n_residual = 0

leakyrelu_alpha = 0.01

train_board = None
train_policies = None
train_value = None

test_raw_board = []
test_board = []
test_policies = []
test_value = []


def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def join_yx(y, x):
    return y * hw + x

def calc_idx(i, j, rnd):
    if rnd == 0:
        return join_yx(i, j)
    elif rnd == 1:
        return join_yx(j, hw - 1 - i)
    elif rnd == 2:
        return join_yx(hw - 1 - i, hw - 1 - j)
    else:
        return join_yx(hw - 1 - j, i)

def collect_data(num):
    global all_data
    try:
        with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        return
    for datum in data:
        board, policy, score = datum.split()
        policy = int(policy)
        score = float(score)
        #print(board, policy, score)
        all_data.append([board, policy, score])

def reshape_data_train():
    global train_board, train_policies, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    train_idx = 0
    for ii in trange(ln):
        board, policies, score = tmp_data[ii]
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = ''
        grid_space0_rev = ''
        grid_space1 = ''
        grid_space1_rev = ''
        grid_space_fill = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space0_rev += '0 ' if board[idx] == '0' else '1 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space1_rev += '0 ' if board[idx] == '1' else '1 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
                grid_space_fill += '0 ' if board[idx] == '.' else '1 '
                stone_num += board[idx] != '.'
        if stone_num < 10 or stone_num > 56:
            continue
        #grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        #train_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)])
        #train_param.append([float(i) for i in param.split()])
        #train_policies.append(policies)
        #train_value.append(score)
        '''
        train_board[idx] = [[[grid_flat[k * hw2 + j * hw + i] for k in range(3)] for j in range(hw)] for i in range(hw)]
        train_param[idx] = [float(i) for i in param.split()]
        train_policies[idx] = policies
        train_value[idx] = score
        '''
        for i in range(hw):
            for j in range(hw):
                for k in range(n_boards):
                    train_board[train_idx][i][j][k] = grid_flat[k * hw2 + j * hw + i]
        #for i, elem in zip(range(15), param.split()):
        #    train_param[train_idx][i] = float(elem)
        for i in range(hw2):
            train_policies[train_idx][i] = policies[i]
        train_value[train_idx] = score
        train_idx += 1
    train_board = train_board[0:train_idx]
    #train_param = train_param[0:train_idx]
    train_policies = train_policies[0:train_idx]
    train_value = train_value[0:train_idx]
    #mean = train_param.mean(axis=0)
    #std = train_param.std(axis=0)
    #print('mean', mean)
    #print('std', std)
    #train_param = (train_param - mean) / std
    '''
    print(train_board[0])
    print(train_param[0])
    print(train_policies[0])
    print(train_value[0])
    '''
    print('train', train_board.shape, train_policies.shape, train_value.shape)

def reshape_data_test():
    global test_board, test_policies, test_value, test_raw_board
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        policies = [0.0 for _ in range(hw2)]
        policies[policy] = 1.0
        #my_evaluate.stdin.write(board.encode('utf-8'))
        #my_evaluate.stdin.flush()
        #additional_data = my_evaluate.stdout.readline().decode().strip()
        #tmp_data.append([board, additional_data, policies, score])
        tmp_data.append([board, policies, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating test data & labels')
    for ii in trange(ln):
        board, policies, score = tmp_data[ii]
        #board, policies, score = tmp_data[ii]
        stone_num = 0
        grid_space0 = ''
        grid_space0_rev = ''
        grid_space1 = ''
        grid_space1_rev = ''
        grid_space_fill = ''
        grid_space_vacant = ''
        for i in range(hw):
            for j in range(hw):
                idx = i * hw + j
                grid_space0 += '1 ' if board[idx] == '0' else '0 '
                grid_space0_rev += '0 ' if board[idx] == '0' else '1 '
                grid_space1 += '1 ' if board[idx] == '1' else '0 '
                grid_space1_rev += '0 ' if board[idx] == '1' else '1 '
                grid_space_vacant += '1 ' if board[idx] == '.' else '0 '
                grid_space_fill += '0 ' if board[idx] == '.' else '1 '
                stone_num += board[idx] != '.'
        if stone_num < 10 or stone_num > 56:
            continue
        if stone_num < 10 or stone_num > 56:
            continue
        test_raw_board.append(board)
        #grid_flat = [float(i) for i in (grid_space0 + grid_space0_rev + grid_space1 + grid_space1_rev + grid_space_fill + grid_space_vacant).split()]
        grid_flat = [float(i) for i in (grid_space0 + grid_space1 + grid_space_vacant).split()]
        test_board.append([[[grid_flat[k * hw2 + j * hw + i] for k in range(n_boards)] for j in range(hw)] for i in range(hw)])
        #test_param.append([float(i) for i in param.split()])
        test_policies.append(policies)
        test_value.append(score)
    test_board = np.array(test_board)
    #test_param = np.array(test_param)
    test_policies = np.array(test_policies)
    test_value = np.array(test_value)
    #test_param = (test_param - mean) / std
    '''
    print(test_board[0])
    print(test_param[0])
    print(test_policies[0])
    print(test_value[0])
    '''
    print('test', test_board.shape, test_policies.shape, test_value.shape)

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)


inputs = Input(shape=(hw, hw, n_boards,))
x = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(inputs)
x = LeakyReLU(x)
for _ in range(n_residual):
    sc = x
    x = Conv2D(n_kernels, kernel_size, padding='same', use_bias=False)(x)
    x = Add()([x, sc])
    x = LeakyReLU(x)
x = GlobalAveragePooling2D()(x)

y = Activation('tanh')(x)
y = Dense(hw2)(y)
y = Activation('softmax')(y)

model = Model(inputs=inputs, outputs=y)

#model = load_model('param/model_rollout.h5')
#plot_model(model, to_file='graph/model.png', show_shapes=True, expand_nested=True)

model.summary()

print('collecting data')

n_train_data = int(game_num * (1.0 - test_ratio))
n_test_data = int(game_num * test_ratio)

idxes = list(range(game_num + 100))
shuffle(idxes)


all_data = []
for i in trange(n_train_data):
    collect_data(idxes[i])
train_board = np.zeros((len(all_data), hw, hw, n_boards))
train_policies = np.zeros((len(all_data), hw2))
train_value = np.zeros(len(all_data))
reshape_data_train()

all_data = []
for i in trange(n_train_data, game_num):
    collect_data(idxes[i])
test_raw_board = []
test_board = []
test_policies = []
test_value = []
reshape_data_test()

model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')


print(model.evaluate(train_board, train_policies))
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_board, train_policies, epochs=n_epochs, validation_data=(test_board, test_policies), callbacks=[early_stop])

with open('param/param_rollout.txt', 'w') as f:
    i = 0
    while True:
        try:
            #print(i, model.layers[i])
            dammy = model.layers[i]
            j = 0
            while True:
                try:
                    print(model.layers[i].weights[j].shape)
                    if len(model.layers[i].weights[j].shape) == 4:
                        for ll in range(model.layers[i].weights[j].shape[3]):
                            for kk in range(model.layers[i].weights[j].shape[2]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    for ii in range(model.layers[i].weights[j].shape[0]):
                                        f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 2:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            for jj in range(model.layers[i].weights[j].shape[1]):
                                f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 1:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                    j += 1
                except:
                    break
            i += 1
        except:
            break
now = datetime.datetime.today()
print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
#model.save('param/additional_learn_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')
model.save('param/model_rollout.h5')

for key in ['loss', 'val_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('policy loss')
plt.legend(loc='best')
plt.savefig('graph/rollout_loss.png')
plt.clf()

all_data = []
for i in trange(game_num, game_num + 100):
    collect_data(idxes[i])
test_raw_board = []
test_board = []
test_policies = []
test_value = []
reshape_data_test()

print(model.evaluate(test_board, test_policies))

prediction = model.predict(test_board[:10])
for i in range(10):
    print(test_raw_board[i])
    mx = 0.0
    policy = -1
    for j in range(hw2):
        print(prediction[i][j], end=' ')
        if mx < prediction[i][j]:
            mx = prediction[i][j]
            policy = j
    print('')
    print(policy, mx)
