import sys
import numpy as np
from tqdm import trange

from tensorflow.keras.layers import Add, Dense, Input, LeakyReLU, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow import __version__ as tf_version
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
from random import randrange
import time
import pickle



print('tensorflow version', tf_version)

'''
def ClippedReLU(x):
    return tf.keras.backend.relu(x, max_value=1)
'''

model = tf.keras.models.Sequential()
model.add(Input(shape=128, name='in'))
model.add(Dense(16, activation='relu', name='layer_A'))
model.add(Dense(16, activation='relu', name='layer_B'))
model.add(Dense(16, activation='relu', name='layer_C'))
model.add(Dense(1, name='output_layer'))

print('model', 'param', model.count_params())
plot_model(model, to_file='./model.png', show_shapes=True)
model.summary()
model.compile(loss='mse', metrics='mae', optimizer='adam')

train_data = []
train_labels = []
for num in range(1): # 4
    strt = len(train_data)
    print(num, strt)
    with open('E:/github/othello/Egaroucid/train_data/board_data/records29/' + str(num) + '.dat', 'br') as f:
        #while len(train_data) - strt < 80000:
        for _ in trange(1000000):
            bits = int.from_bytes(f.read(8), sys.byteorder) << 64
            bits |= int.from_bytes(f.read(8), sys.byteorder)
            if bits == 0:
                break
            in_data = np.zeros(128)
            n_discs = 0
            for i in range(128):
                in_data[i] = (1 & (bits >> (127 - i)))
                if in_data[i]:
                    n_discs += 1
            f.read(2)
            score = int.from_bytes(f.read(1), sys.byteorder, signed=True)
            #if n_discs == 4 + 30:
            if n_discs >= 4 + 12:
                train_data.append(in_data)
                train_labels.append(score)
                #if len(train_data) % 10000 == 0:
                #    print(num, len(train_data))
            '''
            for i in range(2):
                for j in range(8):
                    s = i * 64 + j * 8
                    print(in_data[s:s+8])
                if i == 0:
                    print('')
            print(score)
            print('')
            '''

with open('E:/github/othello/Egaroucid/train_data/board_data/records26/0.dat', 'br') as f:
    for _ in trange(1000000 // 4):
        bits = int.from_bytes(f.read(8), sys.byteorder) << 64
        bits |= int.from_bytes(f.read(8), sys.byteorder)
        if bits == 0:
            break
        in_data = np.zeros(128)
        n_discs = 0
        for i in range(128):
            in_data[i] = (1 & (bits >> (127 - i)))
            if in_data[i]:
                n_discs += 1
        f.read(2)
        score = int.from_bytes(f.read(1), sys.byteorder, signed=True)
        #if n_discs >= 4 + 12:
        train_data.append(in_data)
        train_labels.append(score)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
print('data loaded', len(train_data), len(train_labels))



test_data = []
test_labels = []
with open('E:/github/othello/Egaroucid/train_data/board_data/records29/4.dat', 'br') as f:
    for _ in trange(100000):
        bits = int.from_bytes(f.read(8), sys.byteorder) << 64
        bits |= int.from_bytes(f.read(8), sys.byteorder)
        in_data = np.zeros(128)
        n_discs = 0
        for i in range(128):
            in_data[i] = (1 & (bits >> (127 - i)))
            if in_data[i]:
                n_discs += 1
        f.read(2)
        score = int.from_bytes(f.read(1), sys.byteorder, signed=True)
        #if n_discs >= 4 + 12:
        if n_discs == 4 + 30:
            test_data.append(in_data)
            test_labels.append(score)
        '''
        for i in range(2):
            for j in range(8):
                s = i * 64 + j * 8
                print(in_data[s:s+8])
            if i == 0:
                print('')
        print(score)
        print('')
        '''
test_data = np.array(test_data)
test_labels = np.array(test_labels)
print('data loaded', len(test_data), len(test_labels))





N_EPOCHS = 500
BATCH_SIZE = 2048
EARLY_STOP_PATIENCE = 100

# train
early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
#model_checkpoint = ModelCheckpoint(filepath=os.path.join('./../model/', 'model_{epoch:02d}_{loss:.4f}_{mae:.4f}_{val_loss:.4f}_{val_mae:.4f}.h5'), monitor='val_loss', verbose=1, period=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
history = model.fit(train_data, train_labels, initial_epoch=0, epochs=N_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], validation_data=(test_data, test_labels))

model.save('./model.h5')



cut_epoch = 0

for key in ['loss', 'val_loss']:
    plt.plot(history.history[key][cut_epoch:], label=key)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.savefig('./loss.png')
plt.clf()

for key in ['mae', 'val_mae']:
    plt.plot(history.history[key][cut_epoch:], label=key)
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend(loc='best')
plt.savefig('./mae.png')
plt.clf()