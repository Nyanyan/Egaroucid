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
n_epochs = 100

data = ''
with open('learned_data/param14.txt', 'r') as f:
    data = f.read()
with open('learned_data/param.txt', 'w') as f:
    for i in range(15):
        f.write(data)
exit()
pattern_nums = [8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10]
pattern_pat = []
for i in pattern_nums:
    pattern_pat.append(pow(3, i))
pattern_sums = [0]
for i in pattern_pat:
    pattern_sums.append(pattern_sums[-1] + i)
print(pattern_sums)

for phase in [14]:
    
    for pattern in range(len(pattern_nums)):

        all_data = []
        all_labels = []
        used_idx = set()

        def digit(n, r):
            n = str(n)
            l = len(n)
            for i in range(r - l):
                n = '0' + n
            return n

        def calc_used_idx():
            with open('big_data.txt', 'r') as f:
                data = list(f.read().splitlines())
            for datum in tqdm(data):
                datum_split = datum.split()
                phase_idx = int(datum_split[0])
                if phase_idx == phase:
                    used_idx.add(int(datum_split[pattern + 2]))

        def collect_data():
            global all_data, all_labels
            try:
                with open('param.txt', 'r') as f:
                    data = list(f.read().splitlines())
            except:
                print('cannot open')
                return
            #for _ in range(10000):
            #    datum = data[randrange(len(data))]
            for idx, datum in tqdm(enumerate(data[pattern_sums[pattern]:pattern_sums[pattern + 1]])):
                if idx in used_idx:
                    result = int(datum)
                    arr1 = []
                    arr2 = []
                    for i in range(pattern_nums[pattern]):
                        if idx % 3 == 0:
                            arr1.append(1)
                            arr2.append(0)
                        elif idx % 3 == 1:
                            arr1.append(0)
                            arr2.append(1)
                        else:
                            arr1.append(0)
                            arr2.append(0)
                        idx //= 3
                    arr1.extend(arr2)
                    all_data.append(arr1)
                    all_labels.append(result / 100)
        
        def collect_all_data():
            global all_data, all_labels
            try:
                with open('param.txt', 'r') as f:
                    data = list(f.read().splitlines())
            except:
                print('cannot open')
                return
            #for _ in range(10000):
            #    datum = data[randrange(len(data))]
            for idx, datum in tqdm(enumerate(data[pattern_sums[pattern]:pattern_sums[pattern + 1]])):
                result = int(datum)
                arr1 = []
                arr2 = []
                for i in range(pattern_nums[pattern]):
                    if idx % 3 == 0:
                        arr1.append(1)
                        arr2.append(0)
                    elif idx % 3 == 1:
                        arr1.append(0)
                        arr2.append(1)
                    else:
                        arr1.append(0)
                        arr2.append(0)
                    idx //= 3
                arr1.extend(arr2)
                all_data.append(arr1)
                all_labels.append(result / 100)

        x = Input(shape=pattern_nums[pattern] * 2)
        y = Dense(64, name='dense0')(x)
        y = LeakyReLU(alpha=0.01)(y)
        y = Dense(64, name='dense1')(y)
        y = LeakyReLU(alpha=0.01)(y)
        y = Dense(64, name='dense2')(y)
        y = LeakyReLU(alpha=0.01)(y)
        y = Dense(1, name='out')(y)
        model = Model(inputs=x, outputs=y)

        #model.summary()
        #plot_model(model, to_file='learned_data/model.png', show_shapes=True)

        model.compile(loss='mse', metrics='mae', optimizer='adam')

        calc_used_idx()
        collect_data()
        len_data = len(all_labels)
        print(len_data)
        
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        print('converted to numpy arr')
        
        p = np.random.permutation(len_data)
        all_data = all_data[p]
        all_labels = all_labels[p]

        #print(model.evaluate(all_data, all_labels))
        #early_stop = EarlyStopping(monitor='val_loss', patience=20)
        #model_checkpoint = ModelCheckpoint(filepath=os.path.join('learned_data/' + str(stone_strt) + '_' + str(stone_end), 'model_{epoch:02d}_{val_loss:.5f}_{val_mae:.5f}.h5'), monitor='val_loss', verbose=1)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
        history = model.fit(all_data, all_labels, epochs=n_epochs, batch_size=32, validation_split=0.0, callbacks=[])

        now = datetime.datetime.today()
        print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
        model.save('learned_data/' + str(phase) + '_' + str(pattern) + '.h5')

        all_data = []
        all_labels = []
        collect_all_data()
        all_data = np.array(all_data)
        prediction = model.predict(all_data)
        with open('learned_data/param' + str(phase) + '_' + str(pattern) + '.txt', 'w') as f:
            for i, pred in enumerate(prediction):
                if i in used_idx:
                    f.write(str(round(all_labels[i] * 100)) + '\n')
                else:
                    f.write(str(round(pred[0] * 100)) + '\n')
    
    with open('param.txt', 'r') as f:
        data = list(f.read().splitlines())
    #for _ in range(10000):
    #    datum = data[randrange(len(data))]
    with open('learned_data/param' + str(phase) + '_' + str(len(pattern_nums)) + '.txt', 'w') as f:
        for datum in data[pattern_sums[len(pattern_nums)]:]:
            f.write(str(datum) + '\n')
    
    data = ''
    for i in range(len(pattern_nums) + 1):
        with open('learned_data/param' + str(phase) + '_' + str(i) + '.txt', 'r') as f:
            data += f.read()
    with open('learned_data/param' + str(phase) + '.txt', 'w') as f:
        f.write(data)