import subprocess
import sys
import os

phase = str(sys.argv[1])
hour = '0'
minute = str(sys.argv[2]) #'7'
second = '0'
alpha = str(sys.argv[3]) #'300.0'
n_patience = '100'
reduce_lr_patience = '1000' #'10'
reduce_lr_ratio = '0.7'

model_dir = './../../../model/nomodel/'




'''
# cell weight
#train_data_nums = [29]
train_data_nums = [52]
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20240419_1_cell_weight/'
executable = 'eval_optimizer_cuda_12_2_0_cell_weight.exe'
#'''


'''
# cell weight phase 60
train_data_nums = [97]
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250214_1_cell_weight_phase60/'
executable = 'eval_optimizer_cuda_12_2_0_cell_weight.exe'
#'''




'''
# move ordering end nws
# used in last (13, 12, 11) - 1 empties (random 48, 49, 50)
train_data_nums = [44, 45, 46]
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20240304_1_move_ordering_end_nws/'
executable = 'eval_optimizer_cuda_12_2_0_move_ordering_end_nws.exe'
#'''



#'''
# 7.5
train_data_nums = [
    18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
    34, 35, # mid-endgame data 1
    #36, # old first11 book
    37, # book data
    38, # old test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    77,  # random 18 discs (GGS)
    78, 79, # random 11 & 12 (bug fixed)
            80, # new first11 book
    #81, # test data
    82, # random 12
            #83, # book data (records80 minimum 200000 data)
            #84, 85, 86, 87, 88, 89, # non-regular random starting position
    97, # public data
    #           98,  99, 100, 101, 102, 103, 104, 105, # random boards 12-19
    #106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # random boards 20-29
    #116, 117, 118, 119, 120, 121, 122, 123,           # random boards 30-37
                                            124, 125, # random boards 38-39
         127, 128, 129, 130, 131, 132, 133, 134, 135, # random boards 41-49
    136, 137, 138, 139, 140, 141, 142, 143,           # random boards 50-57
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, # 157, # randomN
    158, 159, 160, 161, 162, 163, 164, 165, # randomN
]
train_data_nums.sort()
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20241125_1/'
executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5.exe'
#'''



train_data = [str(elem) + '.dat' for elem in train_data_nums]
train_dirs = [train_root_dir + str(int(phase)) + '/']

additional_params = ''
for tfile in train_data:
    for train_dir in train_dirs:
        additional_params += ' ' + train_dir + tfile

cmd = executable + ' ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + alpha + ' ' + n_patience + ' ' + reduce_lr_patience + ' ' + reduce_lr_ratio + ' ' + model_dir + phase + '.txt' + additional_params
#print(cmd, file=sys.stderr)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
result = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
print(result)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open('trained/' + phase + '.txt', 'w') as f:
    f.write(param)
