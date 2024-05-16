import subprocess
import sys

phase = str(sys.argv[1])
hour = '0'
minute = '1'
second = '0'
alpha = '10000'
n_patience = '1'

model_dir = './../../../model/nomodel/'


#'''
# 7.0
train_data_nums = [26, 27, 28, 29, 30, 31]
if phase == '11':
    train_data_nums.remove(27) # use book only
if 30 <= int(phase) and int(phase) <= 39:
    train_data_nums.append(18)
    train_data_nums.append(19)
    train_data_nums.append(24)
    train_data_nums.append(25)
train_root_dir = './../../../train_data/bin_data/20240223_1/'
executable = 'eval_optimizer_cuda_12_2_0.exe'
#'''
'''
# 7.0 move ordering end nws
train_data_nums = [24, 28]
train_root_dir = './../../../train_data/bin_data/20240304_1_move_ordering_end_nws/'
executable = 'eval_optimizer_cuda_12_2_0_move_ordering_end_nws.exe'
'''
'''
# cell weight
train_data_nums = [29]
train_root_dir = './../../../train_data/bin_data/20240419_1_cell_weight/'
executable = 'eval_optimizer_cuda_12_2_0_cell_weight.exe'
'''


train_data = [str(elem) + '.dat' for elem in train_data_nums]
train_dirs = [train_root_dir + str(int(phase)) + '/']

additional_params = ''
for tfile in train_data:
    for train_dir in train_dirs:
        additional_params += ' ' + train_dir + tfile

cmd = executable + ' ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + alpha + ' ' + n_patience + ' ' + model_dir + phase + '.txt' + additional_params
print(cmd, file=sys.stderr)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
result = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
print(result)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open('trained/' + phase + '.txt', 'w') as f:
    f.write(param)
