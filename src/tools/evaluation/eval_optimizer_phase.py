import subprocess
import sys

phase = str(sys.argv[1])
hour = '0'
minute = '5'
second = '0'
alpha = '800.0'
n_patience = '1000'

model_dir = './../../../model/nomodel/'


'''
# 7.0
train_data_nums = [
    18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (no records27)
    34, 35, # mid-endgame data 1
    36, 37, # book data
    #38, # test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    64 # cut records27
]
if int(phase) <= 11:
    train_data_nums = [36, 37] # use only book with phase <= 11
train_data_nums.sort()
train_root_dir = './../../../train_data/bin_data/20240223_1/'
executable = 'eval_optimizer_cuda_12_2_0.exe'
'''
'''
# 7.0 move ordering end nws
train_data_nums = [24, 28]
train_root_dir = './../../../train_data/bin_data/20240304_1_move_ordering_end_nws/'
executable = 'eval_optimizer_cuda_12_2_0_move_ordering_end_nws.exe'
'''
'''
# cell weight
#train_data_nums = [29]
train_data_nums = [52]
train_root_dir = './../../../train_data/bin_data/20240419_1_cell_weight/'
executable = 'eval_optimizer_cuda_12_2_0_cell_weight.exe'
'''
'''
# 7.1 beta
train_data_nums = [27, 28, 29, 30, 31, 34, 35, 36, 37]
if int(phase) <= 11:
    train_data_nums = [36, 37] # use book only
train_data_nums.sort()
train_root_dir = './../../../train_data/bin_data/20240525_1/'
executable = 'eval_optimizer_cuda_12_2_0_20240525_1.exe'
if int(phase) <= 11:
    test_data = './../../../train_data/bin_data/20240525_1/' + phase + '/36.dat'
else:
    test_data = './../../../train_data/bin_data/20240525_1/' + phase + '/38.dat'
'''
'''
# 20240611_1_move_ordering_end
train_data_nums = [52]
train_data_nums.sort()
train_root_dir = './../../../train_data/bin_data/20240611_1_move_ordering_end/'
executable = 'eval_optimizer_cuda_12_2_0_20240611_1_move_ordering_end.exe'
test_data = './../../../train_data/bin_data/20240611_1_move_ordering_end/' + phase + '/52.dat'
'''
#'''
# 7.0 light
train_data_nums = [
    18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (no records27)
    34, 35, # mid-endgame data 1
    36, 37, # book data
    #38, # test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    64 # cut records27
]
if int(phase) <= 11:
    train_data_nums = [36, 37] # use only book with phase <= 11
train_data_nums.sort()
train_root_dir = './../../../train_data/bin_data/20240622_1_7_0_light/'
executable = 'eval_optimizer_cuda_12_2_0_20240622_1_7_0_light.exe'
#'''



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
