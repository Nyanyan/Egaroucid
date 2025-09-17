
import subprocess
import sys
import os
from data_range import *

phase = str(sys.argv[1])
hour = '0'
minute = str(sys.argv[2]) #'7'
second = '0'
alpha = str(sys.argv[3]) #'300.0'
n_patience = '100'
reduce_lr_patience = '10'
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



'''
def calc_random_board_used_n_moves(n_random_moves):
    if n_random_moves < 20:
        return n_random_moves
    return n_random_moves + 3
train_data_nums = []
for elem in train_data_nums_all:
    #print(elem, board_n_moves[str(elem)][0], file=sys.stderr)
    if calc_random_board_used_n_moves(board_n_moves[str(elem)][0]) <= int(phase):
        train_data_nums.append(elem)
'''




# 7.5 eval training data
# train_data_nums = [
#     18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
#     34, 35, # mid-endgame data 1
#     #36, # old first11 book
#     37, # book data
#     38, # old test data
#     39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
#     65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
#     77,  # random 18 discs (GGS)
#     78, 79, # random 11 & 12 (bug fixed)
#     80, # new first11 book
#     #81, # test data
#     82 # random 12
# ]
# if int(phase) <= 11:
#     train_data_nums = [37, 80] # use only book with phase <= 11

#'''
# 7.5
train_data_nums = [
    # 18, 19, 20, 21, 24, 25, 28, 29, 30, 31] # old data (without records27)
    34, 35, # mid-endgame data 1
    #36, # old first11 book
    37, # book data
    38, # old test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    # 65, 66,  # Egaroucid 7.4.0 1000000 data (random 10 & 11)
    67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    77,  # random 18 discs (GGS)
    78, 79, # random 11 & 12 (bug fixed)
    80, # new first11 book
    #81, # test data
    82, # random 12
    # 83, # new first11 book data (records80 minimum 200000 data)
    #84, 85, 86, 87, 88, 89, # non-regular random starting position
    97, # public data
    #           98,  99, 100, 101, 102, 103, 104, 105, # random boards 12-19
    #106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # random boards 20-29
    #116, 117, 118, 119, 120, 121, 122, 123,           # random boards 30-37
    #                                        124, 125, # random boards 38-39
    #     127, 128, 129, 130, 131, 132, 133, 134, 135, # random boards 41-49
    #136, 137, 138, 139, 140, 141, 142, 143,           # random boards 50-57
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, # 157, # random N selfplay
    158, 159, 160, 161, 162, 163, 164, 165, # random N selfplay
    #               168, 169, 170, 171, 172, 173, 174, # random boards 13-19
    #175, 176, 177, 178, 179, 180, 181, 182, 183, 184, # random boards 20-29
    #185,                          191, 192, 193, 194, # random boards 30-39
    #195, 196, 197, 198, 199, 200, 201, 202, 203, 204, # random boards 40-49
    #205, 206, 207, 208, 209, 210, 211, 212, 213, # random boards 50-58
    # 214, # random 11 (first11_all)
    216, 217, 218, 219, 220, # random39-35
    # 222, # random0
    223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, # random12-23 level 15
]
if int(phase) >= 12:
    train_data_nums.extend([18, 19, 20, 21, 24, 25, 28, 29, 30, 31]) # old data (without records27)
    train_data_nums.extend([65, 66])
    train_data_nums.extend([214]) # random 11 (first11_all)
train_data_nums.sort()
#print(train_data_nums, file=sys.stderr)
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20241125_1/'
# executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5.exe'
# executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5_ignore_rare.exe'
# executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5_20250914.exe'
executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5_roundminmax.exe'
# executable = 'eval_optimizer_cuda_12_2_0_20241125_1_7_5_new_alpha.exe'
#'''


'''
# 7.7 (not used in 7.7)
train_data_nums = [
    25, #28, # old data (random 30 & 40)
    # 18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
    34, 35, # mid-endgame data 1
    #36, # old first11 book
    37, # book data
    38, # old test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    77,  # random 18 discs (GGS)
    78, 79, # random 11 & 12 (bug fixed)
    80, # new first11 book
    81, # test data
    82, # random 12
    #83, # new first11 book data (records80 minimum 200000 data)
    #84, 85, 86, 87, 88, 89, # non-regular random starting position
    97, # public data
    #           98,  99, 100, 101, 102, 103, 104, 105, # random boards 12-19
    #106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # random boards 20-29
    #116, 117, 118, 119, 120, 121, 122, 123,           # random boards 30-37
    #                                        124, 125, # random boards 38-39
    #     127, 128, 129, 130, 131, 132, 133, 134, 135, # random boards 41-49
    #136, 137, 138, 139, 140, 141, 142, 143,           # random boards 50-57
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, # 157, # random N selfplay
    158, 159, 160, 161, 162, 163, 164, 165, # random N selfplay
                   168, 169, 170, 171, 172, 173, 174, # random boards 13-19
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, # random boards 20-29
    185,                     190, 191, 192, 193, 194, # random boards 30-39
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, # random boards 40-49
    205, 206, 207, 208, 209, 210, 211, 212, 213, # random boards 50-58
    214, # random 11 (first11_all)
    216, 217, 218, 219, 220, # randomN
    222, # random0
    223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, # randomN level 15
]
if int(phase) < 40:
    train_data_nums.extend([18, 19, 20, 21, 24, 29, 30, 31]) # old data (without records27)
train_data_nums.sort()
#print(train_data_nums, file=sys.stderr)
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250513_1/'
#executable = 'eval_optimizer_cuda_12_2_0_20250513_1_7_7.exe'
executable = 'eval_optimizer_cuda_12_2_0_20250513_1_7_7_roundminmax.exe'
#'''


'''
# 7.7 move ordering end nws
# used in last (13, 12, 11) - 1 empties (random 48, 49, 50)
#train_data_nums = [44, 45, 46] # random 48, 49, 50
train_data_nums = [202, 203, 204] # random boards
train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250512_1_move_ordering_end_nws/'
executable = 'eval_optimizer_cuda_12_2_0_20250512_1_7_7_move_ordering_end.exe'
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
# param = p.stdout.read().decode().replace('\r\n', '\n')
# with open('trained/' + phase + '.txt', 'w') as f:
#     f.write(param)
