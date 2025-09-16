import subprocess
import os
import glob
import psutil
import time
from tqdm import tqdm
from data_range import *
import sys


'''
# cell weight
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20240419_1_cell_weight/'
exe = 'data_board_to_idx_cell.out'
N_PHASES = 1
#board_sub_dir_nums = [26, 29] # used [0,10] with data 26
board_sub_dir_nums = [48, 52]
#'''

'''
# cell weight phase 60
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250214_1_cell_weight_phase60/'
exe = 'data_board_to_idx_20250214_cell_weight_phase60.out'
N_PHASES = 60
board_sub_dir_nums = [97]
#'''





'''
# move ordering_end_nws ((11 to 13) - 1 empties)
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20240304_1_move_ordering_end_nws/'
exe = 'data_board_to_idx_move_ordering_end_nws.out'
N_PHASES = 1
board_sub_dir_nums = [43, 44, 45]
#'''


#'''
# 7.5
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20241125_1/'
exe = 'data_board_to_idx_20241125_1_7_5.out'
N_PHASES = 60
board_sub_dir_nums = [
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
    214, # random 11 (first11_all)
    216, 217, 218, 219, 220, # randomN
    # 222, # random0
    223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, # randomN level 15
]
# board_sub_dir_nums = [166, 167] # test data
board_sub_dir_nums = [213] # random boards 58
board_sub_dir_nums.sort()
#'''


'''
# 7.7 (not used for 7.7)
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250513_1/'
exe = 'data_board_to_idx_20250513_1_7_7.out'
N_PHASES = 60
board_sub_dir_nums = [
    222
]
board_sub_dir_nums.sort()
#'''

'''
# 7.7 move ordering end
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250512_1_move_ordering_end_nws/'
exe = 'data_board_to_idx_20250512_1_7_7_move_ordering_end.out'
N_PHASES = 1
board_sub_dir_nums = [202]
board_sub_dir_nums.sort()
#'''


min_n_data_dct = {}
min_n_data_dct['83'] = 200000


input_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/'



procs = []
for phase in range(N_PHASES):
    bin_dir = bin_root_dir + str(phase)
    try:
        os.mkdir(bin_dir)
    except:
        pass
    for board_sub_dir_num in board_sub_dir_nums:
        input_dir = input_root_dir + 'records' + str(board_sub_dir_num)
        if not os.path.isdir(input_dir):
            print(f'Error: 入力ディレクトリが存在しません: {input_dir}', file=sys.stderr)
            continue
        n_files_str = str(len(glob.glob(input_dir + '/*.dat')))
        out_file = bin_dir + '/' + str(board_sub_dir_num) + '.dat'
        if str(board_sub_dir_num) in min_n_data_dct:
            min_n_data = min_n_data_dct[str(board_sub_dir_num)]
        else:
            min_n_data = 0
        cmd = exe + ' ' + input_dir + ' 0 ' + n_files_str + ' ' + out_file + ' ' + str(phase) + ' ' + str(board_n_moves[str(board_sub_dir_num)][0]) + ' ' + str(board_n_moves[str(board_sub_dir_num)][1]) + ' ' + str(min_n_data)
        while True: # wait while cpu is busy
            cpu_percent = 0
            for _ in range(10):
                cpu_percent = max(cpu_percent, psutil.cpu_percent(percpu=False))
                time.sleep(0.01)
            #cpu_percent /= 10
            if cpu_percent < 50.0:
                break
        print(phase, board_sub_dir_num, cpu_percent, cmd)
        procs.append(subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL))
        if board_n_moves[str(board_sub_dir_num)][0] <= phase <= board_n_moves[str(board_sub_dir_num)][1]:
            #time.sleep(1)
            pass

for proc in tqdm(procs):
    proc.wait()
