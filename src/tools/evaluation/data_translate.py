import subprocess
import os
import glob
import psutil
import time
from tqdm import tqdm
from data_range import *


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


'''
# 7.5
bin_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20241125_1/'
exe = 'data_board_to_idx_20241125_1_7_5.out'
N_PHASES = 60
board_sub_dir_nums = [195, 196, 197, 214]
board_sub_dir_nums.sort()
#'''


#'''
# 7.7
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
