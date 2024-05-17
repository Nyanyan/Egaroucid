import subprocess
import os
import glob

#'''
# 7.0
bin_root_dir = './../../../train_data/bin_data/20240223_1/'
exe = 'data_board_to_idx.out'
N_PHASES = 60
#board_sub_dir_nums = [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27]
board_sub_dir_nums = [34, 35]
#'''
'''
# 7.0 move ordering_end_nws
bin_root_dir = './../../../train_data/bin_data/20240304_1_move_ordering_end_nws/'
exe = 'data_board_to_idx_move_ordering_end_nws.out'
N_PHASES = 1
board_sub_dir_nums = [24, 28]
'''
'''
# cell weight
bin_root_dir = './../../../train_data/bin_data/20240419_1_cell_weight/'
exe = 'data_board_to_idx_cell.out'
N_PHASES = 1
board_sub_dir_nums = [26, 29] # used [0,10] with data 26
'''


input_root_dir = './../../../train_data/board_data/'

board_n_moves = {}
board_n_moves['6'] = [20, 60]
board_n_moves['7'] = [20, 60]
board_n_moves['8'] = [20, 60]
board_n_moves['9'] = [20, 60]
board_n_moves['10'] = [40, 60]
board_n_moves['11'] = [20, 60]
board_n_moves['15'] = [20, 60]
board_n_moves['16'] = [19, 60]
board_n_moves['17'] = [19, 60]
board_n_moves['18'] = [19, 60]
board_n_moves['19'] = [19, 60]
board_n_moves['20'] = [11, 60] # 8, 60
board_n_moves['21'] = [11, 60] # 10, 60
board_n_moves['22'] = [11, 60]
board_n_moves['23'] = [0, 10]
board_n_moves['24'] = [21, 60]
board_n_moves['25'] = [30, 60]
board_n_moves['26'] = [0, 60]
#board_n_moves['26'] = [0, 10]
board_n_moves['27'] = [11, 60]
board_n_moves['28'] = [40, 60]
board_n_moves['29'] = [12, 60]
board_n_moves['30'] = [18, 60]
board_n_moves['31'] = [24, 60]
board_n_moves['32'] = [12, 60]
board_n_moves['33'] = [12, 60]
board_n_moves['34'] = [31, 60]
board_n_moves['35'] = [32, 60]

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
        cmd = exe + ' ' + input_dir + ' 0 ' + n_files_str + ' ' + out_file + ' ' + str(phase) + ' ' + str(board_n_moves[str(board_sub_dir_num)][0]) + ' ' + str(board_n_moves[str(board_sub_dir_num)][1])
        #print(phase, board_sub_dir_num, cmd)
        #subprocess.run(cmd.split(), stderr=None, stdout=subprocess.DEVNULL)
        procs.append(subprocess.Popen(cmd.split(), stderr=None, stdout=subprocess.DEVNULL))
        if len(procs) >= 32:
            for proc in procs:
                proc.wait()
            procs = []

for proc in procs:
    proc.wait()
