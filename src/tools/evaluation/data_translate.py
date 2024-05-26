import subprocess
import os
import glob

#'''
# 7.0
bin_root_dir = './../../../train_data/bin_data/20240223_1/'
exe = 'data_board_to_idx.out'
N_PHASES = 60
board_sub_dir_nums = [27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
#board_sub_dir_nums = [38]
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
'''
# 7.1
bin_root_dir = './../../../train_data/bin_data/20240525_1/'
exe = 'data_board_to_idx_20240525_1_7_1.out'
N_PHASES = 60
board_sub_dir_nums = [27, 28, 29, 30, 31, 34, 35, 36, 37, 38]
'''


input_root_dir = './../../../train_data/board_data/'

board_n_moves = {}
'''
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
board_n_moves['27'] = [11, 60]
board_n_moves['28'] = [40, 60]
board_n_moves['29'] = [12, 60]
board_n_moves['30'] = [18, 60]
board_n_moves['31'] = [24, 60]
board_n_moves['32'] = [12, 60]
board_n_moves['33'] = [12, 60]
board_n_moves['34'] = [31, 60]
board_n_moves['35'] = [32, 60]
board_n_moves['36'] = [0, 11]
board_n_moves['37'] = [0, 60]
board_n_moves['38'] = [12, 60]
board_n_moves['39'] = [50, 60]
board_n_moves['40'] = [49, 60]
'''

# 31手まではあるデータ全部使う
# 32手以降はランダム打ち開始+10手くらいの範囲で使う

### useful old data ###
board_n_moves['27'] = [12, 31] # random11
board_n_moves['28'] = [40, 50] # random40
board_n_moves['29'] = [12, 32] # random12
board_n_moves['30'] = [18, 31] # random18
board_n_moves['31'] = [24, 34] # random24
#board_n_moves['32'] = [12, 60] # random11 or 12 test data
#board_n_moves['33'] = [12, 60] # random12 test data

### midgame data 1 ###
board_n_moves['34'] = [31, 41] # random31
board_n_moves['35'] = [32, 42] # random32

### book data ###
board_n_moves['36'] = [0, 11] # book first11
board_n_moves['37'] = [0, 59] # book additional

### test data ###
board_n_moves['38'] = [12, 59] # random8,9,10,11 test data

### mid-endgame data ###
board_n_moves['39'] = [54, 59] # random54
board_n_moves['40'] = [53, 59] # random53
board_n_moves['41'] = [52, 59] # random52
board_n_moves['42'] = [51, 59] # random51
board_n_moves['43'] = [50, 59] # random50
board_n_moves['44'] = [49, 59] # random49
board_n_moves['45'] = [48, 58] # random48
board_n_moves['46'] = [47, 57] # random47
board_n_moves['47'] = [46, 56] # random46
board_n_moves['48'] = [45, 55] # random45
board_n_moves['49'] = [44, 54] # random44


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
