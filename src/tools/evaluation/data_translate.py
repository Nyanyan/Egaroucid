import subprocess
import os
import glob

bin_root_dir = './../../../train_data/bin_data/20230707/'
input_root_dir = './../../../train_data/board_data/'
board_sub_dir_nums = [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

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


N_PHASES = 30

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
        cmd = 'data_board_to_idx.out ' + input_dir + ' 0 ' + n_files_str + ' ' + out_file + ' ' + str(phase) + ' ' + str(board_n_moves[str(board_sub_dir_num)][0]) + ' ' + str(board_n_moves[str(board_sub_dir_num)][1])
        print(phase, board_sub_dir_num, cmd)
        subprocess.run(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
