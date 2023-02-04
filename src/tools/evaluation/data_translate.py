import subprocess
import os
import glob

bin_root_dir = './../../../train_data/bin_data/20230204_1/'
input_root_dir = './../../../train_data/board_data/'
board_sub_dir_nums = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 99]

for n_discs in range(4, 64):
    bin_dir = bin_root_dir + str(n_discs)
    try:
        os.mkdir(bin_dir)
    except:
        pass
    for board_sub_dir_num in [16]: #board_sub_dir_nums:
        input_dir = input_root_dir + 'records' + str(board_sub_dir_num)
        n_files_str = str(len(glob.glob(input_dir + '/*.dat')))
        out_file = bin_dir + '/' + str(board_sub_dir_num) + '.dat'
        cmd = 'data_board_to_idx.out ' + input_dir + ' 0 ' + n_files_str + ' ' + out_file + ' ' + str(n_discs)
        print(n_discs, board_sub_dir_num)
        subprocess.run(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)