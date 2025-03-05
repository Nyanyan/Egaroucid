# for Egaroucid_Train_Data.zip

import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = './../../../train_data/board_data/'
log_file = './../../../train_data/board_data/log.txt'

nums = [123]

FILE_INTERVAL = 100 * 60

for num in nums:
    board_dir = board_root_dir + 'records' + str(num)
    try:
        os.mkdir(board_dir)
    except:
        pass
    #out_file_name = '0.dat'
    #board_file = board_dir + '/' + out_file_name
    in_dir = transcript_root_dir + 'records' + str(num) + '_boards/'
    n_files = len(glob.glob(in_dir + '/*.txt'))
    s_file = 0
    out_file_idx = 0
    while s_file < n_files:
        out_file = board_dir + '/' + str(out_file_idx) + '.dat'
        e_file = min(s_file + FILE_INTERVAL, n_files)
        cmd = 'board_data_processing2.out ' + in_dir + ' ' + str(s_file) + ' ' + str(e_file) + ' ' + out_file
        print(cmd)
        p = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
        log = out_file + '\t' + p.stdout.readline().decode().replace('\r', '').replace('\n', '')
        print(log)
        with open(log_file, 'a') as f:
            f.write(log + '\n')
        s_file = e_file
        out_file_idx += 1
