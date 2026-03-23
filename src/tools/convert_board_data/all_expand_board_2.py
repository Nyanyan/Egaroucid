# for Egaroucid_Train_Data.zip

import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/'
log_file = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/log.txt'

nums = [
              98,  99, 100, 101, 102, 103, 104, 105, # random boards 12-19
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # random boards 20-29
    116, 117, 118, 119, 120, 121, 122, 123,           # random boards 30-37
    127, 128, 129, 130, 131, 132, 133, 134, 135, # random boards 41-49
    136, 137, 138, 139, 140, 141, 142, 143,           # random boards 50-57

                                           124, 125, # random boards 38-39
                  168, 169, 170, 171, 172, 173, 174, # random boards 13-19
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, # random boards 20-29
    185,                          191, 192, 193, 194, # random boards 30-39
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, # random boards 40-49
    205, 206, 207, 208, 209, 210, 211, 212, 213, # random boards 50-58
]

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
