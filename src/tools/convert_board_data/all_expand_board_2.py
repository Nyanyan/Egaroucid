# for Egaroucid_Train_Data.zip

import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = './../../../train_data/board_data/'
log_file = './../../../train_data/board_data/log.txt'

FILE_INTERVAL = 100 * 60

board_dir = board_root_dir + 'records97'
try:
    os.mkdir(board_dir)
except:
    pass
#out_file_name = '0.dat'
#board_file = board_dir + '/' + out_file_name
transcript_dir = transcript_root_dir + 'records97_boards'
files = glob.glob(transcript_dir + '/*.txt')
n_file = len(files)
s_file = 0
out_file_idx = 0
while s_file < n_file:
    out_file = board_dir + '/' + str(out_file_idx) + '.dat'
    cmd = 'board_data_processing2.out ' + out_file + ' '
    e_file = min(s_file + FILE_INTERVAL, n_file)
    for i in range(s_file, e_file):
        cmd += files[i] + ' '
    print(cmd)
    p = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
    log = out_file + '\t' + p.stdout.readline().decode().replace('\r', '').replace('\n', '')
    print(log)
    with open(log_file, 'a') as f:
        f.write(log + '\n')
    s_file = e_file
    out_file_idx += 1
