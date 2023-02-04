import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = './../../../train_data/board_data/'
board_sub_dir_nums = [1, 2, 4, 6, 7, 8, 9, 11, 15, 16, 99]
out_file_name = '0.dat'
log_file = './../../../train_data/board_data/log.txt'

for board_sub_dir_num in board_sub_dir_nums:
    board_dir = board_root_dir + 'records' + str(board_sub_dir_num)
    os.mkdir(board_dir)
    board_file = board_dir + '/' + out_file_name
    transcript_dir = transcript_root_dir + 'records' + str(board_sub_dir_num)
    n_files_str = str(len(glob.glob(transcript_dir + '/*.txt')))
    cmd = 'expand_transcript.out ' + transcript_dir + ' 0 ' + n_files_str + ' ' + board_file
    p = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
    log = board_file + '\t' + p.stdout.readline().decode().replace('\r').replace('\n')
    print(log)
    with open(log_file, 'a') as f:
        f.write(log + '\n')
