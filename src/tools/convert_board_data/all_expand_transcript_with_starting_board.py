import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/'
board_sub_dir_nums = [89]
log_file = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/log.txt'

FILE_INTERVAL = 100

for board_sub_dir_num in board_sub_dir_nums:
    board_dir = board_root_dir + 'records' + str(board_sub_dir_num)
    try:
        os.mkdir(board_dir)
    except:
        pass
    #out_file_name = '0.dat'
    #board_file = board_dir + '/' + out_file_name
    transcript_dir = transcript_root_dir + 'records' + str(board_sub_dir_num)
    n_file = len(glob.glob(transcript_dir + '/*.txt'))
    s_file = 0
    out_file_idx = 0
    while s_file < n_file:
        out_file = board_dir + '/' + str(out_file_idx) + '.dat'
        e_file = min(n_file, s_file + FILE_INTERVAL)
        cmd = 'expand_transcript_with_starting_board.out ' + transcript_dir + ' ' + str(s_file) + ' ' + str(e_file) + ' ' + out_file
        print(cmd)
        p = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
        log = out_file + '\t' + p.stdout.readline().decode().replace('\r', '').replace('\n', '')
        print(log)
        with open(log_file, 'a') as f:
            f.write(log + '\n')
        s_file = e_file
        out_file_idx += 1
