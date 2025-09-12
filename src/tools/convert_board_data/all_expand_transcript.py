import subprocess
import os
import glob

transcript_root_dir = './../../../train_data/transcript/'
board_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/'
log_file = os.environ['EGAROUCID_DATA'] + '/train_data/board_data/log.txt'

'''
board_sub_dir_nums = [
    18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
    34, 35, # mid-endgame data 1
    #36, # old first11 book
    #37, # book data
    38, # old test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    77,  # random 18 discs (GGS)
    78, 79, # random 11 & 12 (bug fixed)
    #80, # new first11 book
    81, # test data
    82, # random 12
    #83, # new first11 book data (records80 minimum 200000 data)
    #84, 85, 86, 87, 88, 89, # non-regular random starting position
    #97, # public data
    #           98,  99, 100, 101, 102, 103, 104, 105, # random boards 12-19
    #106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # random boards 20-29
    #116, 117, 118, 119, 120, 121, 122, 123,           # random boards 30-37
    #                                        124, 125, # random boards 38-39
    #     127, 128, 129, 130, 131, 132, 133, 134, 135, # random boards 41-49
    #136, 137, 138, 139, 140, 141, 142, 143,           # random boards 50-57
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, # 157, # random N selfplay
    158, 159, 160, 161, 162, 163, 164, 165, # random N selfplay
    #               168, 169, 170, 171, 172, 173, 174, # random boards 13-19
    #175, 176, 177, 178, 179, 180, 181, 182, 183, 184, # random boards 20-29
    #185,                          191, 192, 193, 194, # random boards 30-39
    #195, 196, 197, 198, 199, 200, 201, 202, 203, 204, # random boards 40-49
    #205, 206, 207, 208, 209, 210, 211, 212, 213, # random boards 50-58
    214, # random 11 (first11_all)
    216, 217, 218, 219, 220, # randomN
    222, # random0
    
    #166, 
    167, # test data
    
]
'''
board_sub_dir_nums = [223, 224, 225, 226, 234]


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
        cmd = 'expand_transcript.out ' + transcript_dir + ' ' + str(s_file) + ' ' + str(e_file) + ' ' + out_file
        print(cmd)
        p = subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
        log = out_file + '\t' + p.stdout.readline().decode().replace('\r', '').replace('\n', '')
        print(log)
        with open(log_file, 'a') as f:
            f.write(log + '\n')
        s_file = e_file
        out_file_idx += 1
