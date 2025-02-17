import subprocess
import os
import glob
import psutil
import time


'''
# cell weight
bin_root_dir = './../../../train_data/bin_data/20240419_1_cell_weight/'
exe = 'data_board_to_idx_cell.out'
N_PHASES = 1
#board_sub_dir_nums = [26, 29] # used [0,10] with data 26
board_sub_dir_nums = [48, 52]
#'''

'''
# cell weight phase 60
bin_root_dir = './../../../train_data/bin_data/20250214_1_cell_weight_phase60/'
exe = 'data_board_to_idx_20250214_cell_weight_phase60.out'
N_PHASES = 60
board_sub_dir_nums = [97]
#'''





'''
# move ordering_end_nws ((11 to 13) - 1 empties)
bin_root_dir = './../../../train_data/bin_data/20240304_1_move_ordering_end_nws/'
exe = 'data_board_to_idx_move_ordering_end_nws.out'
N_PHASES = 1
board_sub_dir_nums = [43, 44, 45]
#'''


#'''
# 7.5
bin_root_dir = './../../../train_data/bin_data/20241125_1/'
exe = 'data_board_to_idx_20241125_1_7_5.out'
N_PHASES = 60
board_sub_dir_nums = [
    18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
    34, 35, # mid-endgame data 1
    #36, # first11 book (old)
    37, # book data
    38, # old test data
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    77,  # random 18 discs (GGS)
    78, 79, # random 11 & 12 (bug fixed)
    80, # new first11 book
    81, # test data
    82, # random 12
    83, # book data (records80 minimum 200000 data)
    84, 85, 86, 87, 88, 89, # non-regular random starting position
]
board_sub_dir_nums = [83]
board_sub_dir_nums.sort()
#'''


min_n_data_dct = {}
min_n_data_dct['83'] = 200000


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

### old data ###
board_n_moves['18'] = [19, 59] # random10-19     2000000 games
board_n_moves['19'] = [19, 59] # random10-19    10000000 games
board_n_moves['20'] = [11, 59] # random8           90741 games
board_n_moves['21'] = [11, 59] # random10         134230 games
board_n_moves['24'] = [21, 59] # random21        4790000 games
board_n_moves['25'] = [30, 59] # random30        4760000 games
board_n_moves['27'] = [12, 59] # random11 all   19786627 games > up to 5000000 games as records64
board_n_moves['28'] = [40, 59] # random40       14210000 games
board_n_moves['29'] = [12, 59] # random12        4770000 games
board_n_moves['30'] = [18, 59] # random18        4490454 games
board_n_moves['31'] = [24, 59] # random24

#board_n_moves['32'] = [12, 60] # random11 or 12 test data
#board_n_moves['33'] = [12, 60] # random12 test data

### mid-endgame data 1 ###
board_n_moves['34'] = [31, 59] # random31        4294350 games
board_n_moves['35'] = [32, 59] # random32        3772331 games

### book data ###
board_n_moves['36'] = [0, 11] # book first11
board_n_moves['37'] = [0, 40] # book additional

### test data ###
board_n_moves['38'] = [12, 59] # random8,9,10,11 test data

### mid-endgame data 2 ###
board_n_moves['39'] = [54, 59] # random54        3000000 games
board_n_moves['40'] = [53, 59] # random53        3000000 games
board_n_moves['41'] = [52, 59] # random52        3000000 games
board_n_moves['42'] = [51, 59] # random51        3000000 games
#'''
board_n_moves['43'] = [50, 59] # random50        3000000 games
board_n_moves['44'] = [49, 59] # random49        3000000 games
board_n_moves['45'] = [48, 59] # random48        3000000 games
board_n_moves['46'] = [47, 59] # random47        3000000 games
'''
# for move ordering end nws to use more random boards
board_n_moves['43'] = [50, 50] # random50        3000000 games
board_n_moves['44'] = [49, 49] # random49        3000000 games
board_n_moves['45'] = [48, 48] # random48        3000000 games
board_n_moves['46'] = [47, 47] # random47        3000000 games
#'''
board_n_moves['47'] = [46, 59] # random46        3000000 games
board_n_moves['48'] = [45, 59] # random45        3226023 games
board_n_moves['49'] = [44, 59] # random44        3000000 games
board_n_moves['50'] = [43, 59] # random43        3038216 games
board_n_moves['51'] = [42, 59] # random42        3003097 games
board_n_moves['52'] = [41, 59] # random41        3004849 games
board_n_moves['53'] = [39, 59] # random39        1687905 games
board_n_moves['57'] = [35, 59] # random35         144181 games
board_n_moves['60'] = [58, 59] # random58        3000000 games
board_n_moves['61'] = [57, 59] # random57        3000000 games
board_n_moves['62'] = [56, 59] # random56        3000000 games
board_n_moves['63'] = [55, 59] # random55        3000000 games

#board_n_moves['64'] = [12, 59] # random11 all cut 5000000 games with bug


### Egaroucid 7.4.0 lv.11 data
board_n_moves['65'] = [10, 59] # random10         100000 games
board_n_moves['66'] = [11, 59] # random11         100000 games
board_n_moves['67'] = [12, 59] # random12         100000 games
board_n_moves['68'] = [13, 59] # random13         100000 games
board_n_moves['69'] = [14, 59] # random14         100000 games
board_n_moves['70'] = [15, 59] # random15         100000 games
board_n_moves['71'] = [16, 59] # random16         100000 games
board_n_moves['72'] = [17, 59] # random17         100000 games
board_n_moves['73'] = [18, 59] # random18         100000 games
board_n_moves['74'] = [19, 59] # random19         100000 games

#board_n_moves['75'] = [12, 59] # random12         7800000 games with bug

#board_n_moves['76'] = [12, 59] # random11 all cut 8000000 games with bug

board_n_moves['77'] = [14, 59] # random 18 discs

board_n_moves['78'] = [12, 59] # random11 all cut 5999816 games bug fixed (records27)
board_n_moves['79'] = [12, 59] # random12 all cut 7799640 games bug fixed (records75)

board_n_moves['80'] = [0, 11] # new first11 book book_size 23259291
board_n_moves['81'] = [12, 59] # new test data      8702 games

board_n_moves['82'] = [12, 59] # random12           6892063 games

board_n_moves['83'] = [0, 11] # = records80 new first11 book book_size 23259291 (at least 200000 data for phase)

board_n_moves['84'] = [0, 59] # non-regular random board 4 discs    10000 games
board_n_moves['85'] = [1, 59] # non-regular random board 5 discs    20000 games
board_n_moves['86'] = [2, 59] # non-regular random board 6 discs    30000 games
board_n_moves['87'] = [3, 59] # non-regular random board 7 discs    40000 games
board_n_moves['88'] = [4, 59] # non-regular random board 8 discs    50000 games
board_n_moves['89'] = [5, 59] # non-regular random board 9 discs    31273 games


board_n_moves['97'] = [0, 59] # https://github.com/Nyanyan/Egaroucid/releases/download/training_data/Egaroucid_Train_Data.zip


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
        if str(board_sub_dir_num) in min_n_data_dct:
            min_n_data = min_n_data_dct[str(board_sub_dir_num)]
        else:
            min_n_data = 0
        cmd = exe + ' ' + input_dir + ' 0 ' + n_files_str + ' ' + out_file + ' ' + str(phase) + ' ' + str(board_n_moves[str(board_sub_dir_num)][0]) + ' ' + str(board_n_moves[str(board_sub_dir_num)][1]) + ' ' + str(min_n_data)
        while True: # wait while cpu is busy
            cpu_percent = psutil.cpu_percent(percpu=False)
            if cpu_percent < 95.0:
                break
        print(phase, board_sub_dir_num, cmd)
        procs.append(subprocess.Popen(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL))
        if board_n_moves[str(board_sub_dir_num)][0] <= phase <= board_n_moves[str(board_sub_dir_num)][1]:
            time.sleep(0.5)

for proc in procs:
    proc.wait()
