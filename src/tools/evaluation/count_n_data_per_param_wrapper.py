import subprocess
import os


N_PHASE = 60


for phase in range(N_PHASE):

    # 7.5 eval training data?
    # train_data_nums = [
    #     18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
    #     34, 35, # mid-endgame data 1
    #     #36, # old first11 book
    #     37, # book data
    #     38, # old test data
    #     39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
    #     65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
    #     77,  # random 18 discs (GGS)
    #     78, 79, # random 11 & 12 (bug fixed)
    #     80, # new first11 book
    #     #81, # test data
    #     82 # random 12
    # ]
    # if int(phase) <= 11:
    #     train_data_nums = [37, 80] # use only book with phase <= 11
    # train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250513_1/'


    # 7.8 training data
    train_data_nums = [
        18, 19, 20, 21, 24, 25, 28, 29, 30, 31, # old data (without records27)
        34, 35, # mid-endgame data 1
        #36, # old first11 book
        37, # book data
        38, # old test data
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57, 60, 61, 62, 63, # mid-endgame data 2
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, # Egaroucid 7.4.0 1000000 data
        77,  # random 18 discs (GGS)
        78, 79, # random 11 & 12 (bug fixed)
        #80, # new first11 book
        #81, # test data
        82, # random 12
        83, # new first11 book data (records80 minimum 200000 data)
        #84, 85, 86, 87, 88, 89, # non-regular random starting position
        97, # public data
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
        # 222, # random0
        223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, # randomN level 15
    ]
    train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20250513_1/'


    train_dirs = [train_root_dir + str(int(phase)) + '/']
    train_data = [str(elem) + '.dat' for elem in train_data_nums]
    train_files_str = ''
    for tfile in train_data:
        for train_dir in train_dirs:
            train_files_str += ' ' + train_dir + tfile

    cmd = 'count_n_data_per_param.out ' + train_files_str
    print(cmd)
    out_path = os.path.join('trained', f"{phase}_weight.txt")
    with open(out_path, 'wb') as out_f:
        subprocess.run(cmd, shell=True, stdout=out_f, stderr=subprocess.DEVNULL)
