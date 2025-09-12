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
        hogehoge
    ]
    train_root_dir = os.environ['EGAROUCID_DATA'] + '/train_data/bin_data/20241125_1/'


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
