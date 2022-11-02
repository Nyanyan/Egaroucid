from glob import glob

drs = [
    ['./../statistics/data/records15/*.txt', './../statistics/data/records15_eval/*.txt']
]

min_n_discs = 64 - 11
max_n_discs = 64 - 10

for board_dr, eval_dr in drs:
    board_files = glob(board_dr)
    eval_files = glob(eval_dr)
    for board_file, eval_file in zip(board_files, eval_files):
        with open(board_file, 'r') as f:
            boards = f.read().splitlines()
        with open(eval_file, 'r') as f:
            evals = f.read().splitlines()
        