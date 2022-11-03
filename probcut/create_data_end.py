from tqdm import tqdm

data_file = './../statistics/data/records15_1_50_with_eval/0000000_test.txt'
output_file = 'data/end.txt'

transcript_end_accurate_n_discs = 64 - 21
data_max_depth = 15

# data format n_discs, depth, abs_error

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

def str_to_value(elem):
    return (ord(elem) - ord('!')) * 2 - 64

with open(output_file, 'w') as f:
    for line in tqdm(raw_data):
        board = line[:64]
        score = str_to_value(line[65])
        evals = [str_to_value(elem) for elem in line[67:]]
        n_discs = board.count('p') + board.count('o')
        if n_discs >= transcript_end_accurate_n_discs:
            for depth in range(min(data_max_depth, 64 - n_discs)):
                abs_error = abs(evals[depth] - score)
                f.write(str(n_discs) + ' ' + str(depth) + ' ' + str(abs_error) + '\n')
