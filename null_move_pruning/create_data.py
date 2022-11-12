from tqdm import tqdm, trange

data_file = './../statistics/data/records16_with_eval/0000000_mid.txt'
data_file2 = './../statistics/data/records16_with_eval_pass/0000000.txt'

output_file = 'data/mid.txt'

data_max_depth = 11

# data format n_discs, depth1, depth2, abs_error (depth1 < depth2)

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

with open(data_file2, 'r') as f:
    raw_data2 = f.read().splitlines()

def str_to_value(elem):
    return (ord(elem) - ord('!')) * 2 - 64

with open(output_file, 'w') as f:
    for line in trange(len(raw_data)):
        board = raw_data[line][:64]
        score = str_to_value(raw_data[line][65])
        evals = [str_to_value(elem) for elem in raw_data[line][67:]]
        pass_score = -int(raw_data2[line].split()[1])
        n_discs = board.count('p') + board.count('o')
        max_depth = min(len(evals), 64 - n_discs - 1)
        for depth in range(data_max_depth):
            error = pass_score - evals[depth]
            f.write(str(n_discs) + ' ' + str(depth) + ' ' + str(pass_score) + ' ' + str(evals[depth]) + '\n')
