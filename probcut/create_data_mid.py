from tqdm import tqdm

data_file = './../statistics/data/records15_1_50_with_eval/0000000_44807boards.txt'
output_file = 'data/mid.txt'

data_max_depth = 15

# data format n_discs, depth1, depth2, abs_error (depth1 < depth2)

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
        max_depth = min(len(evals), 64 - n_discs - 1)
        for depth1 in range(max_depth - 1):
            for depth2 in range(depth1 + 2, max_depth, 2):
                error = evals[depth1] - evals[depth2]
                f.write(str(n_discs) + ' ' + str(depth1) + ' ' + str(depth2) + ' ' + str(error) + '\n')