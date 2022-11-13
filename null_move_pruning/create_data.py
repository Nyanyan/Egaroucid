from tqdm import tqdm, trange

data_file = './../statistics/data/records16_with_eval_5/0000000.txt'
data_file2 = './../statistics/data/records16_with_eval_pass/0000000.txt'

output_file = 'data/mid_pass_5.txt'

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

with open(data_file2, 'r') as f:
    raw_data2 = f.read().splitlines()

def str_to_value(elem):
    return (ord(elem) - ord('!')) * 2 - 64

with open(output_file, 'w') as f:
    for line in trange(len(raw_data)):
        board = raw_data[line][:64]
        score = int(raw_data[line].split()[1])
        pass_score = int(raw_data2[line].split()[1])
        n_discs = board.count('p') + board.count('o')
        f.write(str(n_discs) + ' ' + str(pass_score) + ' ' + str(score) + '\n')
