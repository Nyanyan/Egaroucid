from random import shuffle
from glob import glob
from tqdm import tqdm, trange

in_dr = 'output/first12_all'
out_dr = 'output/first12_all_shuffled'

files = glob(in_dr + '/*.txt')

data = []
for file in tqdm(files):
    with open(file, 'r') as f:
        data.extend(f.read().splitlines())
print(len(data))

shuffle(data)
print('shuffled')

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

data_idx = 0
file_idx = 0
for file_idx in trange((len(data) + 9999) // 10000):
    out_file = out_dr + '/' + fill0(file_idx, 7) + '.txt'
    with open(out_file, 'w') as f:
        for i in range(10000):
            if data_idx >= len(data):
                break
            f.write(data[data_idx] + '\n')
            data_idx += 1
