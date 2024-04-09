import glob
from tqdm import trange

dr = './transcript/*.txt'
out_dr = './transcript/renumbered/'

files = glob.glob(dr)

all_data = []
for file in files:
    with open(file, 'r') as f:
        all_data.extend(f.read().splitlines())

len_data = len(all_data)
print(len_data)

def fill0(n, d):
    res = str(n)
    for _ in range(d - len(res)):
        res = '0' + res
    return res

for i in trange((len_data + 9999) // 10000):
    with open(out_dr + fill0(i, 7) + '.txt', 'w') as f:
        for j in range(10000):
            if i * 10000 + j >= len_data:
                break
            f.write(all_data[i * 10000 + j] + '\n')        
