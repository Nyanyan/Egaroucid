import glob
import os
from tqdm import trange

# number_dir = 41

# dr = './../transcript/' + str(number_dir) + '/*.txt'
# out_dr = './../transcript/' + str(number_dir) + '_renumbered/'


read_parent_dir = './../transcript/20260307_egaroucid_vs_edax_lv11/*'
write_parent_dir = './../transcript/20260307_egaroucid_vs_edax_lv11_processed'

drs = glob.glob(read_parent_dir)

# print(drs)

def fill0(n, d):
    res = str(n)
    for _ in range(d - len(res)):
        res = '0' + res
    return res

for dr in drs:
    files = glob.glob(dr)

    after_dir = dr.replace(read_parent_dir.rstrip('/*'), '').lstrip('/')

    out_dir = write_parent_dir + after_dir

    print(dr, out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_data = []
    for file in files:
        with open(file, 'r') as f:
            all_data.extend(f.read().splitlines())

    len_data = len(all_data)
    print(len_data)

    for i in trange((len_data + 9999) // 10000):
        with open(out_dir + fill0(i, 7) + '.txt', 'w') as f:
            for j in range(10000):
                if i * 10000 + j >= len_data:
                    break
                f.write(all_data[i * 10000 + j] + '\n')
