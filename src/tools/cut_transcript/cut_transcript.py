from glob import glob

dr = './../../../train_data/transcript/records19/*.txt'

N_MOVES = 30

files = glob(dr)

for file in files:
    filename = file.split('\\')[-1]
    with open(file, 'r') as f:
        s = f.read().splitlines()
    ns = ''
    for ss in s:
        if len(ss) >= N_MOVES * 2:
            ns += ss[:N_MOVES * 2] + '\n'
    
    with open('output/' + filename, 'w') as f:
        f.write(ns)
