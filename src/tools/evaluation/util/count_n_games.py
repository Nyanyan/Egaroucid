import glob

dirs = [
    #'records1',
    #'records2',
    #'records3',
    #'records4',
    #'records6',
    #'records7',
    #'records8',
    #'records9',
    #'records10',
    #'records11',
    #'records15',
    #'records16',
    #'records17',
    #'records18',
    'records19',
    #'records99',
]

n_all_lines = 0
for d in dirs:
    path = './../../../../train_data/transcript/' + d + '/*.txt'
    files = glob.glob(path)
    n_lines = 0
    for file in files:
        with open(file, 'r') as f:
            n_lines += len(f.read().splitlines())
    print(d, n_lines)
    n_all_lines += n_lines

print('all', n_all_lines)