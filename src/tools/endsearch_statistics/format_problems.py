from glob import glob

files = glob('problems/*_problem.txt')

for file in files:
    with open(file, 'r') as f:
        data = f.read().splitlines()
    with open(file, 'w') as f:
        f.write('\n'.join(data[1:]))