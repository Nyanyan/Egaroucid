import sys

file = sys.argv[1]
with open(file, 'r') as f:
    s = list(f.read().splitlines())
ss = [[len(elem), elem] for elem in s]
ss.sort(reverse=True)
with open('.'.join(file.split('.')[:-1]) + '_ordered.txt', 'w') as f:
    for _, elem in ss:
        f.write(elem + '\n')
