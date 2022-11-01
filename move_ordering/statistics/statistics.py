import subprocess

cmds = [
    'statistics.out 2 4',
    'statistics.out 5 7',
    'statistics.out 8 13',
    'statistics.out 2',
    'statistics.out 3',
    'statistics.out 4',
    'statistics.out 5',
    'statistics.out 6',
    'statistics.out 7',
    'statistics.out 8',
    'statistics.out 9',
    'statistics.out 10',
    'statistics.out 11',
    'statistics.out 12',
    'statistics.out 13',
]

files = [
    'statistics/last.txt',
    'statistics/fast.txt',
    'statistics/end.txt',
    'statistics/2.txt',
    'statistics/3.txt',
    'statistics/4.txt',
    'statistics/5.txt',
    'statistics/6.txt',
    'statistics/7.txt',
    'statistics/8.txt',
    'statistics/9.txt',
    'statistics/10.txt',
    'statistics/11.txt',
    'statistics/12.txt',
    'statistics/13.txt'
]

for i in range(len(cmds)):
    p = subprocess.run(cmds[i].split(), stdout=subprocess.PIPE)
    with open(files[i], 'w') as f:
        f.write(p.stdout.decode().replace('\r\n', '\n'))