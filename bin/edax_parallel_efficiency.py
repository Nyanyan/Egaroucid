import subprocess
import sys

data = []

for n_thread in range(1, 25):
    cmd = 'wEdax-x64-modern.exe -l 60 -solve problem/ffo40-49.txt -n ' + str(n_thread)
    print(cmd)
    edax = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    res = ''
    line = edax.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print('#   ' + line)
    for i in range(12):
        line = edax.stdout.readline()
        print(line)
    line = edax.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line)
    n_nodes = int(line.split()[1])
    tim = float(line.split()[4].split(':')[1])
    nps = int(line.split()[5][1:])
    print(n_nodes, tim, nps)
    data.append([n_thread, n_nodes, tim, nps])
    for _ in range(2):
        edax.stdout.readline()
    edax.kill()

for datum in data:
    print(*datum, sep=',')