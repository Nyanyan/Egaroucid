import subprocess
from tqdm import trange
import os
import sys

IDX_START = int(sys.argv[1])
IDX_END = int(sys.argv[2])

print(IDX_START, IDX_END)

# IDX_START = 10
# IDX_END = 100

LEVEL = 11
N_GAMES_PER_FILE = 10000
N_THREAD = 1

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    file = './../problem/first11_all/' + fill0(idx, 7) + '.txt'
    cmd = './../versions/Egaroucid_for_Console_6_5_X/Egaroucid_for_Console.exe -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplayline ' + file
    with open('./../transcript/first11/' + fill0(idx, 7) + '.txt', 'w') as f:
        egaroucid = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        for i in trange(N_GAMES_PER_FILE):
            line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
        egaroucid.kill()
