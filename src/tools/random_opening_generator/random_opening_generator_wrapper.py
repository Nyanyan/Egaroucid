import subprocess
from tqdm import trange
import sys
import os

n_discs = int(sys.argv[1])
IDX_START = int(sys.argv[2])
IDX_END = int(sys.argv[3])

LEVEL = 30
N_GAMES_PER_FILE = 10000
N_THREAD = 31

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

exe = 'random_opening_generator.out'

#cmd = exe + ' -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_GAMES_PER_FILE) + ' ' + str(n_random_moves)
cmd = exe + ' ' + str(n_discs) + ' 10000'
print(cmd)

for idx in range(IDX_START, IDX_END + 1):
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    with open('output/' + fill0(idx, 7) + '.txt', 'w') as f:
        for i in trange(N_GAMES_PER_FILE):
            line = p.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
    p.kill()
