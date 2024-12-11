import subprocess
from tqdm import trange
import os
import sys

line_dr = './../problem/etc/first12_all_shuffled'
out_dr = './../transcript/first12_all_shuffled_after800'

exe = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_clang.exe'


IDX_START = int(sys.argv[1])
IDX_END = int(sys.argv[2])

# IDX_START = 10
# IDX_END = 100

print(IDX_START, IDX_END)


LEVEL = 11
N_GAMES_PER_FILE = 10000
N_THREAD = 30

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    file = line_dr + '/' + fill0(idx, 7) + '.txt'
    cmd = exe + ' -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplayline ' + file
    print(cmd)
    with open(out_dr + '/' + fill0(idx - 800, 7) + '.txt', 'w') as f:
        egaroucid = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        for i in trange(N_GAMES_PER_FILE):
            line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
        egaroucid.kill()
