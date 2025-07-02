import subprocess
from tqdm import trange
import sys
import os

n_random_moves = int(sys.argv[1])
IDX_START = int(sys.argv[2])
IDX_END = int(sys.argv[3])

LEVEL = 17
N_GAMES_PER_FILE = 10000
N_THREAD = 31

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

#exe = './../Egaroucid_for_Console_clang.exe'
exe = './../versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe'

#cmd = exe + ' -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_GAMES_PER_FILE) + ' ' + str(n_random_moves)
cmd = exe + ' -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_GAMES_PER_FILE) + ' ' + str(n_random_moves)
print(cmd)

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    dr = './../transcript/' + str(n_random_moves)
    try:
        os.mkdir(dr)
    except:
        pass
    with open(dr + '/' + fill0(idx, 7) + '.txt', 'w') as f:
        egaroucid = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        for i in trange(N_GAMES_PER_FILE):
            line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
        egaroucid.kill()
