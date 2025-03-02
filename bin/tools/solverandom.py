import subprocess
from tqdm import trange
import sys
import os

n_random_moves = int(sys.argv[1])
IDX_START = int(sys.argv[2])
IDX_END = int(sys.argv[3])

LEVEL = int(sys.argv[4])
N_BOARDS_PER_FILE = 10000
N_THREAD = 31

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

exe = './../Egaroucid_for_Console_clang.exe'
#exe = './../versions/Egaroucid_for_Console_7_5_1_Windows_SIMD/Egaroucid_for_Console_7_5_1_SIMD.exe'

#cmd = exe + ' -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_BOARDS_PER_FILE) + ' ' + str(n_random_moves)
cmd = exe + ' -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -solverandom ' + str(N_BOARDS_PER_FILE) + ' ' + str(n_random_moves)
print(cmd)

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    dr = './../transcript/' + str(n_random_moves)
    try:
        os.mkdir(dr)
    except:
        pass
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode().replace('\r', '')
    with open(dr + '/' + fill0(idx, 7) + '.txt', 'w') as f:
        f.write(result)
