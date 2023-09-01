import subprocess
from tqdm import trange

IDX_START = 0

LEVEL = 11
N_GAMES_PER_FILE = 10000
N_PARALLEL = 4
N_THREAD = 32 // N_PARALLEL

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

cmd = 'Egaroucid_for_Console_6_4_0_x64_SIMD.exe -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_GAMES_PER_FILE // N_PARALLEL)
print(cmd)

for idx in range(IDX_START, IDX_START + 100):
    egaroucids = [subprocess.Popen(cmd.split(), stdout=subprocess.PIPE) for _ in range(N_PARALLEL)]
    with open('transcript/' + fill0(idx, 7) + '.txt', 'w') as f:
        for i in trange(N_GAMES_PER_FILE):
            eg_idx = i % N_PARALLEL
            line = egaroucids[eg_idx].stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
    for i in range(N_PARALLEL):
        egaroucids[i].kill()
