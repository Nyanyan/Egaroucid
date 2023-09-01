import subprocess
from tqdm import trange

IDX_START = 1

LEVEL = 11
N_GAMES_PER_FILE = 10000
N_PARALLEL = 8
N_THREAD = 32 // N_PARALLEL
N_ADDITIONAL_DIVISION = 10

N_PLAY_PER_AI = N_GAMES_PER_FILE / N_ADDITIONAL_DIVISION / N_PARALLEL
if N_PLAY_PER_AI != int(N_PLAY_PER_AI):
    print('error!', N_PLAY_PER_AI)
    exit()
N_PLAY_PER_AI = int(N_PLAY_PER_AI)

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

cmd = 'Egaroucid_for_Console_6_4_0_x64_SIMD.exe -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_PLAY_PER_AI)
print(cmd)

for idx in range(IDX_START, IDX_START + 1000):
    print(fill0(idx, 7))
    with open('transcript/' + fill0(idx, 7) + '.txt', 'w') as f:
        for i in trange(N_GAMES_PER_FILE):
            if i % (N_GAMES_PER_FILE // N_ADDITIONAL_DIVISION) == 0:
                egaroucids = [subprocess.Popen(cmd.split(), stdout=subprocess.PIPE) for _ in range(N_PARALLEL)]
            eg_idx = i % N_PARALLEL
            line = egaroucids[eg_idx].stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
            if (i + 1) % (N_GAMES_PER_FILE // N_ADDITIONAL_DIVISION) == 0:
                for i in range(N_PARALLEL):
                    egaroucids[i].kill()
