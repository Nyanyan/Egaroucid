import subprocess
from tqdm import trange

IDX_START = 0
IDX_END = 10000

LEVEL = 11
N_GAMES_PER_FILE = 10000
N_THREAD = 32

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

#cmd = 'Egaroucid_for_Console_6_4_0_x64_SIMD.exe -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD_PER_EXE) + ' -selfplay ' + str(N_PLAY_PER_AI)
cmd = 'Egaroucid_for_Console.exe -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplay ' + str(N_GAMES_PER_FILE)
print(cmd)

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    with open('transcript/' + fill0(idx, 7) + '.txt', 'w') as f:
        egaroucid = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        for i in trange(N_GAMES_PER_FILE):
            line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
        egaroucid.kill()
