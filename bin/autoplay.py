import subprocess
from tqdm import trange

LEVEL = 11
N_GAMES = 1000

egaroucid = subprocess.Popen(('Egaroucid_for_Console_6_3_0_x64_SIMD.exe -l ' + str(LEVEL) + ' -thread 32 -selfplay ' + str(N_GAMES)).split(), stdout=subprocess.PIPE)


with open('transcript/selfplay_' + str(LEVEL) + '_' + str(N_GAMES) + '.txt', 'w') as f:
    for i in trange(N_GAMES):
        line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
        f.write(line)

egaroucid.kill()
