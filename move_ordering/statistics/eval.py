import subprocess
import glob
from othello_py import *
from copy import deepcopy
from tqdm import tqdm

drs = [
    'data/records15/'
]

out_drs = [
    'data/records15_eval/'
]

def score_to_string(score):
    s = (score + 64) // 2
    return chr(ord('!') + s)

MAX_DEPTH = 3

egaroucids = []
for depth in range(MAX_DEPTH):
    egaroucids.append(subprocess.Popen(('Egaroucid6_test.exe ' + str(depth)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))

for dr, out_dr in zip(drs, out_drs):
    files = glob.glob(dr + '*.txt')
    for file in files:
        with open(file, 'r') as f:
            data = f.read().splitlines()
        out_file = out_dr + file.split('\\')[-1]
        print(out_file)
        with open(out_file, 'w') as f:
            for datum in tqdm(data):
                output_data = ''
                o = othello()
                o.player = black
                idx = 0
                for i in range(hw):
                    for j in range(hw):
                        o.grid[i][j] = black if datum[idx] == 'p' else white if datum[idx] == 'o' else vacant
                        idx += 1
                o.check_legal()
                for i in range(hw):
                    for j in range(hw):
                        if o.grid[i][j] == legal:
                            oo = deepcopy(o)
                            oo.move(i, j)
                            input_datum = '1\n'
                            for k in range(hw):
                                for l in range(hw):
                                    input_datum += '0' if oo.grid[k][l] == black else '1' if oo.grid[k][l] == white else '.'
                                input_datum += '\n'
                            output_data += chr(ord('!') + i * hw + j)
                            for depth in range(MAX_DEPTH):
                                egaroucids[depth].stdin.write(input_datum.encode('utf-8'))
                                egaroucids[depth].stdin.flush()
                                line = egaroucids[depth].stdout.readline().decode()
                                val = int(line.split()[0])
                                output_data += score_to_string(val)
                            output_data += ' '
                f.write(output_data + '\n')

for depth in range(MAX_DEPTH):
    egaroucids[depth].kill()