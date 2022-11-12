import subprocess
import glob
from othello_py import *
from copy import deepcopy
from tqdm import tqdm

drs = [
    'data/records16/'
]

out_drs = [
    'data/records16_with_eval_pass/'
]

egaroucid = subprocess.Popen(('Egaroucid6_test.exe').split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

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
                '''
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
                                egaroucid.stdin.write((input_datum + str(depth) + '\n').encode('utf-8'))
                                egaroucid.stdin.flush()
                                line = egaroucid.stdout.readline().decode()
                                val = int(line)
                                output_data += score_to_string(val)
                            output_data += ' '
                '''
                input_datum = '1\n'
                for k in range(hw):
                    for l in range(hw):
                        input_datum += '0' if o.grid[k][l] == black else '1' if o.grid[k][l] == white else '.'
                    input_datum += '\n'
                egaroucid.stdin.write((str(0) + '\n' + input_datum).encode('utf-8'))
                egaroucid.stdin.flush()
                line = egaroucid.stdout.readline().decode()
                val = int(line)
                output_data += " " + str(val)
                #egaroucid.stdin.write('-1\n'.encode('utf-8'))
                f.write(datum + ' ' + output_data + '\n')

egaroucid.kill()