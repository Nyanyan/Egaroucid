import subprocess
from tqdm import trange
from othello_py import *
import sys
from random import randrange

N_GAMES_PER_FILE = 10000
N_THREAD = 15

LEVEL = int(sys.argv[1])
IDX_START = int(sys.argv[2])
IDX_END = int(sys.argv[3])


exe = './Egaroucid_for_Console_clang.exe'
cmd0 = exe + ' -quiet -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD)
cmd1 = exe + ' -quiet -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -nobook'
egaroucid = [
    subprocess.Popen(cmd0.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
    subprocess.Popen(cmd1.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
]

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

for idx in range(IDX_START, IDX_END + 1, 2):
    print(fill0(idx, 7))
    files = [
        './transcript/selfplay_book/' + fill0(idx, 7) + '.txt',
        './transcript/selfplay_book/' + fill0(idx + 1, 7) + '.txt',
    ]
    for i in trange(N_GAMES_PER_FILE):
        n_book_moves = randrange(0, 31)
        for j in range(2):
            o = othello()
            o.check_legal()
            o.move(4, 5) # f5
            record = 'f5'
            while True:
                if not o.check_legal():
                    o.player = 1 - o.player
                    if not o.check_legal():
                        break
                grid_str = 'setboard '
                for yy in range(hw):
                    for xx in range(hw):
                        if o.grid[yy][xx] == black:
                            grid_str += 'b'
                        elif o.grid[yy][xx] == white:
                            grid_str += 'w'
                        else:
                            grid_str += '.'
                if o.player == black:
                    grid_str += ' b\n'
                else:
                    grid_str += ' w\n'
                egaroucid_idx = o.player ^ j
                if sum(o.n_stones) < n_book_moves + 4:
                    egaroucid_idx = 0
                egaroucid[egaroucid_idx].stdin.write(grid_str.encode('utf-8'))
                egaroucid[egaroucid_idx].stdin.flush()
                egaroucid[egaroucid_idx].stdin.write('go\n'.encode('utf-8'))
                egaroucid[egaroucid_idx].stdin.flush()
                line = ''
                while line == '':
                    line = egaroucid[egaroucid_idx].stdout.readline().decode().replace('\r', '').replace('\n', '')
                coord = line
                try:
                    y = int(coord[1]) - 1
                    x = ord(coord[0]) - ord('a')
                except:
                    print('error')
                    print(grid_str[:-1])
                    print(o.player, j)
                    print(coord)
                    for k in range(2):
                        egaroucid[k].stdin.write('quit\n'.encode('utf-8'))
                        egaroucid[k].stdin.flush()
                    exit()
                record += chr(ord('a') + x) + str(y + 1)
                if not o.move(y, x):
                    o.print_info()
                    print(grid_str[:-1])
                    print(o.player, player)
                    print(coord)
                    print(y, x)
            with open(files[j], 'a') as f:
                f.write(record + '\n')

for i in range(2):
    egaroucid[i].stdin.write('quit\n'.encode('utf-8'))
    egaroucid[i].stdin.flush()