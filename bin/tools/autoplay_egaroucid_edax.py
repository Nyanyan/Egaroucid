import subprocess
from tqdm import trange
import sys
import os
from othello_py2 import *
import random

n_random_moves = int(sys.argv[1])
IDX_START = int(sys.argv[2])
IDX_END = int(sys.argv[3])

LEVEL = 11
N_GAMES_PER_FILE = 10000
N_THREAD = 31

exe_egaroucid = './../versions/Egaroucid_for_Console_7_7_0_Windows_SIMD/Egaroucid_for_Console_7_7_0_SIMD.exe'
# exe_egaroucid = './../Egaroucid_for_Console.exe'
exe_edax = './../versions/edax_4_5_5/bin/wEdax-x86-64-v3.exe'

cmd_egaroucid = exe_egaroucid + ' -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -quiet'
print(cmd_egaroucid)

cmd_edax = exe_edax + ' -l ' + str(LEVEL) + ' -q'
print(cmd_edax)


egaroucid = subprocess.Popen(cmd_egaroucid.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
edax = subprocess.Popen(cmd_edax.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    dr = './../transcript/' + str(n_random_moves)
    try:
        os.mkdir(dr)
    except:
        pass

    for game_idx in trange(N_GAMES_PER_FILE):
        egaroucid_plays_black = game_idx % 2
        valid_opening_found = False
        n_try = 0
        while not valid_opening_found:
            valid_opening_found = True
            record = ''
            o = Othello()
            for i in range(n_random_moves):
                if not o.has_legal():
                    o.move_pass()
                    if not o.has_legal():
                        valid_opening_found = False
                        break
                legals = o.get_legal_moves()
                legal_move = random.choice(legals)
                y, x = legal_move
                record += chr(ord('a') + x) + str(y + 1)
                o.move(y, x)
            n_try += 1
        # if n_try > 1:
        #     print(n_try, record)
        while True:
            if not o.has_legal():
                o.move_pass()
                if not o.has_legal():
                    break
            grid_str = 'setboard '
            for yy in range(HW):
                for xx in range(HW):
                    if o.grid[yy][xx] == BLACK:
                        grid_str += 'b'
                    elif o.grid[yy][xx] == WHITE:
                        grid_str += 'w'
                    else:
                        grid_str += '.'
            if o.player == BLACK:
                grid_str += ' b\n'
            else:
                grid_str += ' w\n'
            egaroucid_turn = False
            if egaroucid_plays_black and o.player == BLACK:
                egaroucid_turn = True
            elif (not egaroucid_plays_black) and o.player == WHITE:
                egaroucid_turn = True
            if egaroucid_turn:
                egaroucid.stdin.write(grid_str.encode('utf-8'))
                egaroucid.stdin.flush()
                egaroucid.stdin.write('go\n'.encode('utf-8'))
                egaroucid.stdin.flush()
                line = ''
                while line == '':
                    line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '')
                coord = line
                try:
                    y = int(coord[1]) - 1
                    x = ord(coord[0]) - ord('a')
                except:
                    print('error')
                    print(grid_str[:-1])
                    print(o.player, egaroucid_plays_black)
                    print(coord)
                    egaroucid.stdin.write('quit\n'.encode('utf-8'))
                    egaroucid.stdin.flush()
                    edax.stdin.write('quit\n'.encode('utf-8'))
                    edax.stdin.flush()
                    exit()
            else:
                edax.stdin.write(grid_str.encode('utf-8'))
                edax.stdin.flush()
                edax.stdin.write('go\n'.encode('utf-8'))
                edax.stdin.flush()
                line = ''
                while len(line) < 3:
                    line = edax.stdout.readline().decode().replace('\r', '').replace('\n', '')
                try:
                    coord = line.split()[2]
                    y = int(coord[1]) - 1
                    x = ord(coord[0]) - ord('A')
                except:
                    print('error')
                    print(grid_str[:-1])
                    print(o.player, egaroucid_plays_black)
                    print(coord)
                    egaroucid.stdin.write('quit\n'.encode('utf-8'))
                    egaroucid.stdin.flush()
                    edax.stdin.write('quit\n'.encode('utf-8'))
                    edax.stdin.flush()
                    exit()
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str[:-1])
                print(o.player, egaroucid_plays_black)
                print(coord)
                print(y, x)
        with open(dr + '/' + fill0(idx, 7) + '.txt', 'a') as f:
            f.write(record + '\n')

