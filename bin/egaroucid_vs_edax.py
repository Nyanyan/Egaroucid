import subprocess
from othello_py import *
import sys
from random import shuffle

with open('problem/xot_small_shuffled.txt', 'r') as f:
    tactic = [elem for elem in f.read().splitlines()]
print(len(tactic), 'openings found')

level = int(sys.argv[1])
n_games = int(sys.argv[2])


file = None
cmd = 'Egaroucid_for_console.exe -quiet -nobook -level ' + str(level)
if len(sys.argv) == 4:
    file = sys.argv[3]
    print('egaroucid eval ', file)
    cmd += ' -eval ' + file

#if level <= 2:
#    cmd += ' -t 1'

print(cmd)
egaroucid = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
egaroucid_win = [0, 0]
edax_win = [0, 0]
draw = [0, 0]

print('level', level)
print('openings', len(tactic))

max_num = min(len(tactic), n_games)
smpl = range(len(tactic))
print('play', max_num, 'games')

edax = subprocess.Popen(('wEdax-x64-modern.exe -q -level ' + str(level)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

for num in range(max_num):
    tactic_idx = smpl[num % len(tactic)]
    shuffled_range2 = [0, 1]
    shuffle(shuffled_range2)
    for player in shuffled_range2:
        #edax = subprocess.Popen(('wEdax-x64-modern.exe -q -level ' + str(level)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        record = ''
        boards = []
        o = othello()
        for i in range(0, len(tactic[tactic_idx]), 2):
            if not o.check_legal():
                o.player = 1 - o.player
                o.check_legal()
            x = ord(tactic[tactic_idx][i].lower()) - ord('a')
            y = int(tactic[tactic_idx][i + 1]) - 1
            record += tactic[tactic_idx][i] + tactic[tactic_idx][i + 1]
            o.move(y, x)
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
            if o.player == player:
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
                    print(o.player, player)
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
                    print(o.player, player)
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
                print(o.player, player)
                print(coord)
                print(y, x)
        if o.n_stones[player] > o.n_stones[1 - player]:
            egaroucid_win[player] += 1
        elif o.n_stones[player] == o.n_stones[1 - player]:
            draw[player] += 1
        else:
            edax_win[player] += 1
            #print(record)
        print('\r', num, max_num, ' ', egaroucid_win, draw, edax_win, sum(egaroucid_win) + sum(draw) * 0.5, sum(edax_win) + sum(draw) * 0.5, 
              (sum(egaroucid_win) + sum(draw) * 0.5) / max(1, sum(egaroucid_win) + sum(edax_win) + sum(draw)), end='                ')
        #egaroucid.stdin.write('clearcache\n'.encode('utf-8'))
        #egaroucid.stdin.flush()
        #edax.kill()

egaroucid.stdin.write('quit\n'.encode('utf-8'))
egaroucid.stdin.flush()
edax.stdin.write('quit\n'.encode('utf-8'))
edax.stdin.flush()
print('')

print('level: ', level, 
      ' Egaroucid plays black WDL: ', egaroucid_win[0], '-', draw[0], '-', edax_win[0], ' ', (egaroucid_win[0] + draw[0] * 0.5) / (egaroucid_win[0] + edax_win[0] + draw[0]), 
      ' Egaroucid plays white WDL: ', egaroucid_win[1], '-', draw[1], '-', edax_win[1], ' ', (egaroucid_win[1] + draw[1] * 0.5) / (egaroucid_win[1] + edax_win[1] + draw[1]), 
      ' Egaroucid win rate: ', (sum(egaroucid_win) + sum(draw) * 0.5) / max(1, sum(egaroucid_win) + sum(edax_win) + sum(draw)), sep='')
