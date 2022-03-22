import subprocess
from othello_py import *
from random import shuffle

with open('close_quest.txt', 'r') as f:
    tactic = [elem for elem in f.read().splitlines()]

level = 5

egaroucid = [
    subprocess.Popen(('a.exe ' + str(level)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
    subprocess.Popen(('b.exe ' + str(level)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
]

results = [0, 0, 0]

use_len = 20

tactic_set = set()
for t in tactic:
    if len(t) >= use_len * 2:
        tactic_set.add(t[:use_len * 2])
tactic = list(tactic_set)
shuffle(tactic)
print(len(tactic))

max_num = min(len(tactic), 200)

smpl = range(len(tactic))

print(len(smpl))

for num in range(max_num):
    tactic_idx = smpl[num % len(tactic)]
    tactic_use_idx = use_len * 2
    for player in range(2):
        record = ''
        boards = []
        o = othello()
        for i in range(0, tactic_use_idx, 2):
            if not o.check_legal():
                o.player = 1 - o.player
                o.check_legal()
            x = ord(tactic[tactic_idx][i]) - ord('a')
            y = int(tactic[tactic_idx][i + 1]) - 1
            record += tactic[tactic_idx][i] + tactic[tactic_idx][i + 1]
            o.move(y, x)
        while True:
            #o.print_info()
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    break
            player_idx = 0 if o.player == player else 1
            grid_str = str(o.player) + '\n'
            for yy in range(hw):
                for xx in range(hw):
                    if o.grid[yy][xx] == black:
                        grid_str += '0'
                    elif o.grid[yy][xx] == white:
                        grid_str += '1'
                    else:
                        grid_str += '.'
                grid_str += '\n'
            #print(grid_str)
            egaroucid[player_idx].stdin.write(grid_str.encode('utf-8'))
            egaroucid[player_idx].stdin.flush()
            _, coord = egaroucid[player_idx].stdout.readline().decode().split()
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('a')
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str)
                print(y, x)
        if o.n_stones[player] > o.n_stones[1 - player]:
            results[0] += 1
        elif o.n_stones[player] == o.n_stones[1 - player]:
            results[1] += 1
        else:
            results[2] += 1
            #print(record)
        print('\r', num, results, end='')
egaroucid[0].kill()
egaroucid[1].kill()
print('')

print('start depth: ', use_len, ' a.exe WDL: ', results[0], '-', results[1], '-', results[2], ' a.exe win rate: ', results[0] / (results[0] + results[2]), ' b.exe win rate: ', results[2] / (results[0] + results[2]), sep='')
