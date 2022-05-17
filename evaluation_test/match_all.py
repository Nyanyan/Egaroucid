import subprocess
from random import randrange, sample, shuffle
from othello_py import *
from matplotlib import pyplot

with open('close_quest.txt', 'r') as f:
    tactic = [elem for elem in f.read().splitlines()]

#shuffle(tactic)

level = 21

egaroucid = subprocess.Popen(('b.exe ' + str(level)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
edax = subprocess.Popen('edax-4.4', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
edax.stdin.write(('level ' + str(level) + '\n').encode('utf-8'))
edax.stdin.flush()
for i in range(25):
    edax.stdout.readline()

egaroucid_win = [0, 0]
edax_win = [0, 0]
draw = [0, 0]



#  0  1   2  3    4   5   6   7   8   9
# [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]

#  0  1  2   3   4   5   6   7   8   9  10  11  12  13  14
# [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]

use_len = 10

tactic_set = set()
for t in tactic:
    if len(t) >= use_len * 2:
        tactic_set.add(t[:use_len * 2])
tactic = list(tactic_set)
#tactic = ['' for _ in range(100)]
print(len(tactic))

#max_num = len(tactic)
max_num = min(len(tactic), 10)
#while max_num < 50:
#    max_num *= 2

smpl = range(len(tactic))
#smpl = sample(range(len(tactic)), max_num)

print(len(smpl))

plot_x = range(max_num)
plot_eg = [[], [], []]
plot_ed = [[], [], []]

for num in range(max_num):
    #tactic_idx = randrange(0, len(tactic))
    tactic_idx = smpl[num % len(tactic)]
    tactic_use_idx = use_len * 2 #randrange(0, len(tactic[tactic_idx]) + 1)
    #while len(tactic[tactic_idx]) < tactic_use_idx:
    #    tactic_idx = randrange(0, len(tactic))
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
            o.print_info()
            former_player = o.player
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    if former_player == player:
                        edax.stdout.readline()
                        edax.stdout.readline()
                    break
            if o.player == player:
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
                egaroucid.stdin.write(grid_str.encode('utf-8'))
                egaroucid.stdin.flush()
                '''
                y, x, _ = [float(elem) for elem in egaroucid.stdout.readline().decode().split()]
                y = int(y)
                x = int(x)
                '''
                _, coord = egaroucid.stdout.readline().decode().split()
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
                board = ''
                for yy in range(hw):
                    for xx in range(hw):
                        if o.grid[yy][xx] == black:
                            board += '0'
                        elif o.grid[yy][xx] == white:
                            board += '1'
                        else:
                            board += '.'
                boards.append([board, o.player, -1000])
                #print('eg', y, x)
            else:
                grid_str = 'setboard '
                for yy in range(hw):
                    for xx in range(hw):
                        if o.grid[yy][xx] == black:
                            grid_str += 'B'
                        elif o.grid[yy][xx] == white:
                            grid_str += 'W'
                        else:
                            grid_str += '.'
                grid_str += 'B' if o.player == black else 'W'
                grid_str += '\n'
                #print(grid_str)
                edax.stdin.write(grid_str.encode('utf-8'))
                edax.stdin.flush()
                edax.stdin.write('go\n'.encode('utf-8'))
                edax.stdin.flush()
                for _ in range(16):
                    edax.stdout.readline()
                score = int(edax.stdout.readline().decode().split()[1])
                if o.player == white:
                    score = -score
                for _ in range(2):
                    edax.stdout.readline()
                edax_out = edax.stdout.readline().decode()
                edax_str = edax_out.split()[2]
                for _ in range(12):
                    edax.stdout.readline()
                x = ord(edax_str[0]) - ord('A')
                y = int(edax_str[1]) - 1
                #print('ed', y, x)
                board = ''
                for yy in range(hw):
                    for xx in range(hw):
                        if o.grid[yy][xx] == black:
                            board += '0'
                        elif o.grid[yy][xx] == white:
                            board += '1'
                        else:
                            board += '.'
                boards.append([board, o.player, score])
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str)
                print(y, x)
        o.print_info()
        if o.n_stones[player] > o.n_stones[1 - player]:
            egaroucid_win[player] += 1
        elif o.n_stones[player] == o.n_stones[1 - player]:
            draw[player] += 1
        else:
            edax_win[player] += 1
            #print(record)
        plot_eg[player].append(egaroucid_win[player])
        plot_ed[player].append(edax_win[player])
        print('\r', num, ' ', egaroucid_win, draw, edax_win, sum(egaroucid_win), sum(edax_win), sum(egaroucid_win) / max(1, sum(egaroucid_win) + sum(edax_win)), ' ', tactic_use_idx // 2, tactic_idx, end='')
        '''
        with open('records.txt', 'a') as f:
            f.write(record + '\n')
        for i in range(len(boards)):
            if boards[i][2] == -1000:
                for j in range(i, len(boards)):
                    if boards[j][2] != -1000:
                        boards[i][2] = boards[j][2]
                        break
                else:
                    for j in reversed(range(i)):
                        if boards[j][2] != -1000:
                            boards[i][2] = boards[j][2]
                            break
        with open('data.txt', 'a') as f:
            for b, p, s in boards:
                f.write(b + ' ' + str(p) + ' ' + str(s) + '\n')
        '''
    plot_eg[2].append(sum(egaroucid_win))
    plot_ed[2].append(sum(edax_win))
egaroucid.kill()
edax.kill()
print('')

print('start depth: ', use_len, ' Egaroucid plays black WDL: ', egaroucid_win[0], '-', draw[0], '-', edax_win[0], ' Egaroucid plays white WDL: ', egaroucid_win[1], '-', draw[1], '-', edax_win[1], ' Egaroucid win rate: ', sum(egaroucid_win) / max(1, sum(egaroucid_win) + sum(edax_win)), sep='')

pyplot.plot(plot_x, plot_ed[0], label='edax_white')
pyplot.plot(plot_x, plot_eg[0], label='egaroucid_black')
pyplot.plot(plot_x, plot_ed[1], label='edax_black')
pyplot.plot(plot_x, plot_eg[1], label='egaroucid_white')
pyplot.plot(plot_x, plot_ed[2], label='edax_all')
pyplot.plot(plot_x, plot_eg[2], label='egaroucid_all')
pyplot.legend()
pyplot.show()