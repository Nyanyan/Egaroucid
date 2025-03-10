import subprocess
from tqdm import trange
from random import shuffle
import matplotlib.pyplot as plt
import sys
from othello_py import *


LEVEL = int(sys.argv[1])
N_SET_GAMES = int(sys.argv[2])

# name, cmd
player_info = [
    #['beta', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -t 32'],
    #['latest',  'Egaroucid_for_Console.exe -quiet -nobook -t 32'],
    ['clang',  'Egaroucid_for_Console_clang.exe -quiet -nobook -t 32'],
    ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -quiet -nobook -t 32'],
    ['7.4.0', 'versions/Egaroucid_for_Console_7_4_0_Windows_x64_SIMD/Egaroucid_for_Console_7_4_0_x64_SIMD.exe -quiet -nobook -t 32'],
    ['7.3.0', 'versions/Egaroucid_for_Console_7_3_0_Windows_x64_SIMD/Egaroucid_for_Console_7_3_0_x64_SIMD.exe -quiet -nobook -t 32'],
    ['7.2.0', 'versions/Egaroucid_for_Console_7_2_0_Windows_x64_SIMD/Egaroucid_for_Console_7_2_0_x64_SIMD.exe -quiet -nobook -t 32'],
    ['7.1.0', 'versions/Egaroucid_for_Console_7_1_0_Windows_x64_SIMD/Egaroucid_for_Console_7_1_0_x64_SIMD.exe -quiet -nobook -t 32'],
    ['7.0.0', 'versions/Egaroucid_for_Console_7_0_0_Windows_x64_SIMD/Egaroucid_for_Console_7_0_0_x64_SIMD.exe -quiet -nobook -t 32'],
    ['6.5.X', 'versions/Egaroucid_for_Console_6_5_X/Egaroucid_for_Console.exe -quiet -nobook -t 32'],
        #['6.5.0', 'versions/Egaroucid_for_Console_6_5_0_Windows_x64_SIMD/Egaroucid_for_Console_6_5_0_x64_SIMD.exe -quiet -nobook'],
    ['6.4.X', 'versions/Egaroucid_for_Console_6_4_X/Egaroucid_for_Console.exe -quiet -nobook -t 32'],
        #['6.4.0', 'versions/Egaroucid_for_Console_6_4_0_Windows_x64_SIMD/Egaroucid_for_Console_6_4_0_x64_SIMD.exe -quiet -nobook'],
    ['6.3.X', 'versions/Egaroucid_for_Console_6_3_X/Egaroucid_for_Console.exe -quiet -nobook -t 32'],
        #['6.3.0', 'versions/Egaroucid_for_Console_6_3_0_Windows_x64_SIMD/Egaroucid_for_Console_6_3_0_x64_SIMD.exe -quiet -nobook'],
        #['6.2.X', 'versions/Egaroucid_for_Console_6_2_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.2.0', 'versions/Egaroucid_for_Console_6_2_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook'], # same as 6.3.0
    ['6.1.X', 'versions/Egaroucid_for_Console_6_1_X/Egaroucid_for_Console.exe -quiet -nobook -t 32'],
        #['6.1.0', 'versions/Egaroucid_for_Console_6_1_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook'],
    ['6.0.X', 'versions/Egaroucid_for_Console_6_0_X/Egaroucid_for_Console_test.exe q'],
    ['Edax ', 'versions/edax_4_4/edax-4.4 -q -n 32'],
]

NAME_IDX = 0
SUBPROCESS_IDX = 1
RESULT_IDX = 2
RESULT_DISC_IDX = 3
N_PLAYED_IDX = 4

players = []
for name, cmd in player_info:
    cmd_with_level = cmd + ' -l ' + str(LEVEL)
    if name == '6.0.X':
        cmd_with_level = cmd + ' ' + str(LEVEL)
    print(name, cmd_with_level)
    players.append([
        name,
        [
            subprocess.Popen(cmd_with_level.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
            subprocess.Popen(cmd_with_level.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ],
        # W D L (vs other players)
        [[0, 0, 0] for _ in range(len(player_info))],
        # sum of disc differences (vs other players)
        [0 for _ in range(len(player_info))],
        # n_played
        [0 for _ in range(len(player_info))]
    ])

with open('problem/xot/openingslarge.txt', 'r') as f:
    openings = [elem for elem in f.read().splitlines()]
shuffle(openings)

def play_battle(p0_idx, p1_idx, opening_idx):
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    shuffled_range2 = [0, 1]
    shuffle(shuffled_range2)
    sum_disc_diff_p0 = 0
    for player in shuffled_range2: # which plays black. p0 plays `player`, p1 plays `1 - player`
        record = ''
        o = othello()
        # play opening
        for i in range(0, len(opening), 2):
            if not o.check_legal():
                o.player = 1 - o.player
                o.check_legal()
            x = ord(opening[i].lower()) - ord('a')
            y = int(opening[i + 1]) - 1
            record += opening[i] + opening[i + 1]
            o.move(y, x)
        # play with ai
        while True:
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    break
            grid_str = 'setboard '
            for yy in range(hw):
                for xx in range(hw):
                    if o.grid[yy][xx] == black:
                        grid_str += 'X'
                    elif o.grid[yy][xx] == white:
                        grid_str += 'O'
                    else:
                        grid_str += '-'
            if o.player == black:
                grid_str += ' X\n'
            else:
                grid_str += ' O\n'
            player_idx = player_idxes[o.player ^ player]
            players[player_idx][SUBPROCESS_IDX][player].stdin.write(grid_str.encode('utf-8'))
            players[player_idx][SUBPROCESS_IDX][player].stdin.flush()
            players[player_idx][SUBPROCESS_IDX][player].stdin.write('go\n'.encode('utf-8'))
            players[player_idx][SUBPROCESS_IDX][player].stdin.flush()
            line = ''
            while line == '' or line == '>':
                line = players[player_idx][SUBPROCESS_IDX][player].stdout.readline().decode().replace('\r', '').replace('\n', '')
            coord = line[-2:].lower()
            try:
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
            except:
                print('error')
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                for i in range(len(players)):
                    for j in range(2):
                        players[i][SUBPROCESS_IDX][j].stdin.write('quit\n'.encode('utf-8'))
                        players[i][SUBPROCESS_IDX][j].stdin.flush()
                exit()
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                print(y, x)
        # update win/draw/loss
        if o.n_stones[player] > o.n_stones[1 - player]: # p0 win
            sum_disc_diff_p0 += o.n_stones[player] - o.n_stones[1 - player] + (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        elif o.n_stones[player] < o.n_stones[1 - player]: # p0 lose
            sum_disc_diff_p0 += o.n_stones[player] - o.n_stones[1 - player] - (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        else:
            sum_disc_diff_p0 += 0
    if sum_disc_diff_p0 > 0: # p0 win
        players[p0_idx][RESULT_IDX][p1_idx][0] += 1 # win
        players[p1_idx][RESULT_IDX][p0_idx][2] += 1 # loss
    elif sum_disc_diff_p0 < 0: # p0 lose
        players[p1_idx][RESULT_IDX][p0_idx][0] += 1 # win
        players[p0_idx][RESULT_IDX][p1_idx][2] += 1 # loss
    else:
        players[p0_idx][RESULT_IDX][p1_idx][1] += 1 # draw
        players[p1_idx][RESULT_IDX][p0_idx][1] += 1 # draw
    # update disc difference
    players[p0_idx][RESULT_DISC_IDX][p1_idx] += sum_disc_diff_p0 / 2
    players[p1_idx][RESULT_DISC_IDX][p0_idx] -= sum_disc_diff_p0 / 2
    players[p0_idx][N_PLAYED_IDX][p1_idx] += 1
    players[p1_idx][N_PLAYED_IDX][p0_idx] += 1

'''
def print_result():
    for i in range(len(players)):
        w = 0
        d = 0
        l = 0
        for ww, dd, ll in players[i][RESULT_IDX]:
            w += ww
            d += dd
            l += ll
        r = (w + d * 0.5) / (w + d + l)
        print(i, players[i][NAME_IDX], w + d + l, w, d, l, r, sep='\t')
'''

def print_all_result():
    print('Win Rate')
    print('vs >', end='\t')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        print(name, end='\t')
    print('all')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_IDX]
        print(name, end='\t')
        # each
        for j in range(len(players)):
            if i == j:
                print('-', end='\t')
            else:
                w, d, l = result[j]
                r = (w + d * 0.5) / max(1, w + d + l)
                print("{:.4f}".format(r), end='\t')
        # all
        w = 0
        d = 0
        l = 0
        for ww, dd, ll in result:
            w += ww
            d += dd
            l += ll
        r = (w + d * 0.5) / max(1, w + d + l)
        print("{:.4f}".format(r))

    print('Average Disc Difference')
    print('vs >', end='\t')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        print(name, end='\t')
    print('all')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_DISC_IDX]
        n_played = players[i][N_PLAYED_IDX]
        print(name, end='\t')
        # each
        for j in range(len(players)):
            if i == j:
                print('-', end='\t')
            else:
                avg_discs = result[j] / max(1, n_played[j])
                s = "{:.2f}".format(avg_discs)
                if avg_discs >= 0:
                    s = '+' + s
                print(s, end='\t')
        # all
        avg_discs_all = sum(result) / max(1, sum(n_played))
        s = "{:.2f}".format(avg_discs_all)
        if avg_discs_all >= 0:
            s = '+' + s
        print(s)


plot_data = [[] for _ in range(len(players))]

def output_plt():
    for i in range(len(players)):
        w = 0
        d = 0
        l = 0
        for ww, dd, ll in players[i][RESULT_IDX]:
            w += ww
            d += dd
            l += ll
        r = (w + d * 0.5) / max(1, w + d + l)
        plot_data[i].append(r)
        name = players[i][NAME_IDX]
        plt.plot(plot_data[i], label=name)
    plt.xlabel('n_battles')
    plt.ylabel('win rate (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3)
    plt.savefig('graph.png', bbox_inches='tight')
    plt.clf()

print('n_players', len(players))
print('level', LEVEL)


matches = []
for p0 in range(len(players)):
    for p1 in range(p0 + 1, len(players)):
        matches.append([p0, p1])

'''
matches = []
p0 = 0
for p1 in range(p0 + 1, len(players)):
    matches.append([p0, p1])
'''

problem_idx = 0
for i in range(N_SET_GAMES):
    shuffle(matches)
    for p0, p1 in matches:
        play_battle(p0, p1, problem_idx)
        problem_idx += 1
        problem_idx %= len(openings)
    print(i, 'level', LEVEL)
    #print_result()
    print_all_result()
    #output_plt()

print(N_SET_GAMES, 'matches played for each win rate at level', LEVEL)
print_all_result()


for i in range(len(players)):
    for j in range(2):
        players[i][SUBPROCESS_IDX][j].stdin.write('quit\n'.encode('utf-8'))
        players[i][SUBPROCESS_IDX][j].stdin.flush()

for i in range(len(players)):
    for j in range(2):
        players[i][SUBPROCESS_IDX][j].kill()