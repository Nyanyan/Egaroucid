import subprocess
from tqdm import trange
from random import shuffle
import matplotlib.pyplot as plt
from othello_py import *

LEVEL = 1

N_SET_GAMES = 100

# name, cmd, cacheclear?
player_info = [
    #['beta ', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook', True],
    ['6.5.0', 'versions/Egaroucid_for_Console_6_5_0_Windows_x64_SIMD/Egaroucid_for_Console_6_5_0_x64_SIMD.exe -quiet -nobook', False],
    ['6.4.0', 'versions/Egaroucid_for_Console_6_4_0_Windows_x64_SIMD/Egaroucid_for_Console_6_4_0_x64_SIMD.exe -quiet -nobook', False],
    ['6.2.0/6.3.0', 'versions/Egaroucid_for_Console_6_3_0_Windows_x64_SIMD/Egaroucid_for_Console_6_3_0_x64_SIMD.exe -quiet -nobook', False],
    #['6.2.0', 'versions/Egaroucid_for_Console_6_2_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook', False],
    ['6.1.0', 'versions/Egaroucid_for_Console_6_1_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook', False],
    ['Edax ', 'versions/edax_4_4/edax-4.4 -q', False],
]

NAME_IDX = 0
SUBPROCESS_IDX = 1
RESULT_IDX = 2

players = []
for name, cmd, _ in player_info:
    cmd_with_level = cmd + ' -l ' + str(LEVEL)
    print(name, cmd_with_level)
    players.append(
        [
            name, 
            subprocess.Popen(cmd_with_level.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
            # W D L
            [[0, 0, 0] for _ in range(len(player_info))]
        ]
    )

with open('problem/xot_small_shuffled.txt', 'r') as f:
    openings = [elem for elem in f.read().splitlines()]

#shuffle(openings)

def play_battle(p0_idx, p1_idx, opening_idx):
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    shuffled_range2 = [0, 1]
    shuffle(shuffled_range2)
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
                        grid_str += 'b'
                    elif o.grid[yy][xx] == white:
                        grid_str += 'w'
                    else:
                        grid_str += '.'
            if o.player == black:
                grid_str += ' b\n'
            else:
                grid_str += ' w\n'
            player_idx = player_idxes[o.player ^ player]
            players[player_idx][SUBPROCESS_IDX].stdin.write(grid_str.encode('utf-8'))
            players[player_idx][SUBPROCESS_IDX].stdin.flush()
            players[player_idx][SUBPROCESS_IDX].stdin.write('go\n'.encode('utf-8'))
            players[player_idx][SUBPROCESS_IDX].stdin.flush()
            line = ''
            while line == '' or line == '>':
                line = players[player_idx][SUBPROCESS_IDX].stdout.readline().decode().replace('\r', '').replace('\n', '')
            coord = line[-2:].lower()
            try:
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
            except:
                print('error')
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                for i in range(2):
                    players[i][SUBPROCESS_IDX].stdin.write('quit\n'.encode('utf-8'))
                    players[i][SUBPROCESS_IDX].stdin.flush()
                exit()
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                print(y, x)
        if o.n_stones[player] > o.n_stones[1 - player]:
            players[p0_idx][RESULT_IDX][p1_idx][0] += 1
            players[p1_idx][RESULT_IDX][p0_idx][2] += 1
        elif o.n_stones[player] < o.n_stones[1 - player]:
            players[p1_idx][RESULT_IDX][p0_idx][0] += 1
            players[p0_idx][RESULT_IDX][p1_idx][2] += 1
        else:
            players[p0_idx][RESULT_IDX][p1_idx][1] += 1
            players[p1_idx][RESULT_IDX][p0_idx][1] += 1
        
        #for pidx in [p0_idx, p1_idx]:
        #    if player_info[pidx][2]:
        #        players[pidx][SUBPROCESS_IDX].stdin.write('clearcache\n'.encode('utf-8'))
        #        players[pidx][SUBPROCESS_IDX].stdin.flush()
        #    else:
        #        players[pidx][SUBPROCESS_IDX].kill()
        #        cmd_with_level = player_info[pidx][1] + ' -l ' + str(LEVEL)
        #        players[pidx][SUBPROCESS_IDX] = subprocess.Popen(cmd_with_level.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


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

def print_all_result():
    print('', end='\t')
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
                r = (w + d * 0.5) / (w + d + l)
                print("{:.4f}".format(r), end='\t')
        # all
        w = 0
        d = 0
        l = 0
        for ww, dd, ll in result:
            w += ww
            d += dd
            l += ll
        r = (w + d * 0.5) / (w + d + l)
        print("{:.4f}".format(r))


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
        r = (w + d * 0.5) / (w + d + l)
        plot_data[i].append(r)
        name = players[i][NAME_IDX]
        plt.plot(plot_data[i], label=name)
    plt.xlabel('n_battles')
    plt.ylabel('win rate (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3)
    plt.savefig('graph.png', bbox_inches='tight')
    plt.clf()

print('n_players', len(players))

for i in range(N_SET_GAMES):
    for p0 in range(len(players)):
        for p1 in range(p0 + 1, len(players)):
            play_battle(p0, p1, i)
    print(i)
    print_result()
    print_all_result()
    #output_plt()

print(N_SET_GAMES * 2, 'games played for each winning rate')
print_all_result()


for i in range(len(players)):
    players[i][SUBPROCESS_IDX].stdin.write('quit\n'.encode('utf-8'))
    players[i][SUBPROCESS_IDX].stdin.flush()
