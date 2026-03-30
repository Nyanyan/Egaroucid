import subprocess
from tqdm import trange
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
from othello_py import *
from elo_rating import Elo_player, update_rating, update_rating_draw
from elo_rating_backcal import fit_elo_from_winrates_with_interval

LEVEL = int(sys.argv[1])
N_SET_GAMES = int(sys.argv[2])
N_THREADS = 32

random.seed(57)

with open('problem/xot/openingslarge.txt', 'r') as f:
    openings = [elem for elem in f.read().splitlines()]
random.shuffle(openings)

# name, cmd
player_info = [
    # ['0325', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260325_1/eval.egev2'],
    # ['0324', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260324_1/eval.egev2'],
    # ['0323', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260323_1/eval.egev2'],
    # ['0322', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260322_1/eval.egev2'],
    # ['0321', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260321_1/eval.egev2'],
    # ['0320', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260320_1/eval.egev2'],
    # ['0318', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260318_1/eval.egev2'],
    # ['0317', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook -eval ./../model/20260317_1/eval.egev2'],
    ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -gtp -quiet -nobook'],
    ['Edax4.6', 'versions/edax_4_6/wEdax-x86-64-v3.exe -gtp -q'],
    ['Neural5', 'versions/neural-reversi-cli-5.0.0-windows-x86_64-v3.exe gtp'],
    ['Ntest', 'versions/ntest/ntest.exe --gtp'],



    # ['beta', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    # ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.7.0', 'versions/Egaroucid_for_Console_7_7_0_Windows_SIMD/Egaroucid_for_Console_7_7_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.4.0', 'versions/Egaroucid_for_Console_7_4_0_Windows_x64_SIMD/Egaroucid_for_Console_7_4_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.3.0', 'versions/Egaroucid_for_Console_7_3_0_Windows_x64_SIMD/Egaroucid_for_Console_7_3_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.2.0', 'versions/Egaroucid_for_Console_7_2_0_Windows_x64_SIMD/Egaroucid_for_Console_7_2_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.1.0', 'versions/Egaroucid_for_Console_7_1_0_Windows_x64_SIMD/Egaroucid_for_Console_7_1_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.0.0', 'versions/Egaroucid_for_Console_7_0_0_Windows_x64_SIMD/Egaroucid_for_Console_7_0_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['Edax4.6', 'versions/edax_4_6/wEdax-x86-64-v3.exe -gtp -q'],

    # ['beta', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    # ['latest',  'Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    # ['clang',  'Egaroucid_for_Console_clang.exe -gtp -quiet -nobook'],
    # ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.7.0', 'versions/Egaroucid_for_Console_7_7_0_Windows_SIMD/Egaroucid_for_Console_7_7_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -gtp -quiet -nobook'],
    # ['7.4.0', 'versions/Egaroucid_for_Console_7_4_0_Windows_x64_SIMD/Egaroucid_for_Console_7_4_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.3.0', 'versions/Egaroucid_for_Console_7_3_0_Windows_x64_SIMD/Egaroucid_for_Console_7_3_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.2.0', 'versions/Egaroucid_for_Console_7_2_0_Windows_x64_SIMD/Egaroucid_for_Console_7_2_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.1.0', 'versions/Egaroucid_for_Console_7_1_0_Windows_x64_SIMD/Egaroucid_for_Console_7_1_0_x64_SIMD.exe -gtp -quiet -nobook'],
    # ['7.0.0', 'versions/Egaroucid_for_Console_7_0_0_Windows_x64_SIMD/Egaroucid_for_Console_7_0_0_x64_SIMD.exe -gtp -quiet -nobook'],
    #['6.5.X', 'versions/Egaroucid_for_Console_6_5_X/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
        #['6.5.0', 'versions/Egaroucid_for_Console_6_5_0_Windows_x64_SIMD/Egaroucid_for_Console_6_5_0_x64_SIMD.exe -gtp -quiet -nobook'],
    #['6.4.X', 'versions/Egaroucid_for_Console_6_4_X/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
        #['6.4.0', 'versions/Egaroucid_for_Console_6_4_0_Windows_x64_SIMD/Egaroucid_for_Console_6_4_0_x64_SIMD.exe -gtp -quiet -nobook'],
    #['6.3.X', 'versions/Egaroucid_for_Console_6_3_X/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
        #['6.3.0', 'versions/Egaroucid_for_Console_6_3_0_Windows_x64_SIMD/Egaroucid_for_Console_6_3_0_x64_SIMD.exe -gtp -quiet -nobook'],
        #['6.2.X', 'versions/Egaroucid_for_Console_6_2_X/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
        #['6.2.0', 'versions/Egaroucid_for_Console_6_2_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -gtp -quiet -nobook'], # same as 6.3.0
    #['6.1.X', 'versions/Egaroucid_for_Console_6_1_X/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
        #['6.1.0', 'versions/Egaroucid_for_Console_6_1_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    #['6.0.X', 'versions/Egaroucid_for_Console_6_0_X/Egaroucid_for_Console_test.exe q'],
    #['Edax4.4', 'versions/edax_4_4/edax-4.4 -gtp -q'],
    # ['Edax4.6', 'versions/edax_4_6/wEdax-x86-64-v3.exe -gtp -q'],
]

NAME_IDX = 0
SUBPROCESS_IDX = 1
RESULT_IDX = 2
RESULT_DISC_IDX = 3
N_PLAYED_IDX = 4
RATING_IDX = 5
CMD_IDX = 6


def start_engine(cmd):
    return subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

players = []
for name, cmd in player_info:
    cmd_with_options = cmd + ' -l ' + str(LEVEL)
    if name == '6.0.X':
        cmd_with_options = cmd + ' ' + str(LEVEL)
    if 'Edax' in name:
        cmd_with_options += ' -n ' + str(N_THREADS)
    elif 'Neural' in name:
        cmd_with_options += ' --threads ' + str(N_THREADS)
    else:
        cmd_with_options += ' -t ' + str(N_THREADS)
    print(name, cmd_with_options)
    players.append([
        name,
        [
            start_engine(cmd_with_options),
            start_engine(cmd_with_options)
        ],
        # W D L (vs other players)
        [[0, 0, 0] for _ in range(len(player_info))],
        # sum of disc differences (vs other players)
        [0 for _ in range(len(player_info))],
        # n_played
        [0 for _ in range(len(player_info))],
        # rating
        Elo_player(1500),
        # command used to start subprocesses
        cmd_with_options
    ])


def restart_process(player_idx, side_idx):
    cmd = players[player_idx][CMD_IDX]
    proc = players[player_idx][SUBPROCESS_IDX][side_idx]
    try:
        proc.kill()
    except Exception:
        pass
    new_proc = start_engine(cmd)
    players[player_idx][SUBPROCESS_IDX][side_idx] = new_proc
    print('restart', players[player_idx][NAME_IDX], 'side', side_idx)
    return new_proc


def send_command(player_idx, side_idx, cmd):
    proc = players[player_idx][SUBPROCESS_IDX][side_idx]
    for _ in range(2):
        if proc.poll() is not None:
            proc = restart_process(player_idx, side_idx)
        try:
            proc.stdin.write(cmd.encode('utf-8'))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc = restart_process(player_idx, side_idx)
            continue

        line = ''
        while line == '':
            raw = proc.stdout.readline()
            if raw == b'':
                proc = restart_process(player_idx, side_idx)
                break
            line = raw.decode(errors='replace').replace('\r', '').replace('\n', '')
        if line != '':
            return line

    raise RuntimeError('failed to communicate with engine: ' + players[player_idx][NAME_IDX] + ' cmd=' + cmd.strip())

def play_battle(p0_idx, p1_idx, opening_idx):
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    shuffled_range2 = [0, 1]
    random.shuffle(shuffled_range2)
    sum_disc_diff_p0 = 0
    for player in shuffled_range2: # which plays black. p0 plays `player`, p1 plays `1 - player`
        record = ''
        o = othello()
        cmd_clear_board = 'clear_board\n'
        for player_idx in [p0_idx, p1_idx]:
            send_command(player_idx, player, cmd_clear_board)
        # play opening
        for i in range(0, len(opening), 2):
            if not o.check_legal():
                cmd_pass = 'play ' + ('b' if o.player == black else 'w') + ' pass\n'
                for player_idx in [p0_idx, p1_idx]:
                    send_command(player_idx, player, cmd_pass)
                o.player = 1 - o.player
                o.check_legal()
            cmd_play = 'play ' + ('b' if o.player == black else 'w') + ' ' + opening[i] + opening[i + 1] + '\n'
            for player_idx in [p0_idx, p1_idx]:
                send_command(player_idx, player, cmd_play)
            x = ord(opening[i].lower()) - ord('a')
            y = int(opening[i + 1]) - 1
            record += opening[i] + opening[i + 1]
            o.move(y, x)
        # play with ai
        while True:
            if not o.check_legal():
                cmd_pass = 'play ' + ('b' if o.player == black else 'w') + ' pass\n'
                for player_idx in [p0_idx, p1_idx]:
                    send_command(player_idx, player, cmd_pass)
                o.player = 1 - o.player
                if not o.check_legal():
                    break
            player_idx = player_idxes[o.player ^ player]
            cmd_genmove = 'genmove ' + ('b' if o.player == black else 'w') + '\n'
            line = send_command(player_idx, player, cmd_genmove)
            coord = line[-2:].lower()
            try:
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
            except:
                print('error')
                print(record)
                print(o.player, player)
                print(coord)
                for i in range(len(players)):
                    for j in range(2):
                        players[i][SUBPROCESS_IDX][j].stdin.write('quit\n'.encode('utf-8'))
                        players[i][SUBPROCESS_IDX][j].stdin.flush()
                exit()
            record += chr(ord('a') + x) + str(y + 1)
            o_player = o.player
            if not o.move(y, x):
                o.print_info()
                print('error')
                print(record)
                print(o.player, player)
                print(coord)
                print(y, x)
            n_player_idx = player_idxes[o_player ^ 1 ^ player]
            cmd_play = 'play ' + ('b' if o_player == black else 'w') + ' ' + chr(ord('a') + x) + str(y + 1) + '\n'
            send_command(n_player_idx, player, cmd_play)
        # update win/draw/loss
        if o.n_stones[player] > o.n_stones[1 - player]: # p0 win
            sum_disc_diff_p0 += o.n_stones[player] - o.n_stones[1 - player] + (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        elif o.n_stones[player] < o.n_stones[1 - player]: # p0 lose
            sum_disc_diff_p0 += o.n_stones[player] - o.n_stones[1 - player] - (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        else:
            sum_disc_diff_p0 += 0
    # update win/draw/loss result and rating
    p0_rating = players[p0_idx][RATING_IDX]
    p1_rating = players[p1_idx][RATING_IDX]
    if sum_disc_diff_p0 > 0: # p0 win
        players[p0_idx][RESULT_IDX][p1_idx][0] += 1 # win
        players[p1_idx][RESULT_IDX][p0_idx][2] += 1 # loss
        n_p0_rating, n_p1_rating = update_rating(p0_rating, p1_rating)
    elif sum_disc_diff_p0 < 0: # p0 lose
        players[p1_idx][RESULT_IDX][p0_idx][0] += 1 # win
        players[p0_idx][RESULT_IDX][p1_idx][2] += 1 # loss
        n_p1_rating, n_p0_rating = update_rating(p1_rating, p0_rating)
    else:
        players[p0_idx][RESULT_IDX][p1_idx][1] += 1 # draw
        players[p1_idx][RESULT_IDX][p0_idx][1] += 1 # draw
        n_p0_rating, n_p1_rating = update_rating_draw(p0_rating, p1_rating)
    # update rating
    players[p0_idx][RATING_IDX] = n_p0_rating
    players[p1_idx][RATING_IDX] = n_p1_rating
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
    estimated_ratings = get_estimated_elo_from_history()

    print('Win Rate')
    print('vs >', end='\t')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        print(name, end='\t')
    print('all', end='\t')
    print('e_rate95')
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
        print("{:.4f}".format(r), end='\t')
        # estimated rating
        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            est, ci = estimated_rating
            print("{:.1f}+-{:.1f}".format(est, ci))

    print('Average Disc Difference')
    print('vs >', end='\t')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        print(name, end='\t')
    print('all', end='\t')
    print('e_rate95')
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
        print(s, end='\t')
        # estimated rating
        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            est, ci = estimated_rating
            print("{:.1f}+-{:.1f}".format(est, ci))


def get_estimated_elo_from_history():
    names = [players[i][NAME_IDX] for i in range(len(players))]
    n_players = len(players)
    win_rates = np.full((n_players, n_players), np.nan, dtype=float)
    games = np.zeros((n_players, n_players), dtype=float)

    for i in range(n_players):
        for j in range(n_players):
            if i == j:
                continue
            w, d, l = players[i][RESULT_IDX][j]
            n = players[i][N_PLAYED_IDX][j]
            games[i, j] = float(n)
            if n > 0:
                win_rates[i, j] = (w + 0.5 * d) / n

    try:
        ratings, intervals = fit_elo_from_winrates_with_interval(win_rates, games=games, names=names, confidence=0.95)
    except ValueError:
        return {}

    return {name: (float(ratings[name]), float(intervals[name])) for name in names}


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
    random.shuffle(matches)
    for p0, p1 in matches:
        play_battle(p0, p1, problem_idx)
        problem_idx += 1
        problem_idx %= len(openings)
    print(i, 'level', LEVEL, 'threads', N_THREADS)
    #print_result()
    print_all_result()
    #output_plt()

print(N_SET_GAMES, 'matches played for each win rate at level', LEVEL, N_THREADS, 'threads')
print_all_result()


for i in range(len(players)):
    for j in range(2):
        players[i][SUBPROCESS_IDX][j].stdin.write('quit\n'.encode('utf-8'))
        players[i][SUBPROCESS_IDX][j].stdin.flush()

for i in range(len(players)):
    for j in range(2):
        players[i][SUBPROCESS_IDX][j].kill()