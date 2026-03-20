import subprocess
from tqdm import trange, tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import queue
from othello_py import *
from elo_rating import Elo_player, update_rating, update_rating_draw
from elo_rating_backcal import fit_elo_from_winrates
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

LEVEL = int(sys.argv[1])
N_SET_GAMES = int(sys.argv[2])
N_THREADS = 1
N_PARALLEL_MATCHES = 8  # 同時並列対戦数
N_TOTAL_PROCESSES = 10 #int(sys.argv[3]) if len(sys.argv) >= 4 else 2  # 各プレイヤーの総プロセス数(2の倍数)

if N_TOTAL_PROCESSES < 2 or N_TOTAL_PROCESSES % 2 != 0:
    print('N_TOTAL_PROCESSES must be an even number >= 2')
    exit(1)

random.seed(57)

with open('problem/xot/openingslarge.txt', 'r') as f:
    openings = [elem for elem in f.read().splitlines()]
random.shuffle(openings)

# name, cmd
player_info = [
    ['beta', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook'],
    # ['latest',  'Egaroucid_for_Console.exe -quiet -nobook'],
    # ['clang',  'Egaroucid_for_Console_clang.exe -quiet -nobook'],
    ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -quiet -nobook'],
    ['7.7.0', 'versions/Egaroucid_for_Console_7_7_0_Windows_SIMD/Egaroucid_for_Console_7_7_0_SIMD.exe -quiet -nobook'],
    ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -quiet -nobook'],
    ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -quiet -nobook'],
    ['7.4.0', 'versions/Egaroucid_for_Console_7_4_0_Windows_x64_SIMD/Egaroucid_for_Console_7_4_0_x64_SIMD.exe -quiet -nobook'],
    ['7.3.0', 'versions/Egaroucid_for_Console_7_3_0_Windows_x64_SIMD/Egaroucid_for_Console_7_3_0_x64_SIMD.exe -quiet -nobook'],
    ['7.2.0', 'versions/Egaroucid_for_Console_7_2_0_Windows_x64_SIMD/Egaroucid_for_Console_7_2_0_x64_SIMD.exe -quiet -nobook'],
    ['7.1.0', 'versions/Egaroucid_for_Console_7_1_0_Windows_x64_SIMD/Egaroucid_for_Console_7_1_0_x64_SIMD.exe -quiet -nobook'],
    ['7.0.0', 'versions/Egaroucid_for_Console_7_0_0_Windows_x64_SIMD/Egaroucid_for_Console_7_0_0_x64_SIMD.exe -quiet -nobook'],
    #['6.5.X', 'versions/Egaroucid_for_Console_6_5_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.5.0', 'versions/Egaroucid_for_Console_6_5_0_Windows_x64_SIMD/Egaroucid_for_Console_6_5_0_x64_SIMD.exe -quiet -nobook'],
    #['6.4.X', 'versions/Egaroucid_for_Console_6_4_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.4.0', 'versions/Egaroucid_for_Console_6_4_0_Windows_x64_SIMD/Egaroucid_for_Console_6_4_0_x64_SIMD.exe -quiet -nobook'],
    #['6.3.X', 'versions/Egaroucid_for_Console_6_3_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.3.0', 'versions/Egaroucid_for_Console_6_3_0_Windows_x64_SIMD/Egaroucid_for_Console_6_3_0_x64_SIMD.exe -quiet -nobook'],
        #['6.2.X', 'versions/Egaroucid_for_Console_6_2_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.2.0', 'versions/Egaroucid_for_Console_6_2_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook'], # same as 6.3.0
    #['6.1.X', 'versions/Egaroucid_for_Console_6_1_X/Egaroucid_for_Console.exe -quiet -nobook'],
        #['6.1.0', 'versions/Egaroucid_for_Console_6_1_0_Windows_x64_SIMD/Egaroucid_for_Console.exe -quiet -nobook'],
    #['6.0.X', 'versions/Egaroucid_for_Console_6_0_X/Egaroucid_for_Console_test.exe q'],
    #['Edax4.4', 'versions/edax_4_4/edax-4.4 -q'],
    ['Edax4.6', 'versions/edax_4_6/wEdax-x86-64-v3.exe -q'],
]

NAME_IDX = 0
SUBPROCESS_IDX = 1
RESULT_IDX = 2
RESULT_DISC_IDX = 3
N_PLAYED_IDX = 4
RATING_IDX = 5
PROC_POOL_IDX = 6

players = []
results_lock = threading.Lock()  # 結果更新の同期化

for name, cmd in player_info:
    cmd_with_options = cmd + ' -l ' + str(LEVEL)
    if name == '6.0.X':
        cmd_with_options = cmd + ' ' + str(LEVEL)
    if 'Edax' in name:
        cmd_with_options += ' -n ' + str(N_THREADS)
    else:
        cmd_with_options += ' -t ' + str(N_THREADS)
    print(name, cmd_with_options)
    # 総プロセス数を 2 の倍数で用意し、前半/後半をそれぞれ色別に利用する
    subprocesses = [
        subprocess.Popen(cmd_with_options.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for _ in range(N_TOTAL_PROCESSES)
    ]
    proc_pool = [queue.Queue(), queue.Queue()]
    half = N_TOTAL_PROCESSES // 2
    for proc_idx in range(half):
        proc_pool[0].put(proc_idx)
        proc_pool[1].put(proc_idx + half)

    players.append([
        name,
        subprocesses,
        # W D L (vs other players)
        [[0, 0, 0] for _ in range(len(player_info))],
        # sum of disc differences (vs other players)
        [0 for _ in range(len(player_info))],
        # n_played
        [0 for _ in range(len(player_info))],
        # rating
        Elo_player(1500),
        # color-based process pool
        proc_pool
    ])

def play_single_game(p0_idx, p1_idx, opening_idx, p0_is_black):
    """1ゲーム分をプレイして、p0の得点差を返す"""
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    player = 1 if p0_is_black else 0
    p0_proc_idx = players[p0_idx][PROC_POOL_IDX][player].get()
    p1_proc_idx = players[p1_idx][PROC_POOL_IDX][player].get()
    record = ''
    o = othello()
    try:
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

            proc_idx = p0_proc_idx if player_idx == p0_idx else p1_proc_idx
            proc = players[player_idx][SUBPROCESS_IDX][proc_idx]
            proc.stdin.write(grid_str.encode('utf-8'))
            proc.stdin.flush()
            proc.stdin.write('go\n'.encode('utf-8'))
            proc.stdin.flush()
            line = ''
            while line == '' or line == '>':
                line = proc.stdout.readline().decode().replace('\r', '').replace('\n', '')

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
                    for j in range(N_TOTAL_PROCESSES):
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

        # calculate disc difference
        if o.n_stones[player] > o.n_stones[1 - player]:
            return o.n_stones[player] - o.n_stones[1 - player] + (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        elif o.n_stones[player] < o.n_stones[1 - player]:
            return o.n_stones[player] - o.n_stones[1 - player] - (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        return 0
    finally:
        players[p0_idx][PROC_POOL_IDX][player].put(p0_proc_idx)
        players[p1_idx][PROC_POOL_IDX][player].put(p1_proc_idx)

def play_battle(p0_idx, p1_idx, opening_idx):
    """対戦をプレイ（黒番と白番を並列実行）"""
    # 黒番と白番を並列実行
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_black = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, True)
        future_white = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, False)
        sum_disc_diff_p0 = future_black.result() + future_white.result()
    
    # update win/draw/loss result and rating (ロック付き)
    with results_lock:
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
    print('r_rate', end='\t')
    print('e_rate')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_IDX]
        rating = players[i][RATING_IDX]
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
        # rating
        print("{:.1f}".format(rating.get_rating()), end='\t')
        # estimated rating
        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            print("{:.1f}".format(estimated_rating))

    print('Average Disc Difference')
    print('vs >', end='\t')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        print(name, end='\t')
    print('all', end='\t')
    print('r_rate', end='\t')
    print('e_rate')
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_DISC_IDX]
        n_played = players[i][N_PLAYED_IDX]
        rating = players[i][RATING_IDX]
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
        # rating
        print("{:.1f}".format(rating.get_rating()), end='\t')
        # estimated rating
        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            print("{:.1f}".format(estimated_rating))


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
        ratings = fit_elo_from_winrates(win_rates, games=games, names=names)
    except ValueError as e:
        return {}

    return ratings


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
print('parallel matches:', N_PARALLEL_MATCHES)
print('total processes per player:', N_TOTAL_PROCESSES)


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
    
    # 異なるマッチを並列実行
    with ThreadPoolExecutor(max_workers=N_PARALLEL_MATCHES) as executor:
        futures = []
        for p0, p1 in matches:
            future = executor.submit(play_battle, p0, p1, problem_idx)
            futures.append(future)
            problem_idx += 1
            problem_idx %= len(openings)
        
        # 全ての対戦が完了するまで待機（進捗表示付き）
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Round {i+1}/{N_SET_GAMES}"):
            pass
    
    print(i, 'level', LEVEL, 'threads', N_THREADS)
    #print_result()
    print_all_result()
    #output_plt()

print(N_SET_GAMES, 'matches played for each win rate at level', LEVEL, N_THREADS, 'threads')
print_all_result()


for i in range(len(players)):
    for j in range(N_TOTAL_PROCESSES):
        players[i][SUBPROCESS_IDX][j].stdin.write('quit\n'.encode('utf-8'))
        players[i][SUBPROCESS_IDX][j].stdin.flush()

for i in range(len(players)):
    for j in range(N_TOTAL_PROCESSES):
        players[i][SUBPROCESS_IDX][j].kill()