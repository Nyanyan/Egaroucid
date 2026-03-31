import subprocess
import random
import numpy as np
import sys
import queue
from othello_py import *
from elo_rating import Elo_player, update_rating, update_rating_draw
from elo_rating_backcal import fit_elo_from_winrates_with_interval
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

LEVEL = int(sys.argv[1])
N_SET_GAMES = int(sys.argv[2])
N_THREADS = 1
N_PARALLEL_MATCHES = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
N_TOTAL_PROCESSES = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
STATUS_EVERY = int(sys.argv[5]) if len(sys.argv) >= 6 else 1

if N_TOTAL_PROCESSES < 2 or N_TOTAL_PROCESSES % 2 != 0:
    print('N_TOTAL_PROCESSES must be an even number >= 2')
    exit(1)

if N_PARALLEL_MATCHES < 1:
    print('N_PARALLEL_MATCHES must be >= 1')
    exit(1)

if STATUS_EVERY < 1:
    print('STATUS_EVERY must be >= 1')
    exit(1)

random.seed(57)

with open('problem/xot/openingslarge.txt', 'r') as f:
    openings = [elem for elem in f.read().splitlines()]
random.shuffle(openings)

# name, cmd
player_info = [
    ['0325', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260325_1/eval.egev2'],
    ['0324', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260324_1/eval.egev2'],
    ['0323', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260323_1/eval.egev2'],
    ['0322', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260322_1/eval.egev2'],
    ['0321', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260321_1/eval.egev2'],
    ['0320', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260320_1/eval.egev2'],
    ['0318', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260318_1/eval.egev2'],
    ['0317', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260317_1/eval.egev2'],
    ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -quiet -nobook'],
    ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -quiet -nobook'],
    ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -quiet -nobook'],
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
results_lock = threading.Lock()

for name, cmd in player_info:
    cmd_with_options = cmd + ' -l ' + str(LEVEL)
    if name == '6.0.X':
        cmd_with_options = cmd + ' ' + str(LEVEL)
    if 'Edax' in name:
        cmd_with_options += ' -n ' + str(N_THREADS)
    else:
        cmd_with_options += ' -t ' + str(N_THREADS)
    print(name, cmd_with_options)

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
        [[0, 0, 0] for _ in range(len(player_info))],
        [0 for _ in range(len(player_info))],
        [0 for _ in range(len(player_info))],
        Elo_player(1500),
        proc_pool
    ])


def play_single_game(p0_idx, p1_idx, opening_idx, p0_is_black):
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    player = 1 if p0_is_black else 0
    p0_proc_idx = players[p0_idx][PROC_POOL_IDX][player].get()
    p1_proc_idx = players[p1_idx][PROC_POOL_IDX][player].get()
    record = ''
    o = othello()
    try:
        for i in range(0, len(opening), 2):
            if not o.check_legal():
                o.player = 1 - o.player
                o.check_legal()
            x = ord(opening[i].lower()) - ord('a')
            y = int(opening[i + 1]) - 1
            record += opening[i] + opening[i + 1]
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
            except Exception:
                print('error')
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                raise RuntimeError('invalid move coordinate from engine')

            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                print(y, x)
                raise RuntimeError('illegal move from engine')

        if o.n_stones[player] > o.n_stones[1 - player]:
            return o.n_stones[player] - o.n_stones[1 - player] + (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        if o.n_stones[player] < o.n_stones[1 - player]:
            return o.n_stones[player] - o.n_stones[1 - player] - (64 - (o.n_stones[player] + o.n_stones[1 - player]))
        return 0
    finally:
        players[p0_idx][PROC_POOL_IDX][player].put(p0_proc_idx)
        players[p1_idx][PROC_POOL_IDX][player].put(p1_proc_idx)


def play_battle(p0_idx, p1_idx, opening_idx):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_black = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, True)
        future_white = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, False)
        sum_disc_diff_p0 = future_black.result() + future_white.result()

    with results_lock:
        p0_rating = players[p0_idx][RATING_IDX]
        p1_rating = players[p1_idx][RATING_IDX]
        if sum_disc_diff_p0 > 0:
            players[p0_idx][RESULT_IDX][p1_idx][0] += 1
            players[p1_idx][RESULT_IDX][p0_idx][2] += 1
            n_p0_rating, n_p1_rating = update_rating(p0_rating, p1_rating)
        elif sum_disc_diff_p0 < 0:
            players[p1_idx][RESULT_IDX][p0_idx][0] += 1
            players[p0_idx][RESULT_IDX][p1_idx][2] += 1
            n_p1_rating, n_p0_rating = update_rating(p1_rating, p0_rating)
        else:
            players[p0_idx][RESULT_IDX][p1_idx][1] += 1
            players[p1_idx][RESULT_IDX][p0_idx][1] += 1
            n_p0_rating, n_p1_rating = update_rating_draw(p0_rating, p1_rating)

        players[p0_idx][RATING_IDX] = n_p0_rating
        players[p1_idx][RATING_IDX] = n_p1_rating
        players[p0_idx][RESULT_DISC_IDX][p1_idx] += sum_disc_diff_p0 / 2
        players[p1_idx][RESULT_DISC_IDX][p0_idx] -= sum_disc_diff_p0 / 2
        players[p0_idx][N_PLAYED_IDX][p1_idx] += 1
        players[p1_idx][N_PLAYED_IDX][p0_idx] += 1


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


def print_all_result_locked():
    estimated_ratings = get_estimated_elo_from_history()

    print('Win Rate')
    print('vs >', end='\t')
    for i in range(len(players)):
        print(players[i][NAME_IDX], end='\t')
    print('all\te_rate95')

    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_IDX]
        print(name, end='\t')

        for j in range(len(players)):
            if i == j:
                print('-', end='\t')
            else:
                w, d, l = result[j]
                r = (w + d * 0.5) / max(1, w + d + l)
                print('{:.4f}'.format(r), end='\t')

        w = 0
        d = 0
        l = 0
        for ww, dd, ll in result:
            w += ww
            d += dd
            l += ll
        r = (w + d * 0.5) / max(1, w + d + l)
        print('{:.4f}'.format(r), end='\t')

        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            est, ci = estimated_rating
            print('{:.1f}+-{:.1f}'.format(est, ci))

    print('Average Disc Difference')
    print('vs >', end='\t')
    for i in range(len(players)):
        print(players[i][NAME_IDX], end='\t')
    print('all\te_rate95')

    for i in range(len(players)):
        name = players[i][NAME_IDX]
        result = players[i][RESULT_DISC_IDX]
        n_played = players[i][N_PLAYED_IDX]
        print(name, end='\t')

        for j in range(len(players)):
            if i == j:
                print('-', end='\t')
            else:
                avg_discs = result[j] / max(1, n_played[j])
                s = '{:.2f}'.format(avg_discs)
                if avg_discs >= 0:
                    s = '+' + s
                print(s, end='\t')

        avg_discs_all = sum(result) / max(1, sum(n_played))
        s = '{:.2f}'.format(avg_discs_all)
        if avg_discs_all >= 0:
            s = '+' + s
        print(s, end='\t')

        estimated_rating = estimated_ratings.get(name)
        if estimated_rating is None:
            print('-')
        else:
            est, ci = estimated_rating
            print('{:.1f}+-{:.1f}'.format(est, ci))


def print_games_progress_locked(target_per_pair):
    print('Games Progress (played/target)')
    print('vs >', end='\t')
    for i in range(len(players)):
        print(players[i][NAME_IDX], end='\t')
    print('all')

    target_per_player = target_per_pair * (len(players) - 1)
    for i in range(len(players)):
        name = players[i][NAME_IDX]
        n_played = players[i][N_PLAYED_IDX]
        print(name, end='\t')
        for j in range(len(players)):
            if i == j:
                print('-', end='\t')
            else:
                print(str(n_played[j]) + '/' + str(target_per_pair), end='\t')
        print(str(sum(n_played)) + '/' + str(target_per_player))


def print_status(completed, total, target_per_pair):
    percent = 100.0 * completed / max(1, total)
    print('\n' + '=' * 80)
    print('Progress: {}/{} ({:.2f}%)'.format(completed, total, percent))
    with results_lock:
        print_all_result_locked()
        print_games_progress_locked(target_per_pair)


def shutdown_all_processes():
    for i in range(len(players)):
        for j in range(N_TOTAL_PROCESSES):
            proc = players[i][SUBPROCESS_IDX][j]
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write('quit\n'.encode('utf-8'))
                    proc.stdin.flush()
            except Exception:
                pass

    for i in range(len(players)):
        for j in range(N_TOTAL_PROCESSES):
            proc = players[i][SUBPROCESS_IDX][j]
            try:
                proc.kill()
            except Exception:
                pass


print('n_players', len(players))
print('level', LEVEL)
print('parallel matches:', N_PARALLEL_MATCHES)
print('total processes per player:', N_TOTAL_PROCESSES)
print('status interval (battles):', STATUS_EVERY)

matches = []
for p0 in range(len(players)):
    for p1 in range(p0 + 1, len(players)):
        matches.append((p0, p1))

all_battles = []
problem_idx = 0
for _ in range(N_SET_GAMES):
    round_matches = matches[:]
    random.shuffle(round_matches)
    for p0, p1 in round_matches:
        all_battles.append((p0, p1, problem_idx))
        problem_idx += 1
        problem_idx %= len(openings)

total_battles = len(all_battles)
completed_battles = 0

try:
    with ThreadPoolExecutor(max_workers=N_PARALLEL_MATCHES) as executor:
        iterator = iter(all_battles)
        futures = {}

        for _ in range(min(N_PARALLEL_MATCHES, total_battles)):
            p0, p1, opening_idx = next(iterator)
            future = executor.submit(play_battle, p0, p1, opening_idx)
            futures[future] = (p0, p1)

        while futures:
            for future in as_completed(list(futures.keys())):
                futures.pop(future)
                future.result()
                completed_battles += 1

                if completed_battles % STATUS_EVERY == 0 or completed_battles == total_battles:
                    print_status(completed_battles, total_battles, N_SET_GAMES)

                try:
                    p0, p1, opening_idx = next(iterator)
                except StopIteration:
                    pass
                else:
                    nf = executor.submit(play_battle, p0, p1, opening_idx)
                    futures[nf] = (p0, p1)
                break
finally:
    shutdown_all_processes()

print('\nAll battles finished.')
print_status(completed_battles, total_battles, N_SET_GAMES)
