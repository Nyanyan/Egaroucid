import os
import random
import re
import subprocess
import sys
from othello_py import *


def default_cmds(script_dir):
    baseline = os.path.join(script_dir, 'Egaroucid_for_Console.out.exe') + ' -noboard -nobook'
    candidate = baseline
    return baseline, candidate


def board_command(o):
    grid_str = 'setboard '
    for yy in range(hw):
        for xx in range(hw):
            if o.grid[yy][xx] == black:
                grid_str += 'X'
            elif o.grid[yy][xx] == white:
                grid_str += 'O'
            else:
                grid_str += '-'
    grid_str += ' X\n' if o.player == black else ' O\n'
    return grid_str


def parse_engine_move(line):
    stripped = line.strip()
    if stripped == 'ps' or re.fullmatch(r'[a-h][1-8]', stripped.lower()):
        return stripped.lower()
    if stripped.startswith('|'):
        cols = [elem.strip() for elem in stripped.split('|')]
        if len(cols) >= 5 and re.fullmatch(r'[a-h][1-8]|ps', cols[3].lower()):
            return cols[3].lower()
    return None


def send_and_read(proc, cmd):
    proc.stdin.write(cmd.encode('utf-8'))
    proc.stdin.flush()
    while True:
        raw = proc.stdout.readline()
        if raw == b'':
            raise RuntimeError('engine terminated while waiting for response')
        line = raw.decode(errors='replace').replace('\r', '').replace('\n', '')
        move = parse_engine_move(line)
        if move is not None:
            return move


def start_engine(cmd, move_time_msec, n_threads):
    full_cmd = cmd + ' -movetime ' + str(move_time_msec) + ' -t ' + str(n_threads)
    return subprocess.Popen(full_cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def apply_opening(o, opening):
    record = ''
    for i in range(0, len(opening), 2):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        x = ord(opening[i].lower()) - ord('a')
        y = int(opening[i + 1]) - 1
        record += opening[i] + opening[i + 1]
        o.move(y, x)
    return record


def play_game(procs, p0_black, opening, progress_interval):
    o = othello()
    record = apply_opening(o, opening)
    black_player = 0 if p0_black else 1
    n_ai_moves = 0
    while True:
        if not o.check_legal():
            o.player = 1 - o.player
            if not o.check_legal():
                break
        player_idx = black_player if o.player == black else 1 - black_player
        proc = procs[player_idx]
        proc.stdin.write(board_command(o).encode('utf-8'))
        proc.stdin.flush()
        coord = send_and_read(proc, 'go\n')
        try:
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('a')
        except ValueError:
            raise RuntimeError('invalid engine response: ' + line)
        record += chr(ord('a') + x) + str(y + 1)
        if not o.move(y, x):
            raise RuntimeError('illegal move: ' + coord + ' response=' + line)
        n_ai_moves += 1
        if progress_interval > 0 and n_ai_moves % progress_interval == 0:
            print('progress p0', 'black' if p0_black else 'white', 'ai_moves', n_ai_moves, 'record_len', len(record), flush=True)

    p0_color = black if p0_black else white
    p0_discs = o.n_stones[p0_color]
    p1_discs = o.n_stones[1 - p0_color]
    empty = 64 - p0_discs - p1_discs
    if p0_discs > p1_discs:
        diff = p0_discs - p1_discs + empty
    elif p0_discs < p1_discs:
        diff = p0_discs - p1_discs - empty
    else:
        diff = 0
    return diff, record


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    move_time_msec = int(sys.argv[1]) if len(sys.argv) >= 2 else 1000
    n_openings = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    n_threads = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
    progress_interval = int(sys.argv[6]) if len(sys.argv) >= 7 else 0
    opening_start = int(sys.argv[7]) if len(sys.argv) >= 8 else 0
    cmd0, cmd1 = default_cmds(script_dir)
    if len(sys.argv) >= 5:
        cmd0 = sys.argv[4]
    if len(sys.argv) >= 6:
        cmd1 = sys.argv[5]

    openings_path = os.path.join(script_dir, 'problem/xot/openingslarge.txt')
    with open(openings_path, 'r') as f:
        openings = [elem for elem in f.read().splitlines() if elem.strip()]
    random.seed(57)
    random.shuffle(openings)

    print('player0', cmd0, flush=True)
    print('player1', cmd1, flush=True)
    print('move_time_msec', move_time_msec, 'openings', n_openings, 'threads', n_threads, 'progress_interval', progress_interval, 'opening_start', opening_start, flush=True)
    procs = [start_engine(cmd0, move_time_msec, n_threads), start_engine(cmd1, move_time_msec, n_threads)]
    try:
        wins = draws = losses = 0
        sum_diff = 0
        n_games = 0
        for i in range(n_openings):
            opening = openings[(opening_start + i) % len(openings)]
            for p0_black in [True, False]:
                diff, record = play_game(procs, p0_black, opening, progress_interval)
                n_games += 1
                sum_diff += diff
                if diff > 0:
                    wins += 1
                elif diff < 0:
                    losses += 1
                else:
                    draws += 1
                print(
                    'game', n_games,
                    'p0', 'black' if p0_black else 'white',
                    'diff', diff,
                    'record', record,
                    flush=True
                )
        score_rate = (wins + 0.5 * draws) / max(1, n_games)
        print('summary games', n_games, 'win_draw_loss', wins, draws, losses, 'score_rate', '{:.4f}'.format(score_rate), 'avg_disc_diff', '{:.2f}'.format(sum_diff / max(1, n_games)))
    finally:
        for proc in procs:
            try:
                proc.stdin.write('quit\n'.encode('utf-8'))
                proc.stdin.flush()
            except Exception:
                pass
            proc.kill()


if __name__ == '__main__':
    main()
