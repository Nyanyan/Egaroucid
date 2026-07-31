import argparse
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from othello_py import *


PROBLEM_FILE = 'problem/xot/openingslarge.txt'
PROCESS_POOL_GET_TIMEOUT_SEC = 0.2
QUIT_TIMEOUT_SEC = 2.0
KILL_TIMEOUT_SEC = 5.0

NAME_IDX = 0
CMD_IDX = 1
POOL_IDX = 2
RESULT_IDX = 3
DISC_SUM_IDX = 4
PLAYED_IDX = 5

process_registry = set()
process_registry_lock = threading.Lock()
shutdown_event = threading.Event()
results_lock = threading.Lock()
failure_lock = threading.Lock()
first_failure = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run paired XOT matches with a fixed time limit for each move.'
    )
    parser.add_argument('move_time_msec', type=int)
    parser.add_argument('n_set_games', type=int, help='number of paired starting positions')
    parser.add_argument('n_parallel_pairs', type=int, nargs='?', default=15)
    parser.add_argument('n_total_processes', type=int, nargs='?', default=30)
    parser.add_argument('status_every', type=int, nargs='?', default=25)
    parser.add_argument(
        '--player',
        action='append',
        default=[],
        metavar='NAME=CMD',
        help='player definition; repeat twice or more'
    )
    parser.add_argument(
        '--problem-file',
        default=PROBLEM_FILE,
        help='starting position file; relative paths are resolved from this script directory'
    )
    parser.add_argument('--opening-seed', type=int, default=57)
    parser.add_argument('--opening-start', type=int, default=0)
    parser.add_argument('--threads-per-engine', type=int, default=1)
    parser.add_argument('--hash-level', type=int, default=None)
    parser.add_argument(
        '--save-kifu',
        nargs='?',
        const=True,
        default=None,
        metavar='PATH',
        help='save every game record as TSV'
    )
    return parser.parse_args()


def parse_player_spec(spec):
    name, sep, cmd = spec.partition('=')
    name = name.strip()
    cmd = cmd.strip()
    if not sep or not name or not cmd:
        raise ValueError('invalid --player spec: ' + spec)
    return [name, cmd, None, None, 0.0, 0]


def default_players():
    return [
        ['player0', 'Egaroucid_for_Console.exe -quiet -nobook', None, None, 0.0, 0],
        ['player1', 'Egaroucid_for_Console.exe -quiet -nobook', None, None, 0.0, 0],
    ]


def build_player_command(cmd, move_time_msec, threads_per_engine, hash_level):
    result = cmd + ' -movetime ' + str(move_time_msec) + ' -t ' + str(threads_per_engine)
    if hash_level is not None:
        result += ' -hash ' + str(hash_level)
    return result


def resolve_script_path(script_dir, path):
    p = Path(path)
    if p.is_absolute():
        return p
    return script_dir / p


def init_result_tables(players):
    n_players = len(players)
    for player in players:
        player[POOL_IDX] = queue.Queue()
        player[RESULT_IDX] = [[0, 0, 0] for _ in range(n_players)]
        player[DISC_SUM_IDX] = [0.0 for _ in range(n_players)]
        player[PLAYED_IDX] = [0 for _ in range(n_players)]


def register_process(proc):
    with process_registry_lock:
        process_registry.add(proc)


def unregister_process(proc):
    with process_registry_lock:
        process_registry.discard(proc)


def start_engine(cmd, cwd):
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=False,
    )
    register_process(proc)
    return proc


def start_player_processes(players, n_total_processes, cwd):
    for player in players:
        name = player[NAME_IDX]
        cmd = player[CMD_IDX]
        print(name, cmd, flush=True)
        for _ in range(n_total_processes):
            player[POOL_IDX].put(start_engine(cmd, cwd))


def shutdown_process(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write(b'quit\n')
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=QUIT_TIMEOUT_SEC)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=KILL_TIMEOUT_SEC)
        except Exception:
            pass
    unregister_process(proc)


def shutdown_all_processes():
    shutdown_event.set()
    with process_registry_lock:
        procs = list(process_registry)
    for proc in procs:
        shutdown_process(proc)


def acquire_process(player_idx, players):
    proc_pool = players[player_idx][POOL_IDX]
    while True:
        if shutdown_event.is_set():
            raise RuntimeError('shutdown in progress')
        try:
            return proc_pool.get(timeout=PROCESS_POOL_GET_TIMEOUT_SEC)
        except queue.Empty:
            pass


def release_process(player_idx, players, proc):
    if proc is not None and proc.poll() is None and not shutdown_event.is_set():
        players[player_idx][POOL_IDX].put(proc)
    else:
        shutdown_process(proc)


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


def send_command(proc, cmd):
    proc.stdin.write(cmd.encode('utf-8'))
    proc.stdin.flush()


def read_move(proc):
    while True:
        raw = proc.stdout.readline()
        if raw == b'':
            raise RuntimeError('engine terminated while waiting for response')
        line = raw.decode(errors='replace').replace('\r', '').replace('\n', '')
        move = parse_engine_move(line)
        if move is not None:
            return move, line


def apply_opening(o, opening):
    record = ''
    for i in range(0, len(opening), 2):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        x = ord(opening[i].lower()) - ord('a')
        y = int(opening[i + 1]) - 1
        coord = opening[i].lower() + opening[i + 1]
        record += coord
        if not o.move(y, x):
            raise RuntimeError('illegal opening move: ' + coord)
    return record


def p0_disc_diff(o, p0_black):
    p0_color = black if p0_black else white
    p0_discs = o.n_stones[p0_color]
    p1_discs = o.n_stones[1 - p0_color]
    empty = 64 - p0_discs - p1_discs
    diff = p0_discs - p1_discs
    if diff > 0:
        return diff + empty
    if diff < 0:
        return diff - empty
    return 0


def play_single_game(players, p0_idx, p1_idx, opening_idx, opening, p0_black):
    p0_proc = None
    p1_proc = None
    p0_proc = acquire_process(p0_idx, players)
    p1_proc = acquire_process(p1_idx, players)
    try:
        send_command(p0_proc, 'clearcache\n')
        send_command(p1_proc, 'clearcache\n')

        o = othello()
        record = apply_opening(o, opening)
        black_player = p0_idx if p0_black else p1_idx

        while True:
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    break

            player_idx = black_player if o.player == black else (p1_idx if black_player == p0_idx else p0_idx)
            proc = p0_proc if player_idx == p0_idx else p1_proc
            send_command(proc, board_command(o))
            send_command(proc, 'go\n')
            coord, line = read_move(proc)
            if coord == 'ps':
                raise RuntimeError('engine returned pass in a legal position: ' + players[player_idx][NAME_IDX])

            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('a')
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                raise RuntimeError('illegal move: ' + coord + ' response=' + line)

        diff = p0_disc_diff(o, p0_black)
        return {
            'game': 'p0_black' if p0_black else 'p0_white',
            'opening_idx': opening_idx,
            'opening': opening,
            'p0_idx': p0_idx,
            'p1_idx': p1_idx,
            'black_idx': p0_idx if p0_black else p1_idx,
            'white_idx': p1_idx if p0_black else p0_idx,
            'p0_disc_diff': diff,
            'black_stones': o.n_stones[black],
            'white_stones': o.n_stones[white],
            'record': record,
        }
    finally:
        release_process(p0_idx, players, p0_proc)
        release_process(p1_idx, players, p1_proc)


def play_pair(players, p0_idx, p1_idx, opening_idx, opening):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_black = executor.submit(play_single_game, players, p0_idx, p1_idx, opening_idx, opening, True)
        future_white = executor.submit(play_single_game, players, p0_idx, p1_idx, opening_idx, opening, False)
        results = [future_black.result(), future_white.result()]

    sum_diff = sum(result['p0_disc_diff'] for result in results)
    with results_lock:
        if sum_diff > 0:
            players[p0_idx][RESULT_IDX][p1_idx][0] += 1
            players[p1_idx][RESULT_IDX][p0_idx][2] += 1
        elif sum_diff < 0:
            players[p0_idx][RESULT_IDX][p1_idx][2] += 1
            players[p1_idx][RESULT_IDX][p0_idx][0] += 1
        else:
            players[p0_idx][RESULT_IDX][p1_idx][1] += 1
            players[p1_idx][RESULT_IDX][p0_idx][1] += 1

        players[p0_idx][DISC_SUM_IDX][p1_idx] += sum_diff / 2.0
        players[p1_idx][DISC_SUM_IDX][p0_idx] -= sum_diff / 2.0
        players[p0_idx][PLAYED_IDX][p1_idx] += 1
        players[p1_idx][PLAYED_IDX][p0_idx] += 1

    return results


def init_kifu_file(path):
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('pair\tgame\topening_idx\topening\tp0\tp1\tblack\twhite\tp0_disc_diff\tblack_stones\twhite_stones\trecord\n')


def save_kifu(path, pair_no, players, game_results):
    if path is None:
        return
    with open(path, 'a', encoding='utf-8', newline='') as f:
        for result in game_results:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                pair_no,
                result['game'],
                result['opening_idx'],
                result['opening'],
                players[result['p0_idx']][NAME_IDX],
                players[result['p1_idx']][NAME_IDX],
                players[result['black_idx']][NAME_IDX],
                players[result['white_idx']][NAME_IDX],
                result['p0_disc_diff'],
                result['black_stones'],
                result['white_stones'],
                result['record'],
            ))


def format_wdl(w, d, l):
    n = w + d + l
    score_rate = (w + 0.5 * d) / max(1, n)
    return '{}-{}-{} score_rate {:.4f}'.format(w, d, l, score_rate)


def print_status(players, completed, total, started_at):
    elapsed = time.time() - started_at
    speed = 60.0 * completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) * elapsed / completed if completed > 0 else 0.0
    print('\n' + '=' * 80, flush=True)
    print('Progress: {}/{} pairs ({:.2f}%)'.format(completed, total, 100.0 * completed / max(1, total)), flush=True)
    print('Elapsed: {:.1f}s  ETA: {:.1f}s  Speed: {:.2f} pairs/min'.format(elapsed, eta, speed), flush=True)

    print('Win Rate', flush=True)
    print('vs >\t' + '\t'.join(player[NAME_IDX] for player in players) + '\tall', flush=True)
    for i, player in enumerate(players):
        row = [player[NAME_IDX]]
        total_w = total_d = total_l = 0
        for j in range(len(players)):
            if i == j:
                row.append('-')
            else:
                w, d, l = player[RESULT_IDX][j]
                total_w += w
                total_d += d
                total_l += l
                n = w + d + l
                row.append('{:.4f}'.format((w + 0.5 * d) / max(1, n)))
        row.append('{:.4f}'.format((total_w + 0.5 * total_d) / max(1, total_w + total_d + total_l)))
        print('\t'.join(row), flush=True)

    print('Pair WDL and average disc difference per game', flush=True)
    for i, player in enumerate(players):
        for j, opponent in enumerate(players):
            if i == j:
                continue
            w, d, l = player[RESULT_IDX][j]
            n = player[PLAYED_IDX][j]
            avg_diff = player[DISC_SUM_IDX][j] / max(1, n)
            print(
                '{} vs {}: {} avg_disc_diff {:+.2f}'.format(
                    player[NAME_IDX],
                    opponent[NAME_IDX],
                    format_wdl(w, d, l),
                    avg_diff,
                ),
                flush=True,
            )


def record_failure(exc):
    global first_failure
    with failure_lock:
        if first_failure is None:
            first_failure = exc


def main():
    args = parse_args()
    if args.move_time_msec <= 0:
        raise ValueError('move_time_msec must be positive')
    if args.n_set_games <= 0:
        raise ValueError('n_set_games must be positive')
    if args.n_parallel_pairs <= 0:
        raise ValueError('n_parallel_pairs must be positive')
    if args.n_total_processes < args.n_parallel_pairs * 2:
        raise ValueError('n_total_processes must be at least 2 * n_parallel_pairs')
    if args.status_every <= 0:
        raise ValueError('status_every must be positive')

    script_dir = Path(__file__).resolve().parent
    players = [parse_player_spec(spec) for spec in args.player] if args.player else default_players()
    if len(players) < 2:
        raise ValueError('at least two players are required')
    for player in players:
        player[CMD_IDX] = build_player_command(
            player[CMD_IDX],
            args.move_time_msec,
            args.threads_per_engine,
            args.hash_level,
        )
    init_result_tables(players)

    problem_file = resolve_script_path(script_dir, args.problem_file)
    with open(problem_file, 'r', encoding='utf-8') as f:
        openings = [line.strip() for line in f.read().splitlines() if line.strip()]
    random.seed(args.opening_seed)
    random.shuffle(openings)

    kifu_path = args.save_kifu
    if kifu_path is True:
        kifu_path = script_dir / 'transcript' / ('battle_parallel_fixed_time_kifu_{}.tsv'.format(time.strftime('%Y%m%d_%H%M%S')))
    elif kifu_path is not None:
        kifu_path = resolve_script_path(script_dir, kifu_path)
    if kifu_path is not None:
        kifu_path = str(kifu_path)
        init_kifu_file(kifu_path)

    print('fixed move time msec:', args.move_time_msec, flush=True)
    print('paired starting positions:', args.n_set_games, flush=True)
    print('parallel pairs:', args.n_parallel_pairs, flush=True)
    print('parallel games:', args.n_parallel_pairs * 2, flush=True)
    print('total processes per player:', args.n_total_processes, flush=True)
    print('threads per engine:', args.threads_per_engine, flush=True)
    print('hash level:', '-' if args.hash_level is None else args.hash_level, flush=True)
    print('problem file:', str(problem_file), flush=True)
    print('opening seed:', args.opening_seed, flush=True)
    print('opening start:', args.opening_start, flush=True)
    if kifu_path is not None:
        print('save kifu:', kifu_path, flush=True)

    start_player_processes(players, args.n_total_processes, script_dir)

    matches = []
    n_players = len(players)
    opening_pos = args.opening_start
    for _ in range(args.n_set_games):
        for p0_idx in range(n_players):
            for p1_idx in range(p0_idx + 1, n_players):
                opening_idx = opening_pos % len(openings)
                matches.append((p0_idx, p1_idx, opening_idx, openings[opening_idx]))
                opening_pos += 1

    started_at = time.time()
    completed = 0
    total = len(matches)
    try:
        with ThreadPoolExecutor(max_workers=args.n_parallel_pairs) as executor:
            iterator = iter(matches)
            futures = {}
            for _ in range(min(args.n_parallel_pairs, total)):
                match = next(iterator)
                futures[executor.submit(play_pair, players, *match)] = match

            while futures:
                for future in as_completed(list(futures.keys())):
                    match = futures.pop(future)
                    try:
                        game_results = future.result()
                    except Exception as exc:
                        record_failure(exc)
                        shutdown_all_processes()
                        raise
                    completed += 1
                    save_kifu(kifu_path, completed, players, game_results)
                    if completed % args.status_every == 0 or completed == total:
                        print_status(players, completed, total, started_at)
                    try:
                        next_match = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        futures[executor.submit(play_pair, players, *next_match)] = next_match
                    break
    finally:
        shutdown_all_processes()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record_failure(exc)
        shutdown_all_processes()
        print('ERROR:', exc, file=sys.stderr, flush=True)
        raise
