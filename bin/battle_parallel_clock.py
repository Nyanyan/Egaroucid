"""Paired XOT matches with a game clock (each side has a fixed total time per game).

The runner keeps both clocks. Before every move it sends the position, then
"settimems <color> <remaining milliseconds>" and "go", so the engine allocates its own time
(ai_time_limit). The time from sending the position to receiving a legal move is subtracted from the
mover's clock. A side whose clock runs out keeps playing with 0 ms left; the game is recorded with a
time-loss flag, and the disc result is kept as played. The clock used is recorded without clamping, so it
can exceed the game time.

--ggs-time sends the time the GGS client would give the search instead of the clock (GGS reports whole
seconds and src/console/ggs.hpp subtracts ggs_time_safety_margin_msec). --board-openings reads the problem
file as start positions ("<64 squares X/O/-> <X|O>" per line, e.g. GGS s8r14 starts) instead of move lists.

Example:
  python battle_parallel_clock.py 60000 200 3 --threads-per-engine 4 --hash-level 23 \
      --player "base=Egaroucid_nnue_base.exe -quiet -nobook" --player "nnue=Egaroucid_nnue.exe -quiet -nobook" \
      --save-kifu ../reports/kifu_clock.tsv
"""
import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import battle_parallel_fixed_time as base
from othello_py import *

NAME_IDX = base.NAME_IDX
CMD_IDX = base.CMD_IDX
RESULT_IDX = base.RESULT_IDX
DISC_SUM_IDX = base.DISC_SUM_IDX
PLAYED_IDX = base.PLAYED_IDX

time_loss_count = {}
time_used = {}
stats_lock = base.results_lock


def parse_args():
    parser = argparse.ArgumentParser(description='Run paired XOT matches with a game clock.')
    parser.add_argument('game_time_msec', type=int, help='total time of each side per game')
    parser.add_argument('n_set_games', type=int, help='number of paired starting positions')
    parser.add_argument('n_parallel_pairs', type=int, nargs='?', default=3)
    parser.add_argument('status_every', type=int, nargs='?', default=25)
    parser.add_argument('--player', action='append', default=[], metavar='NAME=CMD')
    parser.add_argument('--problem-file', default=base.PROBLEM_FILE)
    parser.add_argument('--opening-seed', type=int, default=57)
    parser.add_argument('--opening-start', type=int, default=0)
    parser.add_argument('--threads-per-engine', type=int, default=1)
    parser.add_argument('--hash-level', type=int, default=None)
    parser.add_argument('--save-kifu', default=None, metavar='PATH')
    parser.add_argument('--engine-log-dir', default=None, metavar='DIR',
                        help='save each engine process stderr (and its command) here, to diagnose an engine that exits')
    parser.add_argument('--ggs-time', action='store_true',
                        help='send the search time of the GGS client (whole seconds minus its safety margin)')
    parser.add_argument('--board-openings', action='store_true',
                        help='the problem file holds start positions ("<64 squares> <side>") instead of move lists')
    return parser.parse_args()


def ggs_time_safety_margin_msec(remaining_msec):
    """same as ggs_time_safety_margin_msec in src/console/ggs.hpp"""
    if remaining_msec > 120000:
        return 3000
    if remaining_msec > 60000:
        return 2500
    if remaining_msec > 30000:
        return 2000
    if remaining_msec > 10000:
        return 1500
    return 1000


def ggs_search_time_msec(clock_msec):
    """the time the GGS client gives the search: GGS shows whole seconds, the client subtracts its safety margin"""
    raw = int(clock_msec) // 1000 * 1000
    margin = ggs_time_safety_margin_msec(raw)
    if raw > margin:
        return raw - margin
    return max(int(raw * 0.1), 1)


def install_engine_logging(log_dir):
    """make the shared runner keep every engine's stderr in log_dir/engine_<n>.err"""
    import itertools
    import subprocess
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    counter = itertools.count()

    def start_engine(cmd, cwd):
        k = next(counter)
        (log_dir / 'engine_{}.cmd'.format(k)).write_text(cmd, encoding='utf-8')
        err = open(log_dir / 'engine_{}.err'.format(k), 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, shell=False)
        base.register_process(proc)
        return proc
    base.start_engine = start_engine


def build_player_command(cmd, threads_per_engine, hash_level):
    result = cmd + ' -t ' + str(threads_per_engine)
    if hash_level is not None:
        result += ' -hash ' + str(hash_level)
    return result


def wait_ready(proc):
    """read the engine output up to the end of the version message (the @website line)"""
    while True:
        raw = proc.stdout.readline()
        if raw == b'':
            raise RuntimeError('engine terminated while waiting for the version message')
        if raw.strip().startswith(b'@website'):
            return


def play_single_game(players, p0_idx, p1_idx, opening_idx, opening, p0_black, game_time_msec, ggs_time, board_openings):
    p0_proc = base.acquire_process(p0_idx, players)
    p1_proc = base.acquire_process(p1_idx, players)
    try:
        # clear the caches and wait until both engines answer, so start-up and clearing are not charged
        for proc in (p0_proc, p1_proc):
            base.send_command(proc, 'clearcache\nversion\n')
        for proc in (p0_proc, p1_proc):
            wait_ready(proc)
        o = othello()
        if board_openings:
            if not o.set_board_from_str(opening):
                raise RuntimeError('bad start position: ' + opening)
            record = ''
        else:
            record = base.apply_opening(o, opening)
        black_player = p0_idx if p0_black else p1_idx
        white_player = p1_idx if p0_black else p0_idx
        clock = {black: float(game_time_msec), white: float(game_time_msec)}
        flagged = {black: False, white: False}
        move_times = []
        while True:
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    break
            color = o.player
            player_idx = black_player if color == black else white_player
            proc = p0_proc if player_idx == p0_idx else p1_proc
            remaining = max(0, int(clock[color]))
            if ggs_time:
                remaining = ggs_search_time_msec(remaining)
            t0 = time.perf_counter()
            base.send_command(proc, base.board_command(o) + 'settimems ' + ('X' if color == black else 'O') + ' ' + str(remaining) + '\ngo\n')
            coord, line = base.read_move(proc)
            elapsed = 1000.0 * (time.perf_counter() - t0)
            clock[color] -= elapsed
            if clock[color] < 0:
                flagged[color] = True
            move_times.append(int(round(elapsed)))
            if coord == 'ps':
                raise RuntimeError('engine returned pass in a legal position: ' + players[player_idx][NAME_IDX])
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('a')
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                raise RuntimeError('illegal move: ' + coord + ' response=' + line)
        diff = base.p0_disc_diff(o, p0_black)
        p0_color = black if p0_black else white
        return {
            'game': 'p0_black' if p0_black else 'p0_white',
            'opening_idx': opening_idx, 'opening': opening,
            'p0_idx': p0_idx, 'p1_idx': p1_idx, 'black_idx': black_player, 'white_idx': white_player,
            'p0_disc_diff': diff, 'black_stones': o.n_stones[black], 'white_stones': o.n_stones[white],
            'record': record,
            'p0_time_loss': flagged[p0_color], 'p1_time_loss': flagged[1 - p0_color],
            'black_used': int(round(game_time_msec - clock[black])), 'white_used': int(round(game_time_msec - clock[white])),
            'move_times': move_times,
        }
    finally:
        base.release_process(p0_idx, players, p0_proc)
        base.release_process(p1_idx, players, p1_proc)


def play_pair(players, p0_idx, p1_idx, opening_idx, opening, game_time_msec, ggs_time=False, board_openings=False):
    with ThreadPoolExecutor(max_workers=2) as executor:
        fb = executor.submit(play_single_game, players, p0_idx, p1_idx, opening_idx, opening, True, game_time_msec, ggs_time, board_openings)
        fw = executor.submit(play_single_game, players, p0_idx, p1_idx, opening_idx, opening, False, game_time_msec, ggs_time, board_openings)
        results = [fb.result(), fw.result()]
    sum_diff = sum(r['p0_disc_diff'] for r in results)
    with stats_lock:
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
        for r in results:
            for idx, flag in ((r['p0_idx'], r['p0_time_loss']), (r['p1_idx'], r['p1_time_loss'])):
                time_loss_count[idx] = time_loss_count.get(idx, 0) + int(flag)
            for idx, used in ((r['black_idx'], r['black_used']), (r['white_idx'], r['white_used'])):
                time_used.setdefault(idx, []).append(used)
    return results


def init_kifu_file(path):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('pair\tgame\topening_idx\topening\tp0\tp1\tblack\twhite\tp0_disc_diff\tblack_stones\twhite_stones\trecord'
                '\tp0_time_loss\tp1_time_loss\tblack_used_ms\twhite_used_ms\tmove_times_ms\n')


def save_kifu(path, pair_no, players, game_results):
    if path is None:
        return
    with open(path, 'a', encoding='utf-8', newline='') as f:
        for r in game_results:
            f.write('\t'.join(str(x) for x in (
                pair_no, r['game'], r['opening_idx'], r['opening'], players[r['p0_idx']][NAME_IDX], players[r['p1_idx']][NAME_IDX],
                players[r['black_idx']][NAME_IDX], players[r['white_idx']][NAME_IDX], r['p0_disc_diff'], r['black_stones'],
                r['white_stones'], r['record'], int(r['p0_time_loss']), int(r['p1_time_loss']), r['black_used'], r['white_used'],
                ','.join(str(t) for t in r['move_times']))) + '\n')


def print_status(players, completed, total, started_at):
    base.print_status(players, completed, total, started_at)
    with stats_lock:
        for i, player in enumerate(players):
            used = time_used.get(i, [])
            avg = sum(used) / max(1, len(used))
            print('{}: time losses {} / {} games, average clock used {:.0f} ms'.format(player[NAME_IDX], time_loss_count.get(i, 0), len(used), avg), flush=True)


def main():
    args = parse_args()
    if args.game_time_msec <= 0 or args.n_set_games <= 0 or args.n_parallel_pairs <= 0:
        raise ValueError('game_time_msec, n_set_games and n_parallel_pairs must be positive')
    script_dir = Path(__file__).resolve().parent
    players = [base.parse_player_spec(spec) for spec in args.player] if args.player else base.default_players()
    if len(players) < 2:
        raise ValueError('at least two players are required')
    for player in players:
        player[CMD_IDX] = build_player_command(player[CMD_IDX], args.threads_per_engine, args.hash_level)
    base.init_result_tables(players)
    problem_file = base.resolve_script_path(script_dir, args.problem_file)
    with open(problem_file, 'r', encoding='utf-8') as f:
        openings = [line.strip() for line in f.read().splitlines() if line.strip()]
    random.seed(args.opening_seed)
    random.shuffle(openings)
    kifu_path = None if args.save_kifu is None else str(base.resolve_script_path(script_dir, args.save_kifu))
    if args.engine_log_dir:
        install_engine_logging(base.resolve_script_path(script_dir, args.engine_log_dir))
    init_kifu_file(kifu_path)
    print('game time msec per side:', args.game_time_msec, flush=True)
    print('paired starting positions:', args.n_set_games, flush=True)
    print('parallel pairs:', args.n_parallel_pairs, flush=True)
    print('threads per engine:', args.threads_per_engine, flush=True)
    print('hash level:', '-' if args.hash_level is None else args.hash_level, flush=True)
    print('problem file:', str(problem_file), flush=True)
    print('opening seed:', args.opening_seed, 'start:', args.opening_start, flush=True)
    print('GGS search time:', 'yes' if args.ggs_time else 'no', 'board openings:', 'yes' if args.board_openings else 'no', flush=True)
    base.start_player_processes(players, 2 * args.n_parallel_pairs, script_dir)
    matches = []
    opening_pos = args.opening_start
    for _ in range(args.n_set_games):
        for p0_idx in range(len(players)):
            for p1_idx in range(p0_idx + 1, len(players)):
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
                futures[executor.submit(play_pair, players, *match, args.game_time_msec, args.ggs_time, args.board_openings)] = match
            while futures:
                for future in as_completed(list(futures.keys())):
                    futures.pop(future)
                    try:
                        game_results = future.result()
                    except Exception as exc:
                        base.record_failure(exc)
                        base.shutdown_all_processes()
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
                        futures[executor.submit(play_pair, players, *next_match, args.game_time_msec, args.ggs_time, args.board_openings)] = next_match
                    break
    finally:
        base.shutdown_all_processes()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        base.record_failure(exc)
        base.shutdown_all_processes()
        print('ERROR:', exc, file=sys.stderr, flush=True)
        raise
