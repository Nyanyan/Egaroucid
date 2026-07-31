import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from battle_parallel_fixed_time import (
    CMD_IDX,
    NAME_IDX,
    PLAYED_IDX,
    PROBLEM_FILE,
    default_players,
    format_wdl,
    init_kifu_file,
    init_result_tables,
    parse_player_spec,
    play_pair,
    print_status,
    record_failure,
    resolve_script_path,
    save_kifu,
    shutdown_all_processes,
    start_player_processes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run paired XOT matches with a fixed Egaroucid level.'
    )
    parser.add_argument('level', type=int)
    parser.add_argument('n_set_games', type=int, help='number of paired starting positions')
    parser.add_argument('n_parallel_pairs', type=int, nargs='?', default=15)
    parser.add_argument('n_total_processes', type=int, nargs='?', default=30)
    parser.add_argument('status_every', type=int, nargs='?', default=100)
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


def build_player_command(name, cmd, level, threads_per_engine, hash_level):
    if 'Edax' in name:
        result = cmd + ' -level ' + str(level) + ' -n ' + str(threads_per_engine)
    else:
        result = cmd + ' -l ' + str(level) + ' -t ' + str(threads_per_engine)
    if hash_level is not None:
        result += ' -hash ' + str(hash_level)
    return result


def print_final_summary(players):
    print('\nFinal summary', flush=True)
    for i, player in enumerate(players):
        for j, opponent in enumerate(players):
            if i == j:
                continue
            w, d, l = player[3][j]
            n = player[PLAYED_IDX][j]
            avg_diff = player[4][j] / max(1, n)
            print(
                '{} vs {}: {} avg_disc_diff {:+.2f}'.format(
                    player[NAME_IDX],
                    opponent[NAME_IDX],
                    format_wdl(w, d, l),
                    avg_diff,
                ),
                flush=True,
            )


def main():
    args = parse_args()
    if args.level < 0:
        raise ValueError('level must be non-negative')
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
            player[NAME_IDX],
            player[CMD_IDX],
            args.level,
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
        kifu_path = script_dir / 'transcript' / ('battle_parallel_level_kifu_{}.tsv'.format(time.strftime('%Y%m%d_%H%M%S')))
    elif kifu_path is not None:
        kifu_path = resolve_script_path(script_dir, kifu_path)
    if kifu_path is not None:
        kifu_path = str(kifu_path)
        init_kifu_file(kifu_path)

    print('level:', args.level, flush=True)
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
                    futures.pop(future)
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
    print_final_summary(players)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record_failure(exc)
        shutdown_all_processes()
        print('ERROR:', exc, file=sys.stderr, flush=True)
        raise
