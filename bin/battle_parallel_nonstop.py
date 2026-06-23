import subprocess
import random
import numpy as np
import argparse
import os
import queue
import time
import atexit
import signal
import ctypes
import shutil
import tempfile
from othello_py import *
from elo_rating import Elo_player, update_rating, update_rating_draw
from elo_rating_backcal import fit_elo_from_winrates_with_interval
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


PROBLEM_FILE = 'problem/xot/openingslarge.txt' # XOT (8 moves)
# PROBLEM_FILE = 'problem/random_openings/8_moves/0000000.txt' # 8 random moves

# PROBLEM_FILE = 'problem/ggs_random_openings/14_random_setup2/0000000.txt' # GGS random openings (random_setup_2) under construction


QUIT_TIMEOUT_SEC = 2.0
KILL_TIMEOUT_SEC = 5.0
PROCESS_POOL_GET_TIMEOUT_SEC = 0.2

process_registry = set()
process_registry_lock = threading.Lock()
shutdown_lock = threading.Lock()
shutdown_event = threading.Event()
shutdown_started = False
ntest_runtime_dirs = []
ntest_play_lock = threading.Lock()


def suppress_windows_error_dialogs():
    if os.name != 'nt':
        return
    try:
        sem_failcriticalerrors = 0x0001
        sem_nogpfault_errorbox = 0x0002
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfault_errorbox)
    except Exception:
        pass


suppress_windows_error_dialogs()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('level', type=int)
    parser.add_argument('n_set_games', type=int)
    parser.add_argument('n_parallel_matches', type=int, nargs='?', default=20)
    parser.add_argument('n_total_processes', type=int, nargs='?', default=12)
    parser.add_argument('status_every', type=int, nargs='?', default=20)
    parser.add_argument(
        '--save-kifu',
        '--save-all-kifu',
        dest='save_kifu',
        nargs='?',
        const=True,
        default=None,
        metavar='PATH',
        help='save every game record as TSV. If PATH is omitted, a timestamped file is created.'
    )
    parser.add_argument(
        '--depth',
        '-depth',
        '--depthprobrange',
        '-depthprobrange',
        dest='depth',
        type=int,
        default=None,
        help='set Egaroucid search depth with -depthprobrange 1 60 <depth> 100'
    )
    return parser.parse_args()


args = parse_args()

LEVEL = args.level
DEPTH = args.depth
N_SET_GAMES = args.n_set_games
N_THREADS = 1
N_PARALLEL_MATCHES = args.n_parallel_matches
N_TOTAL_PROCESSES = args.n_total_processes
STATUS_EVERY = args.status_every
SAVE_KIFU_PATH = args.save_kifu
if SAVE_KIFU_PATH is True:
    SAVE_KIFU_PATH = 'transcript/battle_parallel_nonstop_kifu_{}.tsv'.format(time.strftime('%Y%m%d_%H%M%S'))

SCRIPT_START_TIME = time.time()

if N_TOTAL_PROCESSES < 2 or N_TOTAL_PROCESSES % 2 != 0:
    print('N_TOTAL_PROCESSES must be an even number >= 2')
    exit(1)

if N_PARALLEL_MATCHES < 1:
    print('N_PARALLEL_MATCHES must be >= 1')
    exit(1)

if STATUS_EVERY < 1:
    print('STATUS_EVERY must be >= 1')
    exit(1)

if DEPTH is not None and not (1 <= DEPTH <= 60):
    print('DEPTH must be in [1, 60]')
    exit(1)


def format_elapsed(seconds):
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds %= 60
    if hours > 0:
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return '{}:{:02d}'.format(minutes, seconds)


def get_elapsed_text():
    return format_elapsed(time.time() - SCRIPT_START_TIME)


def get_eta_text(completed, total):
    elapsed = time.time() - SCRIPT_START_TIME
    if completed <= 0 or elapsed <= 0:
        return '-'
    remaining = max(0, total - completed)
    return format_elapsed(remaining * elapsed / completed)


def get_battles_per_minute(completed):
    elapsed = time.time() - SCRIPT_START_TIME
    if elapsed <= 0:
        return 0.0
    return 60.0 * completed / elapsed


def init_kifu_file():
    if SAVE_KIFU_PATH is None:
        return

    parent = os.path.dirname(SAVE_KIFU_PATH)
    if parent != '':
        os.makedirs(parent, exist_ok=True)

    with open(SAVE_KIFU_PATH, 'w', encoding='utf-8', newline='') as f:
        f.write('battle\tgame\topening_idx\tp0\tp1\tblack\twhite\tp0_disc_diff\tblack_stones\twhite_stones\trecord\n')


def save_kifu_results(battle_no, opening_idx, game_results):
    if SAVE_KIFU_PATH is None:
        return

    with open(SAVE_KIFU_PATH, 'a', encoding='utf-8', newline='') as f:
        for result in game_results:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                battle_no,
                result['game'],
                opening_idx,
                players[result['p0_idx']][NAME_IDX],
                players[result['p1_idx']][NAME_IDX],
                players[result['black_idx']][NAME_IDX],
                players[result['white_idx']][NAME_IDX],
                result['p0_disc_diff'],
                result['black_stones'],
                result['white_stones'],
                result['record'],
            ))


PROTOCOL_CONSOLE = 'console'
PROTOCOL_GTP = 'gtp'
NTEST_TOTAL_PROCESSES = 2

random.seed(57)

with open(PROBLEM_FILE, 'r') as f:
    openings = [elem for elem in f.read().splitlines()]
random.shuffle(openings)

# name, cmd, protocol (protocol defaults to console)
player_info = [
    # ['latest',  'Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    # ['beta', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -gtp -quiet -nobook'],
    ['new-r16', 'versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe -quiet -nobook -eval ./../model/20260621_1_afterrand16/eval.egev2'],
    # ['0325', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260325_1/eval.egev2'],
    # ['0324', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260324_1/eval.egev2'],
    # ['0323', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260323_1/eval.egev2'],
    # ['0322', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260322_1/eval.egev2'],
    # ['0321', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260321_1/eval.egev2'],
    # ['0320', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260320_1/eval.egev2'],
    # ['0318', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260318_1/eval.egev2'],
    # ['0317', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -quiet -nobook -eval ./../model/20260317_1/eval.egev2'],
    ['7.8.1', 'versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe -quiet -nobook'],
    # ['7.8.0', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe -quiet -nobook'],
    # ['7.6.0', 'versions/Egaroucid_for_Console_7_6_0_Windows_SIMD/Egaroucid_for_Console_7_6_0_SIMD.exe -quiet -nobook'],
    # ['7.5.0', 'versions/Egaroucid_for_Console_7_5_0_Windows_SIMD/Egaroucid_for_Console_7_5_0_SIMD.exe -quiet -nobook'],
    ['Edax4.6', 'versions/edax_4_6/wEdax-x86-64-v3.exe -q'],
    ['Ntest', 'versions/ntest/ntest.exe --gtp', PROTOCOL_GTP],
]

N_BATTLES_PER_ROUND = len(player_info) * (len(player_info) - 1) // 2

NAME_IDX = 0
SUBPROCESS_IDX = 1
RESULT_IDX = 2
RESULT_DISC_IDX = 3
N_PLAYED_IDX = 4
RATING_IDX = 5
PROC_POOL_IDX = 6
CMD_IDX = 7
PROTOCOL_IDX = 8
CWD_IDX = 9


def get_ntest_runtime_root():
    return os.path.join(
        tempfile.gettempdir(),
        'egaroucid_ntest_{}_{}'.format(os.getpid(), time.strftime('%Y%m%d_%H%M%S')),
    )


NTEST_RUNTIME_ROOT = get_ntest_runtime_root()


def ignore_ntest_runtime_files(src, names):
    return [name for name in names if name.lower().endswith('.book')]


def make_ntest_process_start_info(cmd, proc_idx):
    cmd_parts = cmd.split()
    source_exe = cmd_parts[0]
    source_dir = os.path.abspath(os.path.dirname(source_exe))
    target_dir = os.path.join(NTEST_RUNTIME_ROOT, 'proc_{}'.format(proc_idx))
    shutil.copytree(source_dir, target_dir, ignore=ignore_ntest_runtime_files)
    ntest_runtime_dirs.append(target_dir)
    cmd_parts[0] = os.path.join(target_dir, os.path.basename(source_exe))
    return ' '.join(cmd_parts), target_dir


def make_process_start_info(name, cmd, protocol, proc_idx):
    if name == 'Ntest' and protocol == PROTOCOL_GTP:
        return make_ntest_process_start_info(cmd, proc_idx)
    return cmd, None


def get_total_processes_for_player(name):
    if name == 'Ntest':
        return NTEST_TOTAL_PROCESSES
    return N_TOTAL_PROCESSES


def cleanup_ntest_runtime_dirs():
    while ntest_runtime_dirs:
        target_dir = ntest_runtime_dirs.pop()
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass
    try:
        shutil.rmtree(NTEST_RUNTIME_ROOT, ignore_errors=True)
    except Exception:
        pass


def start_engine(cmd, cwd=None):
    popen_kwargs = {
        'stdin': subprocess.PIPE,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.DEVNULL,
    }
    if cwd is not None:
        popen_kwargs['cwd'] = cwd
    if os.name == 'nt':
        popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs['start_new_session'] = True

    with process_registry_lock:
        if shutdown_event.is_set():
            raise RuntimeError('shutdown in progress')
        proc = subprocess.Popen(cmd.split(), **popen_kwargs)
        process_registry.add(proc)
        return proc


def unregister_process(proc):
    with process_registry_lock:
        process_registry.discard(proc)


def kill_process_tree(proc):
    if proc is None or proc.poll() is not None:
        return

    if os.name == 'nt':
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=KILL_TIMEOUT_SEC,
            )
            if result.returncode == 0 or proc.poll() is not None:
                return
        except Exception:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass


def close_process(proc, send_quit=True):
    if proc is None:
        return

    try:
        try:
            if send_quit and proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('quit\n'.encode('utf-8'))
                proc.stdin.flush()
        except Exception:
            pass

        if send_quit:
            try:
                if proc.poll() is None:
                    proc.wait(timeout=QUIT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc)
            except Exception:
                pass
        else:
            kill_process_tree(proc)

        try:
            if proc.poll() is None:
                proc.wait(timeout=KILL_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            try:
                proc.wait(timeout=KILL_TIMEOUT_SEC)
            except Exception:
                pass
        except Exception:
            pass
    finally:
        close_process_pipes(proc)
        unregister_process(proc)


def close_process_pipes(proc):
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:
            pass


def shutdown_all_processes():
    global shutdown_started

    with shutdown_lock:
        if shutdown_started:
            return
        shutdown_started = True
        shutdown_event.set()

    with process_registry_lock:
        all_procs = list(process_registry)

    for proc in all_procs:
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('quit\n'.encode('utf-8'))
                proc.stdin.flush()
        except Exception:
            pass

    quit_deadline = time.time() + QUIT_TIMEOUT_SEC
    for proc in all_procs:
        try:
            if proc.poll() is None:
                proc.wait(timeout=max(0.0, quit_deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    for proc in all_procs:
        if proc.poll() is None:
            kill_process_tree(proc)

    for proc in all_procs:
        try:
            if proc.poll() is None:
                proc.wait(timeout=KILL_TIMEOUT_SEC)
        except Exception:
            pass
        close_process_pipes(proc)
        unregister_process(proc)

    cleanup_ntest_runtime_dirs()


def handle_shutdown_signal(signum, frame):
    shutdown_event.set()
    if shutdown_started:
        return
    raise KeyboardInterrupt


def install_shutdown_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_shutdown_signal)
        except Exception:
            pass
    if hasattr(signal, 'SIGBREAK'):
        try:
            signal.signal(signal.SIGBREAK, handle_shutdown_signal)
        except Exception:
            pass


atexit.register(shutdown_all_processes)
install_shutdown_handlers()


def restart_process(player_idx, proc_idx):
    if shutdown_event.is_set():
        raise RuntimeError('shutdown in progress')
    cmd = players[player_idx][CMD_IDX][proc_idx]
    cwd = players[player_idx][CWD_IDX][proc_idx]
    proc = players[player_idx][SUBPROCESS_IDX][proc_idx]
    close_process(proc, send_quit=False)
    if shutdown_event.is_set():
        raise RuntimeError('shutdown in progress')
    new_proc = start_engine(cmd, cwd=cwd)
    players[player_idx][SUBPROCESS_IDX][proc_idx] = new_proc
    print('restart', players[player_idx][NAME_IDX], 'proc', proc_idx)
    return new_proc


def build_setboard_command(o):
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
    return grid_str


def send_move_command(player_idx, proc_idx, board_command):
    if shutdown_event.is_set():
        raise RuntimeError('shutdown in progress')

    proc = players[player_idx][SUBPROCESS_IDX][proc_idx]
    for _ in range(2):
        if shutdown_event.is_set():
            raise RuntimeError('shutdown in progress')
        if proc.poll() is not None:
            proc = restart_process(player_idx, proc_idx)
        try:
            proc.stdin.write(board_command.encode('utf-8'))
            proc.stdin.flush()
            proc.stdin.write('go\n'.encode('utf-8'))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc = restart_process(player_idx, proc_idx)
            continue

        line = ''
        while line == '' or line == '>':
            raw = proc.stdout.readline()
            if raw == b'':
                proc = restart_process(player_idx, proc_idx)
                line = ''
                break
            line = raw.decode(errors='replace').replace('\r', '').replace('\n', '')
        if line != '':
            return line

    raise RuntimeError('failed to communicate with engine: ' + players[player_idx][NAME_IDX])


def send_gtp_command(player_idx, proc_idx, cmd):
    if shutdown_event.is_set():
        raise RuntimeError('shutdown in progress')

    proc = players[player_idx][SUBPROCESS_IDX][proc_idx]
    if shutdown_event.is_set():
        raise RuntimeError('shutdown in progress')
    if proc.poll() is not None:
        raise RuntimeError('gtp engine exited: ' + players[player_idx][NAME_IDX] + ' cmd=' + cmd.strip())
    try:
        proc.stdin.write(cmd.encode('utf-8'))
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        raise RuntimeError('failed to write to gtp engine: ' + players[player_idx][NAME_IDX] + ' cmd=' + cmd.strip())

    while True:
        raw = proc.stdout.readline()
        if raw == b'':
            raise RuntimeError('gtp engine closed stdout: ' + players[player_idx][NAME_IDX] + ' cmd=' + cmd.strip())

        line = raw.decode(errors='replace').replace('\r', '').replace('\n', '').strip()
        if line == '':
            continue
        if line.startswith('='):
            return line[1:].strip()
        if line.startswith('?'):
            raise RuntimeError('gtp command failed: ' + players[player_idx][NAME_IDX] + ' cmd=' + cmd.strip() + ' response=' + line)

        # Ntest prints startup and status lines before the first GTP response.


def acquire_process_idx(player_idx, player):
    proc_pool = players[player_idx][PROC_POOL_IDX][player]
    while True:
        if shutdown_event.is_set():
            raise RuntimeError('shutdown in progress')
        try:
            return proc_pool.get(timeout=PROCESS_POOL_GET_TIMEOUT_SEC)
        except queue.Empty:
            pass


def supports_depthprobrange(cmd):
    return 'Egaroucid_for_Console' in cmd


def normalize_player_info(info):
    if len(info) == 2:
        return info[0], info[1], PROTOCOL_CONSOLE
    if len(info) == 3:
        return info[0], info[1], info[2]
    raise ValueError('invalid player_info entry: ' + str(info))


def is_gtp_player(player_idx):
    return players[player_idx][PROTOCOL_IDX] == PROTOCOL_GTP


def gtp_color(player):
    return 'b' if player == black else 'w'


def clear_engine_board(player_idx, proc_idx):
    if is_gtp_player(player_idx):
        send_gtp_command(player_idx, proc_idx, 'clear_board\n')


def play_engine_move(player_idx, proc_idx, player, coord):
    if is_gtp_player(player_idx):
        send_gtp_command(player_idx, proc_idx, 'play ' + gtp_color(player) + ' ' + coord + '\n')


def pass_engine_move(player_idx, proc_idx, player):
    if is_gtp_player(player_idx):
        send_gtp_command(player_idx, proc_idx, 'play ' + gtp_color(player) + ' pass\n')


def gen_engine_move(player_idx, proc_idx, o):
    if is_gtp_player(player_idx):
        return send_gtp_command(player_idx, proc_idx, 'genmove ' + gtp_color(o.player) + '\n')
    return send_move_command(player_idx, proc_idx, build_setboard_command(o))


def is_ntest_player(player_idx):
    return players[player_idx][NAME_IDX] == 'Ntest'


def should_serialize_game(p0_idx, p1_idx):
    return is_ntest_player(p0_idx) or is_ntest_player(p1_idx)


players = []
results_lock = threading.Lock()

for info in player_info:
    name, cmd, protocol = normalize_player_info(info)
    # level option
    if name == '6.0.X':
        cmd_with_options = cmd + ' ' + str(LEVEL)
    else:
        cmd_with_options = cmd + ' -l ' + str(LEVEL)
    # depth option
    if DEPTH is not None and supports_depthprobrange(cmd):
        cmd_with_options += ' -depthprobrange 1 60 ' + str(DEPTH) + ' 100'
    # thread option
    if 'Edax' in name:
        cmd_with_options += ' -n ' + str(N_THREADS)
    elif 'Neural' in name:
        cmd_with_options += ' --threads ' + str(N_THREADS)
    else:
        cmd_with_options += ' -t ' + str(N_THREADS)
    n_player_processes = get_total_processes_for_player(name)
    if n_player_processes == N_TOTAL_PROCESSES:
        print(name, cmd_with_options)
    else:
        print(name, cmd_with_options, '(processes: {})'.format(n_player_processes))

    process_cmds = []
    process_cwds = []
    subprocesses = []
    for proc_idx in range(n_player_processes):
        process_cmd, process_cwd = make_process_start_info(name, cmd_with_options, protocol, proc_idx)
        process_cmds.append(process_cmd)
        process_cwds.append(process_cwd)
        subprocesses.append(start_engine(process_cmd, cwd=process_cwd))

    proc_pool = [queue.Queue(), queue.Queue()]
    half = n_player_processes // 2
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
        proc_pool,
        process_cmds,
        protocol,
        process_cwds
    ])


def play_single_game(p0_idx, p1_idx, opening_idx, p0_is_black):
    if should_serialize_game(p0_idx, p1_idx):
        with ntest_play_lock:
            return play_single_game_impl(p0_idx, p1_idx, opening_idx, p0_is_black)
    return play_single_game_impl(p0_idx, p1_idx, opening_idx, p0_is_black)


def play_single_game_impl(p0_idx, p1_idx, opening_idx, p0_is_black):
    player_idxes = [p0_idx, p1_idx]
    opening = openings[opening_idx]
    player = 1 if p0_is_black else 0
    p0_proc_idx = None
    p1_proc_idx = None
    record = ''
    o = othello()
    try:
        p0_proc_idx = acquire_process_idx(p0_idx, player)
        p1_proc_idx = acquire_process_idx(p1_idx, player)

        clear_engine_board(p0_idx, p0_proc_idx)
        clear_engine_board(p1_idx, p1_proc_idx)

        for i in range(0, len(opening), 2):
            if not o.check_legal():
                for player_idx in [p0_idx, p1_idx]:
                    proc_idx = p0_proc_idx if player_idx == p0_idx else p1_proc_idx
                    pass_engine_move(player_idx, proc_idx, o.player)
                o.player = 1 - o.player
                o.check_legal()
            x = ord(opening[i].lower()) - ord('a')
            y = int(opening[i + 1]) - 1
            coord = opening[i] + opening[i + 1]
            for player_idx in [p0_idx, p1_idx]:
                proc_idx = p0_proc_idx if player_idx == p0_idx else p1_proc_idx
                play_engine_move(player_idx, proc_idx, o.player, coord)
            record += coord
            o.move(y, x)

        while True:
            if not o.check_legal():
                for player_idx in [p0_idx, p1_idx]:
                    proc_idx = p0_proc_idx if player_idx == p0_idx else p1_proc_idx
                    pass_engine_move(player_idx, proc_idx, o.player)
                o.player = 1 - o.player
                if not o.check_legal():
                    break

            player_idx = player_idxes[o.player ^ player]
            proc_idx = p0_proc_idx if player_idx == p0_idx else p1_proc_idx
            line = gen_engine_move(player_idx, proc_idx, o)
            coord = line[-2:].lower()
            try:
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
            except Exception:
                print('error')
                print(build_setboard_command(o)[:-1])
                print(o.player, player)
                print(line)
                print(coord)
                raise RuntimeError('invalid move coordinate from engine')

            record += chr(ord('a') + x) + str(y + 1)
            o_player = o.player
            if not o.move(y, x):
                o.print_info()
                print(build_setboard_command(o)[:-1])
                print(o.player, player)
                print(line)
                print(coord)
                print(y, x)
                raise RuntimeError('illegal move from engine')

            n_player_idx = player_idxes[o_player ^ 1 ^ player]
            n_proc_idx = p0_proc_idx if n_player_idx == p0_idx else p1_proc_idx
            play_engine_move(n_player_idx, n_proc_idx, o_player, coord)

        p0_disc_diff = o.n_stones[player] - o.n_stones[1 - player]
        empty = 64 - (o.n_stones[player] + o.n_stones[1 - player])
        if p0_disc_diff > 0:
            p0_disc_diff += empty
        elif p0_disc_diff < 0:
            p0_disc_diff -= empty

        return {
            'game': 'p0_black' if player == black else 'p0_white',
            'p0_idx': p0_idx,
            'p1_idx': p1_idx,
            'black_idx': p0_idx if player == black else p1_idx,
            'white_idx': p0_idx if player == white else p1_idx,
            'p0_disc_diff': p0_disc_diff,
            'black_stones': o.n_stones[black],
            'white_stones': o.n_stones[1 - black],
            'record': record,
        }
    finally:
        if p0_proc_idx is not None:
            players[p0_idx][PROC_POOL_IDX][player].put(p0_proc_idx)
        if p1_proc_idx is not None:
            players[p1_idx][PROC_POOL_IDX][player].put(p1_proc_idx)


def play_battle(p0_idx, p1_idx, opening_idx):
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        future_black = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, True)
        future_white = executor.submit(play_single_game, p0_idx, p1_idx, opening_idx, False)
        future_results = {}

        for future in as_completed([future_black, future_white]):
            try:
                future_results[future] = future.result()
            except Exception:
                shutdown_all_processes()
                raise

        game_results = [future_results[future_black], future_results[future_white]]
        sum_disc_diff_p0 = sum(result['p0_disc_diff'] for result in game_results)
    finally:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=True)

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

    return game_results


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


def format_estimated_rating_value(estimated_rating):
    if estimated_rating is None:
        return '-'
    est, _ = estimated_rating
    return '{:.1f}'.format(est)


def build_estimated_ratings_excel_text(estimated_ratings):
    names = [players[i][NAME_IDX] for i in range(len(players))]
    values = [
        format_estimated_rating_value(estimated_ratings.get(name))
        for name in names
    ]

    return '\n'.join(values)


def print_estimated_ratings_for_excel(estimated_ratings):
    print(build_estimated_ratings_excel_text(estimated_ratings))


def copy_estimated_ratings_excel_text_to_clipboard(estimated_ratings):
    try:
        import pyperclip
        pyperclip.copy(build_estimated_ratings_excel_text(estimated_ratings))
        print('Estimated ratings copied to clipboard.')
    except ImportError:
        print('Clipboard copy skipped: pyperclip is not installed.')
    except Exception as e:
        print('Clipboard copy failed: {}'.format(e))


def print_all_result_locked(copy_estimated_ratings_to_clipboard=False):
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

    print_estimated_ratings_for_excel(estimated_ratings)
    if copy_estimated_ratings_to_clipboard:
        copy_estimated_ratings_excel_text_to_clipboard(estimated_ratings)


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


def print_status(completed, total, target_per_pair, copy_estimated_ratings_to_clipboard=False):
    percent = 100.0 * completed / max(1, total)
    print('\n' + '=' * 80)
    print('Progress: {}/{} ({:.2f}%)'.format(completed, total, percent))
    print('Elapsed: {}  ETA: {}  Speed: {:.2f} battles/min'.format(
        get_elapsed_text(),
        get_eta_text(completed, total),
        get_battles_per_minute(completed),
    ))
    depth_text = ''
    if DEPTH is not None:
        depth_text = ' depth ' + str(DEPTH)
    print(str(completed // N_BATTLES_PER_ROUND) + ' matches played for each win rate at level ' + str(LEVEL) + depth_text + ' ' + str(N_THREADS) + ' threads')
    with results_lock:
        print_all_result_locked(copy_estimated_ratings_to_clipboard)
        print_games_progress_locked(target_per_pair)


def print_collect_progress(completed, total):
    percent = 100.0 * completed / max(1, total)
    print(
        'Collected battle {}/{} ({:.2f}%) elapsed {} eta {} speed {:.2f} battles/min'.format(
            completed,
            total,
            percent,
            get_elapsed_text(),
            get_eta_text(completed, total),
            get_battles_per_minute(completed),
        ),
        flush=True,
    )


print('n_players', len(players))
print('level', LEVEL)
if DEPTH is not None:
    print('depthprobrange:', 0, 60, DEPTH, 100)
print('parallel matches:', N_PARALLEL_MATCHES)
print('total processes per player:', N_TOTAL_PROCESSES)
print('status interval (battles):', STATUS_EVERY)
if SAVE_KIFU_PATH is not None:
    init_kifu_file()
    print('save kifu:', SAVE_KIFU_PATH)

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

executor = None
try:
    executor = ThreadPoolExecutor(max_workers=N_PARALLEL_MATCHES)
    iterator = iter(all_battles)
    futures = {}

    for _ in range(min(N_PARALLEL_MATCHES, total_battles)):
        p0, p1, opening_idx = next(iterator)
        future = executor.submit(play_battle, p0, p1, opening_idx)
        futures[future] = (p0, p1, opening_idx)

    while futures:
        for future in as_completed(list(futures.keys())):
            p0, p1, opening_idx = futures.pop(future)
            game_results = future.result()
            completed_battles += 1
            save_kifu_results(completed_battles, opening_idx, game_results)
            # print_collect_progress(completed_battles, total_battles)

            if completed_battles % STATUS_EVERY == 0 or completed_battles == total_battles:
                print_status(completed_battles, total_battles, N_SET_GAMES)

            try:
                p0, p1, opening_idx = next(iterator)
            except StopIteration:
                pass
            else:
                nf = executor.submit(play_battle, p0, p1, opening_idx)
                futures[nf] = (p0, p1, opening_idx)
            break
finally:
    shutdown_all_processes()
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=True)

print('\nAll battles finished.')
print_status(completed_battles, total_battles, N_SET_GAMES, copy_estimated_ratings_to_clipboard=True)
