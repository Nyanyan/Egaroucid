"""
Egaroucid と他のオセロAIとのベンチマーク対戦スクリプト (Issue #616)

3種類の対戦条件に対応する
    depth    : 固定レベル(固定深さ)        --level <level>
    movetime : 1手あたりの時間を固定       --time <秒/手>
    gametime : 持ち時間を固定 (切れ負け)   --time <秒/局>

使い方 (bin/ で実行する)
    python battle_engines.py depth --level 11 --rounds 100
    python battle_engines.py movetime --time 1 --rounds 50 --players beta Edax4.6 Neural
    python battle_engines.py gametime --time 60 --rounds 20 --gauntlet --parallel 4 --forfeit
    python battle_engines.py --list

1ラウンドで、全組み合わせ (--gauntlet なら先頭プレイヤー vs 他) が同じ序盤を先後入れ替えて2局ずつ打つ。
勝敗は既存の battle*.py と同じく、2局の石差の合計で1勝負として数える。

対戦相手は下の PLAYERS に登録する。エンジンごとの起動オプションや時間設定の違いは PRESETS で吸収している。
"""

import argparse
import collections
import copy
import datetime
import json
import os
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from elo_rating_backcal import fit_elo_from_winrates_with_interval

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# エンジンの種類ごとの設定
#
# protocol
#   'gtp'  : GTP (clear_board / play / genmove)
#   'text' : Edax 形式のコンソール (setboard / go)。毎手 setboard で局面を送る
# args     : 常に付ける起動オプション
# init     : 起動直後に1回だけ送るコマンド
# sync     : (text のみ) 起動完了を待つために送るコマンド。何か1行返ってくれば良い
# depth / movetime / gametime : 対戦条件ごとの設定。None ならその条件には非対応
#   args   : 追加の起動オプション
#   setup  : (gtp) 各対局の clear_board の前に送るコマンド
#   before : (gtp) 毎回の genmove の直前に送るコマンド
#   pre    : (text) 毎手 setboard の前に送るコマンド
#   post   : (text) 毎手 setboard の後, go の前に送るコマンド
# integer_seconds : 時間を整数秒でしか受け付けないエンジン
# min_time        : movetime で指定できる最小の秒数
#
# テンプレートで使える値
#   {threads} {level}
#   {time} {time_int} {ms}            : --time の値 (秒, 整数秒, ミリ秒)
#   {remaining} {remaining_int}       : その手番の残り時間 (秒, 整数秒に切り捨て)。movetime では --time と同じ
#   {color} {color_name}              : b / w, black / white
# ---------------------------------------------------------------------------
PRESETS = {
    # Egaroucid for Console 7.9.0 以降 (GTP の time_settings / time_left に対応したもの)
    'egaroucid': {
        'protocol': 'gtp',
        'args': '-gtp -quiet -nobook -t {threads}',
        'depth': {'args': '-l {level}'},
        'movetime': {'setup': ['time_settings 0 {time} 1'], 'before': ['time_left {color} {time} 1']},
        'gametime': {'setup': ['time_settings {time} 0 0'], 'before': ['time_left {color} {remaining} 0']},
    },
    # Egaroucid for Console 7.8.0 以前 (コンソールモード)。1手あたりの時間指定はできない
    # settime が整数秒なので、持ち時間は1秒未満を切り捨てて渡す
    'egaroucid_text': {
        'protocol': 'text',
        'args': '-quiet -nobook -t {threads}',
        'sync': 'version',
        'depth': {'args': '-l {level}'},
        'movetime': None,
        'gametime': {'post': ['settime {color} {remaining_int}']},
    },
    # Edax 4.x 系: abulmo/edax-reversi, okuhara/edax-reversi-AVX, Yoka0014/edax-fm-reversi (GTP)
    # Edax は -move-time に1秒未満を指定しても1秒に丸める
    # Edax の GTP の time_settings は main_time を整数で読む (time_left は小数可)
    'edax': {
        'protocol': 'gtp',
        'args': '-gtp -q -n {threads} -book-usage off',
        'depth': {'args': '-l {level}'},
        'movetime': {'args': '-l 60 -move-time {time}', 'min_time': 1.0},
        'gametime': {'args': '-l 60', 'setup': ['time_settings {time_int} 0 0'], 'before': ['time_left {color} {remaining} 0']},
    },
    # Edax 形式のコンソール (setboard / go)。Edax 本体でも使えるが, Edax なら GTP の 'edax' を推奨
    # 毎手 setboard するので Edax の置換表が毎手消える。また game-time は1秒未満を1秒に切り上げるので,
    # 短い持ち時間では時間切れしやすい
    'edax_text': {
        'protocol': 'text',
        'args': '-q -n {threads} -book-usage off',
        'init': ['mode 3'],
        'sync': 'version',
        'depth': {'args': '-l {level}'},
        'movetime': {'args': '-l 60 -move-time {time}', 'min_time': 1.0},
        'gametime': {'args': '-l 60', 'pre': ['game-time {remaining}']},
    },
    # MRConsole (MasterReversi のコンソール版, Edax 互換の操作体系) 動作未確認
    # Edax と同じオプション・コマンド (-l, -n, -move-time, setboard, go, game-time, mode) を想定している。
    # 動かない場合はここを調整する (GTP に対応していれば 'edax' プリセットを試す)
    # sync を指定しないので, 最初の1手の時間に起動時間が含まれる
    'mrconsole': {
        'protocol': 'text',
        'args': '-q -n {threads}',
        'init': ['mode 3'],
        'sync': None,
        'depth': {'args': '-l {level}'},
        'movetime': {'args': '-l 60 -move-time {time}'},
        'gametime': {'args': '-l 60', 'pre': ['game-time {remaining}']},
    },
    # natsutteatsuiyone/neural-reversi の cli (GTP)。time_settings / time_left は整数秒のみ
    # time_settings 0 <t> 0 が「1手 t 秒」
    'neural': {
        'protocol': 'gtp',
        'args': 'gtp --threads {threads}',
        'depth': {'args': '--level {level}'},
        'movetime': {'setup': ['time_settings 0 {time_int} 0']},
        'gametime': {'setup': ['time_settings {time_int} 0 0'], 'before': ['time_left {color} {remaining_int} 0']},
        'integer_seconds': True,
    },
    # htz/kuroobi の gtp (GTP)。level はそのまま中盤の探索深さ
    # 1手の時間は --time-ms で打ち切る。読み切りは --solve-empties 以下の空きマスで行う
    'kuroobi': {
        'protocol': 'gtp',
        'args': '-t {threads}',
        'depth': {'args': '-l {level}'},
        'movetime': {'args': '-l 60 --solve-empties 20 --time-ms {ms}'},
        'gametime': {'setup': ['time_settings {time} 0 0'], 'before': ['time_left {color} {remaining} 0']},
    },
}

# ---------------------------------------------------------------------------
# 対戦するプレイヤー
#   [名前, プリセット名, コマンド(実行ファイル + 追加オプション), (省略可) オプション辞書]
#   追加オプションはプリセットのオプションの後ろに付く
#   オプション辞書: 'cwd' (作業ディレクトリ), 'level' (このプレイヤーだけ depth のレベルを変える),
#                   'args' / 'init' / 'depth' / 'movetime' / 'gametime' など PRESETS の項目の上書き
# --players を省略したときは, 実行ファイルが見つかるプレイヤー全員が対戦する
# レベルの意味はエンジンごとに異なる (Edax / neural-reversi はそれぞれのレベル, kuroobi は中盤の探索深さ)
# ---------------------------------------------------------------------------
PLAYERS = [
    # Egaroucid (7.9.0 以降は GTP の時間制御に対応。7.8.0 以前は egaroucid_text を使う)
    ['beta', 'egaroucid', 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe'],
    ['7.8.0', 'egaroucid_text', 'versions/Egaroucid_for_Console_7_8_0_Windows_SIMD/Egaroucid_for_Console_7_8_0_SIMD.exe'],
    # Edax 系は作業ディレクトリの data/eval.dat を読むので, cwd を設定するか -eval-file <file> を付ける
    # https://github.com/abulmo/edax-reversi
    ['Edax4.6', 'edax', 'versions/edax_4_6/wEdax-x86-64-v3.exe', {'cwd': 'versions/edax_4_6'}],
    # https://github.com/okuhara/edax-reversi-AVX
    ['EdaxAVX', 'edax', 'versions/edax_avx/wEdax-x64-avx2.exe', {'cwd': 'versions/edax_avx'}],
    # https://github.com/Yoka0014/edax-fm-reversi (FM の潜在ベクトル付きの専用 eval.dat が必要)
    ['EdaxFM', 'edax', 'versions/edax_fm/wEdax-x86-64-v3.exe', {'cwd': 'versions/edax_fm'}],
    # https://github.com/natsutteatsuiyone/neural-reversi (時間制御には GTP の time_settings に対応した版が必要。6.3.0 で確認)
    ['Neural', 'neural', 'versions/neural-reversi-cli-5.0.0-windows-x86_64-v3.exe'],
    # https://github.com/htz/kuroobi の gtp。作業ディレクトリの weights/ (nnue.bin, linear.bin) を読む
    ['Kuroobi', 'kuroobi', 'versions/kuroobi/gtp.exe', {'cwd': 'versions/kuroobi'}],
    # https://t-ishii.la.coocan.jp/download/MRConsole.html
    ['MRConsole', 'mrconsole', 'versions/MRConsole/MRConsole.exe', {'cwd': 'versions/MRConsole'}],
]

DEFAULT_OPENINGS = 'problem/xot/openingslarge.txt'
STARTUP_TIMEOUT = 300.0  # 起動完了を待つ秒数 (neural-reversi などは重みの読み込みに時間がかかる)
COMMAND_TIMEOUT = 60.0   # genmove 以外のコマンドの応答を待つ秒数


# ---------------------------------------------------------------------------
# 盤面 (bitboard, bit = y * 8 + x, a1 = 0, h8 = 63)
# ---------------------------------------------------------------------------
FULL = (1 << 64) - 1
NOT_EDGE_X = 0x7E7E7E7E7E7E7E7E
DIRECTIONS = ((1, NOT_EDGE_X), (8, FULL), (7, NOT_EDGE_X), (9, NOT_EDGE_X))
BLACK = 0
WHITE = 1
COLOR_GTP = ('b', 'w')
COLOR_NAME = ('black', 'white')


def legal_moves(p, o):
    empty = ~(p | o) & FULL
    moves = 0
    for d, mask in DIRECTIONS:
        om = o & mask
        t = om & (p << d)
        for _ in range(5):
            t |= om & (t << d)
        moves |= t << d
        t = om & (p >> d)
        for _ in range(5):
            t |= om & (t >> d)
        moves |= t >> d
    return moves & empty


def calc_flips(p, o, pos):
    x = 1 << pos
    flips = 0
    for d, mask in DIRECTIONS:
        om = o & mask
        t = 0
        y = (x << d) & FULL
        while y & om:
            t |= y
            y = (y << d) & FULL
        if y & p:
            flips |= t
        t = 0
        y = x >> d
        while y & om:
            t |= y
            y >>= d
        if y & p:
            flips |= t
    return flips


def coord_to_pos(coord):
    """'f5' / 'F5' -> 37, それ以外は None"""
    coord = coord.strip().lower()
    if len(coord) != 2 or not ('a' <= coord[0] <= 'h') or not ('1' <= coord[1] <= '8'):
        return None
    return (ord(coord[1]) - ord('1')) * 8 + (ord(coord[0]) - ord('a'))


def pos_to_coord(pos):
    return chr(ord('a') + pos % 8) + str(pos // 8 + 1)


class Board:
    def __init__(self):
        self.discs = [0x0000000810000000, 0x0000001008000000]  # black, white
        self.turn = BLACK

    def legal(self):
        return legal_moves(self.discs[self.turn], self.discs[self.turn ^ 1])

    def move(self, pos):
        p = self.discs[self.turn]
        o = self.discs[self.turn ^ 1]
        if not (self.legal() >> pos) & 1:
            return False
        flips = calc_flips(p, o, pos)
        self.discs[self.turn] = p | flips | (1 << pos)
        self.discs[self.turn ^ 1] = o ^ flips
        self.turn ^= 1
        return True

    def pass_turn(self):
        self.turn ^= 1

    def is_end(self):
        return self.legal() == 0 and legal_moves(self.discs[self.turn ^ 1], self.discs[self.turn]) == 0

    def black_disc_diff(self):
        """黒から見た石差 (空きマスは勝った方に加える)"""
        b = bin(self.discs[BLACK]).count('1')
        w = bin(self.discs[WHITE]).count('1')
        e = 64 - b - w
        if b > w:
            return b - w + e
        if b < w:
            return b - w - e
        return 0

    def to_setboard(self):
        """setboard 用の文字列 (a1, b1, ..., h8 の順に X / O / -, 最後に手番)"""
        s = []
        for i in range(64):
            if (self.discs[BLACK] >> i) & 1:
                s.append('X')
            elif (self.discs[WHITE] >> i) & 1:
                s.append('O')
            else:
                s.append('-')
        return ''.join(s) + (' X' if self.turn == BLACK else ' O')


# ---------------------------------------------------------------------------
# エンジンとの通信
# ---------------------------------------------------------------------------
class EngineError(Exception):
    pass


def format_seconds(t):
    s = '{:.3f}'.format(max(0.0, t)).rstrip('0').rstrip('.')
    return s if s else '0'


def split_command(cmd):
    if os.name == 'nt':
        return [elem.strip('"') for elem in shlex.split(cmd, posix=False)]
    return shlex.split(cmd)


class Player:
    """PLAYERS の1エントリ"""

    def __init__(self, idx, entry):
        if len(entry) < 3:
            raise ValueError('player entry must be [name, preset, command(, options)]: ' + str(entry))
        self.idx = idx
        self.name = entry[0]
        self.preset_name = entry[1]
        if self.preset_name not in PRESETS:
            raise ValueError('unknown preset ' + self.preset_name + ' for ' + self.name)
        options = entry[3] if len(entry) >= 4 else {}
        self.conf = copy.deepcopy(PRESETS[self.preset_name])
        for key, value in options.items():
            if key not in ('cwd', 'level'):
                self.conf[key] = value
        self.cwd = options.get('cwd')
        self.level = options.get('level')
        self.cmd_tokens = split_command(entry[2])

    def mode_conf(self, mode):
        return self.conf.get(mode)

    def check(self, settings):
        """この対戦条件で使えるか確認し, 問題があればエラーメッセージを返す"""
        conf = self.mode_conf(settings.mode)
        if conf is None:
            return '{} (preset {}) does not support {} mode'.format(self.name, self.preset_name, settings.mode)
        if settings.mode != 'depth':
            if self.conf.get('integer_seconds') and settings.time != int(settings.time):
                return '{} (preset {}) accepts only integer seconds (--time {})'.format(self.name, self.preset_name, settings.time)
            if settings.mode == 'movetime' and settings.time < conf.get('min_time', 0.0):
                return '{} (preset {}) needs --time >= {} in movetime mode'.format(self.name, self.preset_name, conf['min_time'])
        exe = self.executable()
        if os.path.isabs(exe) and not os.path.isfile(exe):
            return '{}: executable not found: {}'.format(self.name, exe)
        if self.cwd is not None and not os.path.isdir(self.cwd):
            return '{}: cwd not found: {}'.format(self.name, self.cwd)
        return None

    def executable(self):
        exe = self.cmd_tokens[0]
        if not os.path.isabs(exe) and (os.sep in exe or '/' in exe):
            exe = os.path.abspath(exe)  # cwd を変えても実行ファイルが見つかるように
        return exe

    def values(self, settings):
        level = self.level if self.level is not None else settings.level
        v = {
            'threads': settings.threads,
            'level': level,
        }
        if settings.time is not None:
            v['time'] = format_seconds(settings.time)
            v['time_int'] = int(settings.time)
            v['ms'] = int(round(settings.time * 1000))
        return v

    def argv(self, settings):
        v = self.values(settings)
        conf = self.mode_conf(settings.mode)
        argv = [self.executable()]
        argv += split_command(self.conf.get('args', '').format(**v))
        argv += split_command(conf.get('args', '').format(**v))
        argv += self.cmd_tokens[1:]
        return argv


ALL_PROCESSES = []
ALL_PROCESSES_LOCK = threading.Lock()


class EngineProcess:
    def __init__(self, player, settings):
        self.player = player
        self.settings = settings
        self.conf = player.mode_conf(settings.mode)
        self.base_values = player.values(settings)
        self.argv = player.argv(settings)
        self.lines = queue.Queue()
        self.stderr_tail = collections.deque(maxlen=5)  # エラー表示用
        self.alive = True
        try:
            self.proc = subprocess.Popen(self.argv, cwd=player.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise EngineError('{}: failed to start {}: {}'.format(player.name, ' '.join(self.argv), e))
        with ALL_PROCESSES_LOCK:
            ALL_PROCESSES.append(self)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        try:
            for raw in iter(self.proc.stdout.readline, b''):
                self.lines.put(raw.decode('utf-8', errors='replace').rstrip('\r\n'))
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def _read_stderr(self):
        try:
            for raw in iter(self.proc.stderr.readline, b''):
                line = raw.decode('utf-8', errors='replace').strip()
                if line:
                    self.stderr_tail.append(line[-200:])
        except (OSError, ValueError):
            pass

    def terminated_message(self):
        try:
            self.proc.wait(timeout=0.5)  # stderr を読み終えるのを少し待つ
        except subprocess.TimeoutExpired:
            pass
        msg = self.player.name + ': engine terminated'
        if self.stderr_tail:
            msg += ' (stderr: ' + ' / '.join(self.stderr_tail) + ')'
        return msg

    def send(self, line):
        if not self.alive:
            raise EngineError(self.player.name + ': engine is not running')
        try:
            self.proc.stdin.write((line + '\n').encode('utf-8'))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.alive = False
            raise EngineError(self.terminated_message() + ' while sending "' + line + '"')

    def readline(self, deadline):
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise EngineError(self.player.name + ': timeout')
        if line is None:
            self.alive = False
            raise EngineError(self.terminated_message())
        return line

    def drain(self):
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.alive = False
                return

    def template_values(self, color=None, remaining=None):
        v = dict(self.base_values)
        if color is not None:
            v['color'] = COLOR_GTP[color]
            v['color_name'] = COLOR_NAME[color]
        if remaining is not None:
            v['remaining'] = format_seconds(remaining)
            v['remaining_int'] = int(max(0.0, remaining))
        return v

    def kill(self):
        self.alive = False
        try:
            self.proc.kill()
        except OSError:
            pass

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'quit\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self.kill()


class GtpEngine(EngineProcess):
    def command(self, cmd, timeout=COMMAND_TIMEOUT):
        """GTP コマンドを送り, (成功したか, 応答) を返す"""
        self.send(cmd)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            line = self.readline(deadline).strip()
            if line.startswith('=') or line.startswith('?'):
                break
            # GTP の応答以外の出力は読み飛ばす
        ok = line[0] == '='
        body = line[1:].strip()
        while True:  # 応答は空行で終わる
            if self.readline(deadline).strip() == '':
                break
        return ok, body

    def check(self, cmd, timeout=COMMAND_TIMEOUT):
        ok, body = self.command(cmd, timeout)
        if not ok:
            raise EngineError('{}: "{}" failed: {}'.format(self.player.name, cmd, body))
        return body

    def startup(self):
        self.check('protocol_version', STARTUP_TIMEOUT)
        v = self.template_values()
        for cmd in self.player.conf.get('init', []):
            self.check(cmd.format(**v))
        # 時間制御のコマンドに対応しているか, 対局前に1回送って確かめる (残り時間は小数になる値で試す)
        remaining = None if self.settings.time is None else self.settings.time * 0.987
        v = self.template_values(BLACK, remaining)
        for cmd in self.conf.get('setup', []) + self.conf.get('before', []):
            ok, body = self.command(cmd.format(**v))
            if not ok:
                raise EngineError('{}: "{}" is not accepted ({}). check the preset ({})'.format(self.player.name, cmd.format(**v), body, self.player.preset_name))

    def new_game(self):
        v = self.template_values()
        for cmd in self.conf.get('setup', []):
            self.check(cmd.format(**v))
        self.check('clear_board')

    def play(self, color, coord):
        self.check('play {} {}'.format(COLOR_GTP[color], coord))

    def genmove(self, board, color, remaining, timeout):
        v = self.template_values(color, remaining)
        for cmd in self.conf.get('before', []):
            self.check(cmd.format(**v))
        strt = time.perf_counter()
        ok, body = self.command('genmove ' + COLOR_GTP[color], timeout)
        elapsed = time.perf_counter() - strt
        if not ok:
            raise EngineError('{}: genmove failed: {}'.format(self.player.name, body))
        return body.split()[-1] if body else '', elapsed


MOVE_PATTERN = re.compile(r'(?:^|plays\s+)([A-Ha-h][1-8]|[Pp][Ss]|[Pp][Aa][Ss][Ss])\s*$')


class TextEngine(EngineProcess):
    def startup(self):
        v = self.template_values()
        for cmd in self.player.conf.get('init', []):
            self.send(cmd.format(**v))
        sync = self.player.conf.get('sync')
        if sync:
            self.send(sync)
            self.readline(time.monotonic() + STARTUP_TIMEOUT)
            time.sleep(0.3)
            self.drain()

    def new_game(self):
        pass

    def play(self, color, coord):
        pass  # 毎手 setboard で局面を送るので何もしない

    def genmove(self, board, color, remaining, timeout):
        v = self.template_values(color, remaining)
        self.drain()
        for cmd in self.conf.get('pre', []):
            self.send(cmd.format(**v))
        self.send('setboard ' + board.to_setboard())
        for cmd in self.conf.get('post', []):
            self.send(cmd.format(**v))
        strt = time.perf_counter()
        self.send('go')
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            line = self.readline(deadline)
            m = MOVE_PATTERN.search(line.strip())
            if m:
                return m.group(1), time.perf_counter() - strt


def start_engine(player, settings):
    cls = GtpEngine if player.conf['protocol'] == 'gtp' else TextEngine
    engine = cls(player, settings)
    try:
        engine.startup()
    except EngineError:
        engine.kill()
        raise
    return engine


class EnginePool:
    """プレイヤーごとのエンジンプロセスの使い回し (最大 max_size 個まで起動)"""

    def __init__(self, player, settings, max_size):
        self.player = player
        self.settings = settings
        self.max_size = max_size
        self.idle = queue.Queue()
        self.n_procs = 0
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            try:
                engine = self.idle.get_nowait()
            except queue.Empty:
                engine = None
                with self.lock:
                    can_start = self.n_procs < self.max_size
                    if can_start:
                        self.n_procs += 1
                if can_start:
                    try:
                        return start_engine(self.player, self.settings)
                    except EngineError:
                        with self.lock:
                            self.n_procs -= 1
                        raise
                engine = self.idle.get()
            if engine.alive and engine.proc.poll() is None:
                return engine
            self.discard(engine)

    def release(self, engine):
        if engine.alive and engine.proc.poll() is None:
            self.idle.put(engine)
        else:
            self.discard(engine)

    def discard(self, engine):
        engine.kill()
        with self.lock:
            self.n_procs -= 1


# ---------------------------------------------------------------------------
# 対局
# ---------------------------------------------------------------------------
class Settings:
    def __init__(self, args):
        self.mode = args.mode
        self.level = args.level
        self.time = args.time
        self.threads = args.threads
        self.forfeit = args.forfeit
        self.grace = args.grace
        if args.timeout is not None:
            self.timeout = args.timeout if args.timeout > 0 else None
        elif self.mode == 'depth':
            self.timeout = None
        elif self.mode == 'movetime':
            self.timeout = self.time + 60.0
        else:
            self.timeout = self.time * 2 + 60.0

    def description(self):
        if self.mode == 'depth':
            return 'level {}'.format(self.level)
        if self.mode == 'movetime':
            return '{} sec/move'.format(format_seconds(self.time))
        return '{} sec/game'.format(format_seconds(self.time))


def parse_opening(line):
    line = line.strip()
    moves = []
    for i in range(0, len(line) - 1, 2):
        coord = line[i:i + 2]
        if coord.lower() in ('ps', 'pa'):
            continue
        pos = coord_to_pos(coord)
        if pos is None:
            raise ValueError('invalid opening: ' + line)
        moves.append(pos)
    return moves


def play_game(engines, opening, settings):
    """
    engines[BLACK], engines[WHITE] で1局打つ
    戻り値: dict (黒から見た石差, 棋譜, 各手の思考時間, 反則負けの情報)
    """
    board = Board()
    record = ''
    times = [[], []]
    clock = [settings.time, settings.time]  # movetime では毎手 settings.time のまま
    overtime = [0, 0]
    result = {'forfeit': None}

    def forfeit(color, reason):
        result['forfeit'] = {'color': COLOR_NAME[color], 'player': engines[color].player.name, 'reason': reason}

    def notify_all(color, coord, skip=None):
        for c in (BLACK, WHITE):
            if c == skip:
                continue
            try:
                engines[c].play(color, coord)
            except EngineError as e:
                engines[c].kill()
                forfeit(c, str(e))
                return False
        return True

    for c in (BLACK, WHITE):
        try:
            engines[c].new_game()
        except EngineError as e:
            engines[c].kill()
            forfeit(c, str(e))
            break
    # 序盤 (定められた手順)
    if result['forfeit'] is None:
        for pos in opening:
            if board.legal() == 0:
                if not notify_all(board.turn, 'pass'):
                    break
                board.pass_turn()
            color = board.turn
            if not board.move(pos):
                raise ValueError('illegal opening move ' + pos_to_coord(pos) + ' after ' + record)
            record += pos_to_coord(pos)
            if not notify_all(color, pos_to_coord(pos)):
                break
    # AI 同士の対局
    while result['forfeit'] is None:
        if board.legal() == 0:
            if board.is_end():
                break
            if not notify_all(board.turn, 'pass'):
                break
            board.pass_turn()
            continue
        color = board.turn
        engine = engines[color]
        try:
            coord, elapsed = engine.genmove(board, color, clock[color], settings.timeout)
        except EngineError as e:
            engine.kill()  # 応答しないエンジンは作り直す
            forfeit(color, str(e))
            break
        times[color].append(elapsed)
        pos = coord_to_pos(coord)
        if pos is None or not board.move(pos):
            engine.kill()  # エンジン内部の局面が壊れている可能性があるので作り直す
            forfeit(color, 'illegal move "{}" after {}'.format(coord, record))
            break
        record += pos_to_coord(pos)
        # 時間の確認
        if settings.mode == 'gametime':
            clock[color] -= elapsed
            if clock[color] < -settings.grace and overtime[color] == 0:
                overtime[color] = 1
                if settings.forfeit:
                    forfeit(color, 'time over ({:.3f} sec over)'.format(-clock[color]))
                    break
        elif settings.mode == 'movetime':
            if elapsed > settings.time + settings.grace:
                overtime[color] += 1
                if settings.forfeit:
                    forfeit(color, 'time over ({:.3f} sec for a move)'.format(elapsed))
                    break
        if not notify_all(color, pos_to_coord(pos), skip=color):
            break

    if result['forfeit'] is not None:
        loser = BLACK if result['forfeit']['color'] == 'black' else WHITE
        result['black_disc_diff'] = -64 if loser == BLACK else 64
    else:
        result['black_disc_diff'] = board.black_disc_diff()
    result['record'] = record
    result['times'] = times
    result['overtime'] = overtime
    result['clock'] = clock
    return result


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self, players):
        n = len(players)
        self.players = players
        self.wdl = [[[0, 0, 0] for _ in range(n)] for _ in range(n)]  # 2局1組の勝ち, 引き分け, 負け
        self.disc = [[0.0 for _ in range(n)] for _ in range(n)]       # 1局あたりの石差の合計
        self.n_played = [[0 for _ in range(n)] for _ in range(n)]      # 2局1組の数
        self.game_wdl = [[0, 0, 0] for _ in range(n)]                  # 1局ごとの勝ち, 引き分け, 負け
        self.n_moves = [0] * n
        self.time_sum = [0.0] * n
        self.time_max = [0.0] * n
        self.n_games = [0] * n
        self.n_overtime = [0] * n
        self.n_forfeit = [0] * n
        self.forfeit_reasons = []
        self.lock = threading.Lock()

    def add_game(self, idx_black, idx_white, game):
        with self.lock:
            diff = game['black_disc_diff']
            for idx, sign in ((idx_black, 1), (idx_white, -1)):
                d = diff * sign
                self.game_wdl[idx][0 if d > 0 else (1 if d == 0 else 2)] += 1
            for idx, color in ((idx_black, BLACK), (idx_white, WHITE)):
                t = game['times'][color]
                self.n_moves[idx] += len(t)
                self.time_sum[idx] += sum(t)
                self.time_max[idx] = max([self.time_max[idx]] + t)
                self.n_games[idx] += 1
                self.n_overtime[idx] += game['overtime'][color]
            if game['forfeit'] is not None:
                loser = idx_black if game['forfeit']['color'] == 'black' else idx_white
                self.n_forfeit[loser] += 1
                name, reason = game['forfeit']['player'], game['forfeit']['reason']
                self.forfeit_reasons.append(reason if reason.startswith(name + ':') else name + ': ' + reason)

    def add_pair(self, p0, p1, sum_disc_diff_p0):
        with self.lock:
            if sum_disc_diff_p0 > 0:
                self.wdl[p0][p1][0] += 1
                self.wdl[p1][p0][2] += 1
            elif sum_disc_diff_p0 < 0:
                self.wdl[p0][p1][2] += 1
                self.wdl[p1][p0][0] += 1
            else:
                self.wdl[p0][p1][1] += 1
                self.wdl[p1][p0][1] += 1
            self.disc[p0][p1] += sum_disc_diff_p0 / 2
            self.disc[p1][p0] -= sum_disc_diff_p0 / 2
            self.n_played[p0][p1] += 1
            self.n_played[p1][p0] += 1

    def win_rate(self, i, j):
        w, d, l = self.wdl[i][j]
        return (w + d * 0.5) / max(1, w + d + l)

    def estimated_elo(self):
        names = [p.name for p in self.players]
        n = len(names)
        win_rates = np.full((n, n), np.nan, dtype=float)
        games = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j and self.n_played[i][j] > 0:
                    games[i, j] = float(self.n_played[i][j])
                    win_rates[i, j] = self.win_rate(i, j)
        try:
            ratings, intervals = fit_elo_from_winrates_with_interval(win_rates, games=games, names=names, confidence=0.95)
        except ValueError:
            return {}
        return {name: (float(ratings[name]), float(intervals[name])) for name in names}

    def pairwise_elo(self, i, j):
        """i から見た j とのレーティング差 (2者だけで推定)"""
        if self.n_played[i][j] == 0:
            return None
        win_rates = np.array([[np.nan, self.win_rate(i, j)], [np.nan, np.nan]])
        games = np.array([[0.0, float(self.n_played[i][j])], [float(self.n_played[i][j]), 0.0]])
        try:
            ratings, intervals = fit_elo_from_winrates_with_interval(win_rates, games=games, names=['i', 'j'], confidence=0.95)
        except ValueError:
            return None
        return float(ratings['i'] - ratings['j']), float(intervals['i'])

    def print_result(self, settings, gauntlet, out=sys.stdout):
        with self.lock:
            self._print_result(settings, gauntlet, out)

    def _print_result(self, settings, gauntlet, out):
        n = len(self.players)
        names = [p.name for p in self.players]
        estimated = self.estimated_elo()
        if gauntlet:
            last_col = 'elo_vs_' + names[0]
            last = {}
            for i in range(1, n):
                r = self.pairwise_elo(i, 0)
                if r is not None:
                    last[names[i]] = r
        else:
            last_col = 'e_rate95'
            last = estimated

        def last_str(name):
            if name not in last:
                return '-'
            est, ci = last[name]
            return '{:.1f}+-{:.1f}'.format(est, ci)

        print('mode', settings.mode, settings.description(), 'threads', settings.threads, file=out)
        print('Win Rate', file=out)
        print('\t'.join(['vs >'] + names + ['all', last_col]), file=out)
        for i in range(n):
            row = [names[i]]
            w = d = l = 0
            for j in range(n):
                if i == j or self.n_played[i][j] == 0:
                    row.append('-')
                else:
                    row.append('{:.4f}'.format(self.win_rate(i, j)))
                ww, dd, ll = self.wdl[i][j]
                w += ww
                d += dd
                l += ll
            row.append('{:.4f}'.format((w + d * 0.5) / max(1, w + d + l)))
            row.append(last_str(names[i]))
            print('\t'.join(row), file=out)

        print('Average Disc Difference', file=out)
        print('\t'.join(['vs >'] + names + ['all', last_col]), file=out)
        for i in range(n):
            row = [names[i]]
            for j in range(n):
                if i == j or self.n_played[i][j] == 0:
                    row.append('-')
                else:
                    row.append('{:+.2f}'.format(self.disc[i][j] / self.n_played[i][j]))
            row.append('{:+.2f}'.format(sum(self.disc[i]) / max(1, sum(self.n_played[i]))))
            row.append(last_str(names[i]))
            print('\t'.join(row), file=out)

        print('Games (per game W-D-L) and Time', file=out)
        header = ['name', 'games', 'W-D-L', 'moves', 'avg_sec/move', 'max_sec/move', 'avg_sec/game', 'overtime', 'forfeit']
        print('\t'.join(header), file=out)
        for i in range(n):
            w, d, l = self.game_wdl[i]
            row = [
                names[i],
                str(self.n_games[i]),
                '{}-{}-{}'.format(w, d, l),
                str(self.n_moves[i]),
                '{:.3f}'.format(self.time_sum[i] / max(1, self.n_moves[i])),
                '{:.3f}'.format(self.time_max[i]),
                '{:.2f}'.format(self.time_sum[i] / max(1, self.n_games[i])),
                str(self.n_overtime[i]),
                str(self.n_forfeit[i]),
            ]
            print('\t'.join(row), file=out)
        if self.forfeit_reasons:
            print('forfeits (latest 10):', file=out)
            for reason in self.forfeit_reasons[-10:]:
                print('  ' + reason, file=out)
        out.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def list_players():
    print('presets:')
    for name, conf in PRESETS.items():
        modes = [m for m in ('depth', 'movetime', 'gametime') if conf.get(m) is not None]
        print('  {:15s} protocol={:5s} modes={}'.format(name, conf['protocol'], ','.join(modes)))
    print('players:')
    for entry in PLAYERS:
        print('  {:12s} {:15s} {}'.format(entry[0], entry[1], entry[2]) + ('  ' + str(entry[3]) if len(entry) >= 4 else ''))


def parse_args():
    parser = argparse.ArgumentParser(description='Egaroucid vs other Othello engines (depth / movetime / gametime)')
    parser.add_argument('mode', nargs='?', choices=['depth', 'movetime', 'gametime'], help='depth: fixed level, movetime: fixed seconds per move, gametime: seconds per game')
    parser.add_argument('-l', '--level', type=int, help='level (depth mode)')
    parser.add_argument('--time', type=float, help='seconds per move (movetime) / seconds per game (gametime)')
    parser.add_argument('-n', '--rounds', type=int, default=10, help='number of rounds. each pairing plays 2 games (colors swapped) per round')
    parser.add_argument('-p', '--players', nargs='+', help='player names in PLAYERS (default: all)')
    parser.add_argument('--gauntlet', action='store_true', help='only the first player vs the others')
    parser.add_argument('-t', '--threads', type=int, default=1, help='threads per engine')
    parser.add_argument('-j', '--parallel', type=int, default=1, help='number of game pairs played simultaneously')
    parser.add_argument('--openings', default=DEFAULT_OPENINGS, help='opening file (one transcript per line)')
    parser.add_argument('--seed', type=int, default=57, help='random seed for shuffling openings')
    parser.add_argument('--forfeit', action='store_true', help='lose by time over (movetime / gametime)')
    parser.add_argument('--grace', type=float, default=0.1, help='allowed time over in seconds before counting as overtime')
    parser.add_argument('--timeout', type=float, help='hard timeout for genmove in seconds (0: none). an engine that does not answer loses the game')
    parser.add_argument('--record', help='save games to this file (JSON lines)')
    parser.add_argument('--list', action='store_true', help='show presets and players and exit')
    args = parser.parse_args()
    if args.list:
        return args
    if args.mode is None:
        parser.error('mode is required (depth / movetime / gametime)')
    if args.mode == 'depth' and args.level is None:
        parser.error('depth mode needs --level')
    if args.mode != 'depth' and (args.time is None or args.time <= 0):
        parser.error(args.mode + ' mode needs --time > 0')
    if args.rounds < 1 or args.threads < 1 or args.parallel < 1:
        parser.error('--rounds, --threads and --parallel must be positive')
    return args


def main():
    args = parse_args()
    if args.list:
        list_players()
        return
    settings = Settings(args)

    # players
    entries = PLAYERS
    if args.players:
        by_name = {entry[0]: entry for entry in PLAYERS}
        unknown = [name for name in args.players if name not in by_name]
        if unknown:
            print('unknown players:', ' '.join(unknown), '(see --list)')
            sys.exit(1)
        entries = [by_name[name] for name in args.players]
    players = [Player(i, entry) for i, entry in enumerate(entries)]
    if not args.players:  # 指定がなければ, 実行ファイルが見つからないプレイヤーは除く
        found = []
        for p in players:
            exe = p.executable()
            if os.path.isabs(exe) and not os.path.isfile(exe):
                print('[WARNING] skip {}: executable not found: {}'.format(p.name, exe))
            else:
                found.append(p)
        players = found
        for i, p in enumerate(players):
            p.idx = i
    if len(players) < 2:
        print('need at least 2 players (see --list)')
        sys.exit(1)
    errors = [e for e in (p.check(settings) for p in players) if e is not None]
    if errors:
        for e in errors:
            print('[ERROR]', e)
        sys.exit(1)

    # openings
    with open(args.openings, 'r') as f:
        openings = [line.strip() for line in f.read().splitlines() if line.strip() and not line.startswith('#')]
    rng = random.Random(args.seed)
    rng.shuffle(openings)
    openings = [(line, parse_opening(line)) for line in openings]

    # pairings
    if args.gauntlet:
        pairings = [(0, j) for j in range(1, len(players))]
    else:
        pairings = [(i, j) for i in range(len(players)) for j in range(i + 1, len(players))]
    tasks = []
    opening_idx = 0
    for r in range(args.rounds):
        round_pairings = pairings[:]
        rng.shuffle(round_pairings)
        for p0, p1 in round_pairings:
            tasks.append((r, p0, p1, opening_idx % len(openings)))
            opening_idx += 1

    print('mode', settings.mode, settings.description())
    print('players', len(players), 'rounds', args.rounds, 'pairs', len(tasks), 'games', len(tasks) * 2)
    print('threads per engine', args.threads, 'parallel', args.parallel, 'openings', args.openings, len(openings))
    if settings.mode != 'depth':
        print('forfeit on time' if settings.forfeit else 'no forfeit on time (overtime is only counted)', 'grace', settings.grace)
    for p in players:
        print(' ', p.name, '[' + p.preset_name + ']', ' '.join(p.argv(settings)), '(cwd ' + p.cwd + ')' if p.cwd else '')
    sys.stdout.flush()

    pools = [EnginePool(p, settings, args.parallel) for p in players]
    stats = Stats(players)
    record_file = open(args.record, 'a', encoding='utf-8') if args.record else None
    record_lock = threading.Lock()
    stop = threading.Event()

    def play_pair(round_idx, p0, p1, opening_idx):
        if stop.is_set():
            return None
        opening_str, opening = openings[opening_idx]
        e0 = e1 = None
        sum_disc_diff_p0 = 0
        try:
            e0 = pools[p0].acquire()
            e1 = pools[p1].acquire()
            for p0_color in (BLACK, WHITE):
                if stop.is_set():
                    return None
                engines = [e0, e1] if p0_color == BLACK else [e1, e0]
                idx = [p0, p1] if p0_color == BLACK else [p1, p0]
                game = play_game(engines, opening, settings)
                if stop.is_set():  # 中断でエンジンを止めた対局は数えない
                    return None
                diff = game['black_disc_diff'] if p0_color == BLACK else -game['black_disc_diff']
                sum_disc_diff_p0 += diff
                stats.add_game(idx[BLACK], idx[WHITE], game)
                if record_file is not None:
                    data = {
                        'datetime': datetime.datetime.now().isoformat(timespec='seconds'),
                        'mode': settings.mode, 'level': settings.level, 'time': settings.time,
                        'round': round_idx, 'opening': opening_str,
                        'black': players[idx[BLACK]].name, 'white': players[idx[WHITE]].name,
                        'record': game['record'], 'black_disc_diff': game['black_disc_diff'],
                        'forfeit': game['forfeit'],
                        'black_times': [round(t, 4) for t in game['times'][BLACK]],
                        'white_times': [round(t, 4) for t in game['times'][WHITE]],
                    }
                    with record_lock:
                        record_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                        record_file.flush()
                if p0_color == BLACK:
                    # 落ちた / 止めたエンジンは次の対局のために作り直す
                    if not e0.alive or e0.proc.poll() is not None:
                        pools[p0].discard(e0)
                        e0 = None
                        e0 = pools[p0].acquire()
                    if not e1.alive or e1.proc.poll() is not None:
                        pools[p1].discard(e1)
                        e1 = None
                        e1 = pools[p1].acquire()
        finally:
            if e0 is not None:
                pools[p0].release(e0)
            if e1 is not None:
                pools[p1].release(e1)
        stats.add_pair(p0, p1, sum_disc_diff_p0)
        return sum_disc_diff_p0

    n_done = 0
    error = None
    progress = None
    executor = ThreadPoolExecutor(max_workers=args.parallel)
    try:
        futures = {executor.submit(play_pair, *task): task for task in tasks}
        progress = tqdm(total=len(tasks), desc='pairs', dynamic_ncols=True) if tqdm is not None else None
        for future in as_completed(futures):
            _, p0, p1, _ = futures[future]
            try:
                if future.result() is not None:
                    n_done += 1
            except Exception as e:
                error = 'pair {} vs {} failed: {}: {}'.format(players[p0].name, players[p1].name, type(e).__name__, e)
                break
            if progress is not None:
                progress.update(1)
            if n_done % len(pairings) == 0 and n_done != len(tasks):
                if progress is not None:
                    progress.clear()
                print('{} / {} pairs done'.format(n_done, len(tasks)))
                stats.print_result(settings, args.gauntlet)
    except KeyboardInterrupt:
        error = 'interrupted'
    finally:
        stop.set()
        if progress is not None:
            progress.close()
        if error is not None:
            print('\n[ERROR] ' + error if error != 'interrupted' else '\ninterrupted')
            with ALL_PROCESSES_LOCK:  # 対局中のエンジンを止めて, 実行中のスレッドを終わらせる
                for engine in ALL_PROCESSES:
                    engine.kill()
        executor.shutdown(wait=True, cancel_futures=True)
        with ALL_PROCESSES_LOCK:
            for engine in ALL_PROCESSES:
                engine.close()
        if record_file is not None:
            record_file.close()

    print('{} pairs ({} games) played, {}'.format(n_done, n_done * 2, settings.description()))
    stats.print_result(settings, args.gauntlet)
    if error is not None and error != 'interrupted':
        sys.exit(1)


if __name__ == '__main__':
    main()
