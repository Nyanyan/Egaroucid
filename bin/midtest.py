import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def parse_args(args, names):
    # positional (in the order of names) or name=value
    n_positional = 0
    for arg in sys.argv[1:]:
        name, sep, value = arg.partition('=')
        if sep and name.isidentifier():
            if not (name in names):
                raise ValueError('unknown option ' + name)
        else:
            if n_positional >= len(names):
                raise ValueError('too many arguments')
            name = names[n_positional]
            value = arg
            n_positional += 1
        args[name] = type(args[name])(value)

args = {
    'level': 23,
    'n_threads': 32,
    'hash_level': 25,
    'exe': 'Egaroucid_for_Console.exe',
    'eval_file': '',
    'eval_mo_file': '',
    'eval_mo_shallow_depth': 0,
    'eval_mo_shallow_tt': 0,
    'seed': '',
    'depth': 0,
    'prob': '74',
    'problem_file': '',
    'eval_mo_aux_file': '',
    'eval_mo_shallow_end_only': 0,
}
try:
    parse_args(args, ['level', 'n_threads', 'hash_level', 'exe', 'eval_file', 'eval_mo_file', 'eval_mo_shallow_depth', 'eval_mo_shallow_tt', 'seed', 'depth', 'prob', 'problem_file', 'eval_mo_aux_file', 'eval_mo_shallow_end_only'])
    if not 0 <= args['depth'] <= 60:
        raise ValueError('depth must be 0 (use level) or 1..60')
    if args['prob'] not in ('74', '88', '93', '98', '99', '99.9', '100'):
        raise ValueError('invalid MPC probability')
except Exception as e:
    print(e)
    print('usage: python midtest.py [level=23] [n_threads=32] [hash_level=25] [exe=Egaroucid_for_Console.exe] [eval_file=] [eval_mo_file=] [eval_mo_shallow_depth=0] [eval_mo_shallow_tt=0] [seed=] [depth=0] [prob=74] [problem_file=] [eval_mo_aux_file=] [eval_mo_shallow_end_only=0]')
    print('arguments can be given in this order or as name=value (e.g. exe=Egaroucid_for_Console_clang.exe)')
    sys.exit(2)
level = args['level']
n_threads = args['n_threads']
hash_level = args['hash_level']
exe = args['exe']
eval_file = args['eval_file']


script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)

cmd_version = [exe, '-v']
# print(cmd_version)
version = subprocess.run(cmd_version, stdin=subprocess.PIPE, stdout=subprocess.PIPE, check=True).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)

problem_file = args['problem_file'] or os.path.join(script_dir, 'problem/midgame_test.txt')
if not os.path.isabs(problem_file):
    problem_file = os.path.join(script_dir, problem_file)
with open(problem_file, encoding='utf-8') as input_file:
    positions = [line for line in input_file.read().splitlines() if line.strip()]
if not positions:
    raise ValueError('problem file contains no positions')
cmd = [exe, '-nobook', '-thread', str(n_threads), '-hash', str(hash_level)]
if args['depth']:
    cmd += ['-dpr', '1', '60', str(args['depth']), args['prob']]
else:
    cmd += ['-l', str(level)]
if args['seed']:
    cmd += ['-seed', args['seed']]
if eval_file != '':
    cmd += ['-eval', eval_file]
if args['eval_mo_file']:
    cmd += ['-evalmo', args['eval_mo_file']]
if args['eval_mo_shallow_depth']:
    cmd += ['-evalmo-shallow-depth', str(args['eval_mo_shallow_depth'])]
if args['eval_mo_shallow_tt']:
    cmd += ['-evalmo-shallow-tt']
if args['eval_mo_aux_file']:
    cmd += ['-evalmo-aux', args['eval_mo_aux_file']]
if args['eval_mo_shallow_end_only']:
    cmd += ['-evalmo-shallow-end-only']
cmd += ['-solve', problem_file]

print(subprocess.list2cmdline(cmd).replace(script_dir, 'script_dir'), flush=True)
egaroucid = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

try:
    res = ''
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print('#   ' + line)
    for i, position in enumerate(positions):
        line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
        columns = [part.strip() for part in line.split('|')]
        if len(columns) < 9:
            raise RuntimeError('incomplete engine row: ' + line)
        if args['depth']:
            empties = position.split()[0].count('-') + position.split()[0].count('.')
            expected_depth = str(min(args['depth'], empties)) + '@' + args['prob'] + '%'
            if columns[1] != 'custom' or columns[2] != expected_depth:
                raise RuntimeError('unexpected fixed depth/probability: ' + line)
        line = '#' + fill0(i, 2) + ' ' + line
        print(line, flush=True)
        res += line + '\n'
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line, flush=True)
    if not line.startswith('total '):
        raise RuntimeError('missing engine total; benchmark incomplete')
finally:
    egaroucid.kill()
    egaroucid.wait()
