# Endgame complete search with MPC (selective endgame search)
# Positions: GGS positions with 40 empties (problem/ggs_mpc_endgame_40_20260823.txt)
#
# Every position is solved to the end with the given MPC probability
# using -dpr (-depthprobrange), so the level setting is not used.
# prob: 74, 88, 93, 98, 99, 99.9, 100

import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

PROBS = ['74', '88', '93', '98', '99', '99.9', '100']

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
    'prob': '74',
    'n_threads': 42,
    'hash_level': 25,
    'exe': 'Egaroucid_for_Console.exe',
    'eval_file': '',
    'problem_file': 'problem/ggs_mpc_endgame_40_20260823.txt',
    'eval_mo_file': '',
    'eval_mo_aux_file': '',
    'eval_mo_shallow_end_only': 0,
    'eval_mo_shallow_depth': 0,
    'eval_mo_shallow_tt': 0,
    'seed': '',
}
try:
    parse_args(args, ['prob', 'n_threads', 'hash_level', 'exe', 'eval_file', 'problem_file', 'eval_mo_file', 'eval_mo_shallow_depth', 'eval_mo_shallow_tt', 'seed', 'eval_mo_aux_file', 'eval_mo_shallow_end_only'])
    if not (args['prob'] in PROBS):
        raise ValueError('prob must be one of ' + ', '.join(PROBS))
except Exception as e:
    print(e)
    print('usage: python mpcendtest.py [prob=74 (74, 88, 93, 98, 99, 99.9, 100)] [n_threads=42] [hash_level=25] [exe=Egaroucid_for_Console.exe] [eval_file=] [problem_file=problem/ggs_mpc_endgame_40_20260823.txt] [eval_mo_file=] [eval_mo_shallow_depth=0] [eval_mo_shallow_tt=0] [eval_mo_aux_file=] [eval_mo_shallow_end_only=0]')
    print('arguments can be given in this order or as name=value (e.g. exe=Egaroucid_for_Console_clang.exe)')
    sys.exit(2)
prob = args['prob']
n_threads = args['n_threads']
hash_level = args['hash_level']
exe = args['exe']
eval_file = args['eval_file']
problem_file = args['problem_file']


script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)
if not os.path.isabs(problem_file):
    problem_file = os.path.join(script_dir, problem_file)

with open(problem_file, 'r') as f:
    n_problems = len([line for line in f.read().splitlines() if line.strip() != ''])

cmd_version = [exe, '-v']
# print(cmd_version)
version = subprocess.run(cmd_version, stdin=subprocess.PIPE, stdout=subprocess.PIPE, check=True).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)

# depth 60 is clipped to the number of empties, so every move range is solved to the end
cmd = [exe, '-dpr', '1', '60', '60', prob, '-nobook', '-thread', str(n_threads), '-hash', str(hash_level)]
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
    print('#   ' + line, flush=True)
    for i in range(n_problems):
        line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
        columns = [elem.strip() for elem in line.split('|')]
        if len(columns) < 2 or columns[1] != 'custom':
            line += ' NOT DPR SEARCH'
        line = '#' + fill0(i, 2) + ' ' + line
        print(line, flush=True)
        res += line + '\n'
        if 'NOT DPR SEARCH' in line:
            raise RuntimeError('engine did not use the requested depth/probability')
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line, flush=True)
    if not line.startswith('total '):
        raise RuntimeError('missing engine total; benchmark incomplete')
finally:
    egaroucid.kill()
    egaroucid.wait()
