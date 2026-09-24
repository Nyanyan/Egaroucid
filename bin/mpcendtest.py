# Endgame complete search with MPC (selective endgame search)
# Positions: GGS positions with 40 empties (problem/ggs_mpc_endgame_40_20260823.txt)
#
# The MPC probability is determined by the level and the number of empties.
# For 40 empties:
#   level 35-36: 40@88%
#   level 37-40: 40@93%
#   level 41-50: 40@98%
#   level 51-54: 40@99%
#   level 55-60: 40@100% (no MPC)

import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

level = 35
n_threads = 42
hash_level = 25
exe = 'Egaroucid_for_Console.exe'
eval_file = ''
problem_file = 'problem/ggs_mpc_endgame_40_20260823.txt'

try:
    if len(sys.argv) >= 2:
        level = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n_threads = int(sys.argv[2])
    if len(sys.argv) >= 4:
        hash_level = int(sys.argv[3])
    if len(sys.argv) >= 5:
        exe = sys.argv[4]
    if len(sys.argv) >= 6:
        eval_file = sys.argv[5]
    if len(sys.argv) >= 7:
        problem_file = sys.argv[6]
except:
    print('usage: python mpcendtest.py [level=35] [n_threads=42] [hash_level=25] [exe=Egaroucid_for_Console.exe] [eval_file=] [problem_file=problem/ggs_mpc_endgame_40_20260823.txt]')
    exit()


script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)
if not os.path.isabs(problem_file):
    problem_file = os.path.join(script_dir, problem_file)

with open(problem_file, 'r') as f:
    n_problems = len([line for line in f.read().splitlines() if line.strip() != ''])

cmd_version = exe + ' -v'
# print(cmd_version)
version = subprocess.run((cmd_version).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)

cmd = exe + ' -l ' + str(level) + ' -nobook -thread ' + str(n_threads) + ' -hash ' + str(hash_level)
if eval_file != '':
    cmd += ' -eval ' + eval_file
cmd += ' -solve ' + problem_file

print(cmd.replace(script_dir, 'script_dir'))
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line, flush=True)
for i in range(n_problems):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    line = '#' + fill0(i, 2) + ' ' + line
    print(line, flush=True)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line, flush=True)
egaroucid.kill()
