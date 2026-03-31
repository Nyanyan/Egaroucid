import os
import subprocess
import sys


def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n


def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s


level = 23
n_threads = 32
hash_level = 25
exe = 'Egaroucid_for_Console.exe'
problem_file = 'problem/midgame_test.txt'

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
        problem_file = sys.argv[5]
except Exception:
    print('usage: python midtest.py [level=23] [n_threads=32] [hash_level=25] [exe=Egaroucid_for_Console.exe] [problem=problem/midgame_test.txt]')
    exit()

script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)
if not os.path.isabs(problem_file):
    problem_file = os.path.join(script_dir, problem_file)

cmd_version = [exe, '-v']
version = subprocess.run(cmd_version, stdin=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode()
version = strip_newlines(version)
print(version)

cmd = [
    exe,
    '-l',
    str(level),
    '-nobook',
    '-thread',
    str(n_threads),
    '-hash',
    str(hash_level),
    '-solve',
    problem_file,
]

print(' '.join(cmd).replace(script_dir, 'script_dir'))
egaroucid = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

line = strip_newlines(egaroucid.stdout.readline().decode())
print('#   ' + line)
i = 0
while True:
    line = strip_newlines(egaroucid.stdout.readline().decode())
    if line == '':
        break
    if line.startswith('total '):
        print(line)
        break
    print('#' + fill0(i, 2) + ' ' + line)
    i += 1
egaroucid.kill()
