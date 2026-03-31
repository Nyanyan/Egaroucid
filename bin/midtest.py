import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

level = 23
n_threads = 32
hash_level = 25
exe = 'Egaroucid_for_Console.exe'

try:
    if len(sys.argv) >= 2:
        level = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n_threads = int(sys.argv[2])
    if len(sys.argv) >= 4:
        hash_level = int(sys.argv[3])
    if len(sys.argv) >= 5:
        exe = sys.argv[5]
except:
    print('usage: python midtest.py [level=23] [n_threads=32] [hash_level=25] [exe=Egaroucid_for_Console.exe]')
    exit()


script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)

cmd_version = exe + ' -v'
# print(cmd_version)
version = subprocess.run((cmd_version).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)

cmd = exe + ' -l ' + str(level) + ' -nobook -thread ' + str(n_threads) + '-hash ' + str(hash_level) + ' -solve ' + os.path.join(script_dir, 'problem/midgame_test.txt')

print(cmd.replace(script_dir, 'script_dir'))
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line)
for i in range(32):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    line = '#' + fill0(i, 2) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()