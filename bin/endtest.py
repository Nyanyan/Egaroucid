import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

N_PROBLEMS = 10

exe = 'Egaroucid_for_Console.exe'
#exe = 'versions/Egaroucid_for_Console_7_5_1_Windows_SIMD/Egaroucid_for_Console_7_5_1_SIMD.exe'


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


cmd = exe + ' -l 27 -nobook -thread 32 -solve ' + os.path.join(script_dir, 'problem/endgame_test_' + str(N_PROBLEMS) + '.txt')

# print(cmd)
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line)
for i in range(N_PROBLEMS):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    line = '#' + fill0(i, 2) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()