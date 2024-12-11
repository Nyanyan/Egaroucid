import subprocess
import sys

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

cmd = 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -l 23 -nobook -thread 32 -solve problem/midgame_test.txt'

print(cmd)
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