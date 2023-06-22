import subprocess
import sys

cmd = 'Egaroucid_for_Console.exe -l 23 -nobook -thread 19 -hash 25 -solve problem/problem_10_discs_20.txt'

print(cmd)
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line)
for i in range(20):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    line = '#' + str(10 + i) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()