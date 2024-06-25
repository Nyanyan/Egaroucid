import subprocess
import sys

answer = '''#60 20  c2 
#61 -14 h3 g1
#62 28  e8
#63 -2  f2
#64 20  b4
#65 10  g1
#66 30  h3
#67 22  h3
#68 28  e8 
#69 0   h3
#70 -24 e3
#71 20  d2
#72 24  e1
#73 -4  g4
#74 -30 f1
#75 14  d2
#76 32  a3
#77 34  b7
#78 8  f1
#79 64 d7'''

answer = answer.splitlines()

cmd = 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -l 60 -hash 25 -nobook -solve problem/ffo60-79.txt -thread 42'

print(cmd)
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line)
for i in range(20):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    policies = answer[i].split()[2:]
    if not (policy in policies):
        line += ' WRONG POLICY'
    score = line.split()[4][:-1]
    correct_score = answer[i].split()[1]
    if int(score) != int(correct_score):
        line += ' WRONG SCORE'
    line = '#' + str(60 + i) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()