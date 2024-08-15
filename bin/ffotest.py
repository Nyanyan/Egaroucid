import subprocess
import sys

answer = '''#40 38  a2
#41 0   h4
#42 6   g2
#43 -12 c7  g3
#44 -14 d2  b8
#45 6   b2
#46 -8  b3
#47 4   g2
#48 28  f6
#49 16  e1
#50 10  d8
#51 6   e2  a3
#52 0   a3
#53 -2  d8
#54 -2  c7
#55 0   g6  g4  b7  e2
#56 2   h5
#57 -10 a6
#58 4   g1
#59 64  g8  h4  e8'''

answer = answer.splitlines()

cmd = 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe -l 60 -hash 25 -nobook -solve problem/ffo40-59.txt -thread 40'
if len(sys.argv) == 2:
    eval_file = sys.argv[1]
    print('eval ', eval_file)
    cmd += ' -eval ' + eval_file
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
    line = '#' + str(40 + i) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()