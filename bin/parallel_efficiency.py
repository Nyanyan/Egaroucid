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
#55 0   g6  g4  b7
#56 2   h5
#57 -10 a6
#58 4   g1
#59 64  g8  h4  e8
#60 20  c2 
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

data = []

for n_thread in range(1, 25):
    cmd = 'Egaroucid_for_Console.exe -l 60 -nobook -hash 25 -solve problem/ffo40-49.txt -thread ' + str(n_thread)
    print(cmd)
    egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    res = ''
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print('#   ' + line)
    for i in range(10):
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
    n_nodes = int(line.split()[1])
    tim = float(line.split()[4][:-1])
    nps = int(line.split()[6])
    print(n_nodes, tim, nps)
    data.append([n_thread, n_nodes, tim, nps])
    egaroucid.kill()

for datum in data:
    print(*datum)