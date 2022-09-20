import subprocess
from time import time
import sys

strt_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])

if len(sys.argv) >= 4:
    n_threads = int(sys.argv[3])
else:
    n_threads = 23

if n_threads >= 2:
    egaroucid = subprocess.Popen(('Egaroucid6_test.exe ' + str(n_threads - 1)).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
else:
    egaroucid = subprocess.Popen('Egaroucid6_test_single.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res_str = ''
tim = 0
nodes = 0
strt = time()
for i in range(strt_idx, end_idx):
    print('#', i)
    with open('./../../benchmark/ffotest/' + str(i) + '.txt', 'r') as f:
        s = f.read()
    egaroucid.stdin.write(s.encode('utf-8'))
    egaroucid.stdin.flush()
    result = egaroucid.stdout.readline().decode()
    res_str += '#' + str(i) + ' ' + result
    tim += int(result.split()[9])
    nodes += int(result.split()[7])
egaroucid.kill()

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

res_str_proc = ''
for line, ans_line in zip(res_str.splitlines(), answer.splitlines()):
    ans_score = ans_line.split()[1]
    ans_policies = ans_line.split()[2:]
    score = line.split()[4]
    policy = line.split()[6]
    res_str_proc += line
    if ans_score != score:
        res_str_proc += ' WRONG_SCORE'
    if not (policy in ans_policies):
        res_str_proc += ' WRONG_POLICY'
    res_str_proc += '\n'

print('done')
print(res_str_proc, end='')
print(n_threads, 'threads')
print(tim / 1000, 'sec')
print(time() - strt, 'sec total')
print(nodes, 'nodes')
print(nodes / tim * 1000, 'nps')