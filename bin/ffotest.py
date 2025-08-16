# The FFO endgame test suite
# http://radagast.se/othello/ffotest.html
# https://github.com/abulmo/edax-reversi/tree/master/problem

import subprocess
import sys

start = 40
end = 59
n_threads = 42
hash_level = 25
#exe = 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe'
exe = 'Egaroucid_for_Console.exe'
try:
    if len(sys.argv) >= 2:
        start = int(sys.argv[1])
    if len(sys.argv) >= 3:
        end = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n_threads = int(sys.argv[3])
    if len(sys.argv) >= 5:
        hash_level = int(sys.argv[4])
    if len(sys.argv) >= 6:
        exe = sys.argv[5]
except:
    print('usage: python ffotest.py [start=40] [end=59] [n_threads=42] [hash_level=25] [exe=Egaroucid_for_Console.exe]')
    exit()



cmd_version = exe + ' -v'
print(cmd_version)
version = subprocess.run((cmd_version).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)


cmd = exe + ' -l 60 -hash ' + str(hash_level) + ' -nobook -solve problem/ffo' + str(start) + '-' + str(end) + '.txt -thread ' + str(n_threads)

print(cmd)

egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

answers = [ # best score, [best move, best move...]
    [], 
    [18, ['g8']], #1
    [10, ['a4']], #2
    [2, ['d1']], #3
    [0, ['h8', 'a5']], #4
    [32, ['g8']], #5
    [14, ['a1', 'h3']], #6
    [8, ['a6']], #7
    [8, ['e1']], #8
    [-8, ['g7', 'a4']], #9
    [10, ['b2']], #10
    [30, ['b3']], #11
    [-8, ['b7']], #12
    [14, ['b7']], #13
    [18, ['a3']], #14
    [4, ['g3', 'b8']], #15
    [24, ['f8']], #16
    [8, ['f8']], #17
    [-2, ['g2']], #18
    [8, ['b6']], #19
    [6, ['h5']], #20
    [0, ['g5']], #21
    [2, ['g8']], #22
    [4, ['a2']], #23
    [0, ['c3']], #24
    [0, ['g1', 'a5']], #25
    [0, ['d8']], #26
    [-2, ['b7']], #27
    [0, ['f1', 'b2', 'e1']], #28
    [10, ['g2']], #29
    [0, ['g3']], #30
    [-2, ['g6']], #31
    [-4, ['g3']], #32
    [-8, ['e7', 'a3']], #33
    [-2, ['c2']], #34
    [0, ['c7']], #35
    [0, ['b7']], #36
    [-20, ['g2']], #37
    [4, ['b2']], #38
    [64, ['a8', 'b1', 'g1', 'g5', 'g6', 'c8', 'h3', 'e8', 'h4']], #39
    [38, ['a2']], #40
    [0, ['h4']], #41
    [6, ['g2']], #42
    [-12, ['g3', 'c7']], #43
    [-14, ['d2', 'b8']], #44
    [6, ['b2']], #45
    [-8, ['b3']], #46
    [4, ['g2']], #47
    [28, ['f6']], #48
    [16, ['e1']], #49
    [10, ['d8']], #50
    [6, ['e2', 'a3']], #51
    [0, ['a3']], #52
    [-2, ['d8']], #53
    [-2, ['c7']], #54
    [0, ['g6', 'b7', 'e2', 'g4']], #55
    [2, ['h5']], #56
    [-10, ['a6']], #57
    [4, ['g1']], #58
    [64, ['h4', 'g8', 'e8']], #59
    [20, ['c2']], #60
    [-14, ['h3', 'g1']], #61
    [28, ['e8']], #62
    [-2, ['f2']], #63
    [20, ['b4']], #64
    [10, ['g1']], #65
    [30, ['h3']], #66
    [22, ['h3']], #67
    [28, ['e8']], #68
    [0, ['h3']], #69
    [-24, ['e3']], #70
    [20, ['d2']], #71
    [24, ['e1']], #72
    [-4, ['g4']], #73
    [-30, ['f1']], #74
    [14, ['d2']], #75
    [32, ['a3']], #76
    [34, ['b7']], #77
    [8, ['f1']], #78
    [64, ['d7']], #79
]

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line, flush=True)
for i in range(start, end + 1):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    correct_policies = answers[i][1]
    if not (policy in correct_policies):
        line += ' WRONG POLICY'
    score = line.split()[4][:-1]
    correct_score = answers[i][0]
    if int(score) != int(correct_score):
        line += ' WRONG SCORE'
    line = '#' + str(i) + ' ' + line
    print(line, flush=True)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line, flush=True)
egaroucid.kill()
