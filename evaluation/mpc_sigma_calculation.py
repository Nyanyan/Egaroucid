from random import randint, randrange
import subprocess
from tqdm import trange, tqdm
from time import sleep, time
from math import exp, tanh
from random import random
import statistics

inf = 10000000.0

hw = 8
min_n_stones = 4 + 10

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def calc_n_stones(board):
    res = 0
    for elem in board:
        res += int(elem != '.')
    return res

evaluate = subprocess.Popen('../new_src/test/a.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
sleep(1)

min_depth = 2
max_depth = 20

n_phases = 15
n_scores = 17

depth_width = max_depth - min_depth + 1

vhs = [[[] for _ in range(65)] for _ in range(65)]
vds = [[[] for _ in range(65)] for _ in range(65)]

vh_vd = []

mpcd = [
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 
    6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 
    10, 9, 10, 11, 12, 11, 12, 13, 14, 13, 
    14
]

def calc_stones(board):
    res = 0
    for i in board:
        if i != '.':
            res += 1
    return res

def collect_data(num):
    global vhs, vds, vh_vd
    try:
        with open('data/records3/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        print('cannot open')
        return
    #for _ in trange(1000):
    depth = min_depth
    max_num = 10000
    for tt, datum in enumerate(tqdm(data[:max_num])):
        #datum = data[randrange(0, len(data))]
        board, player, score = datum.split()
        player = int(player)
        score = int(score)
        if player == 1:
            score = -score
        n_stones = calc_n_stones(board)
        depth = tt * depth_width // max_num + min_depth #(depth - min_depth + 1) % depth_width + min_depth
        if depth >= 64 - calc_stones(board):
            continue
        board_proc = str(player) + '\n'
        for i in range(hw):
            for j in range(hw):
                board_proc += board[i * hw + j]
            board_proc += '\n'
        '''
        board_proc1 = board_proc + '0\n0\n1.18\n'
        evaluate.stdin.write(board_proc1.encode('utf-8'))
        evaluate.stdin.flush()
        v0 = int(evaluate.stdout.readline().decode().strip())
        '''
        board_proc2 = board_proc + str(mpcd[depth]) + '\n0\n1.18\n'
        #print(board_proc)
        evaluate.stdin.write(board_proc2.encode('utf-8'))
        evaluate.stdin.flush()
        vd = float(evaluate.stdout.readline().decode().strip())
        vh = float(score)
        #print(score)
        vhs[n_stones][depth].append(vh)
        vds[n_stones][depth].append(vd)

for i in range(20, 23):
    collect_data(i)
evaluate.kill()

vh_vd = [[[vhs[i][j][k] - vds[i][j][k] for k in range(len(vhs[i][j]))] for j in range(len(vhs[i]))] for i in range(len(vhs))]
sd = []
for i in range(len(vh_vd)):
    sd.append([])
    for j in range(len(vh_vd[i])):
        if len(vh_vd[i][j]) > 1:
            sd[i].append(round(statistics.stdev(vh_vd[i][j]), 4))
        else:
            sd[i].append(1000000.0)

with open('sigma_calculation.txt', 'w') as f:
    for a in sd:
        for b in a:
            f.write(str(b) + '\n')
