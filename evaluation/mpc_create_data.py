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

min_depth = 0
max_depth = 15

depth_width = max_depth - min_depth + 1

def calc_stones(board):
    res = 0
    for i in board:
        if i != '.':
            res += 1
    return res

def collect_data(directory, num):
    global vhs, vds, vh_vd
    try:
        with open('data/' + directory + '/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        print('cannot open')
        return
    #for _ in trange(1000):
    depth = min_depth
    max_num = 5000
    print(len(data))
    for _ in tqdm(range(max_num)):
        datum = data[randrange(0, len(data))]
        #datum = data[randrange(0, len(data))]
        board, player, score = datum.split()
        score = float(score)
        if player == '1':
            score = -score
        if abs(score) == 64.0 and random() > 0.15:
            continue
        n_stones = calc_n_stones(board)
        if n_stones >= 60:
            continue
        depth1 = randint(min_depth, max_depth)
        depth1 = min(depth1, 64 - n_stones)
        board_proc = player + '\n'
        for i in range(hw):
            for j in range(hw):
                board_proc += board[i * hw + j]
            board_proc += '\n'
        board_proc += str(depth1) + '\n0\n1.3\n'
        #print(board_proc)
        if depth1 == 64 - n_stones:
            v1 = score
        else:
            evaluate.stdin.write(board_proc.encode('utf-8'))
            evaluate.stdin.flush()
            v1 = float(evaluate.stdout.readline().decode().strip())
        depth2 = depth1
        while depth1 == depth2 or depth1 % 2 != depth2 % 2:
            depth2 = randint(min_depth, min(max_depth, 64 - n_stones))
            depth2 = min(depth2, 64 - n_stones)
        board_proc = player + '\n'
        for i in range(hw):
            for j in range(hw):
                board_proc += board[i * hw + j]
            board_proc += '\n'
        board_proc += str(depth2) + '\n0\n1.3\n'
        #print(board_proc)
        if depth2 == 64 - n_stones:
            v2 = score
        else:
            evaluate.stdin.write(board_proc.encode('utf-8'))
            evaluate.stdin.flush()
            v2 = float(evaluate.stdout.readline().decode().strip())
        with open('sigma_data.txt', 'a') as f:
            f.write(str(n_stones) + ' ' + str(depth1) + ' ' + str(v1) + ' ' + str(depth2) + ' ' + str(v2) + '\n')

for i in range(20, 173):
    collect_data('records3', i)
    collect_data('records9', i)
evaluate.kill()
