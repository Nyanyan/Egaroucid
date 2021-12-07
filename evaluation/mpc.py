from random import randint, randrange
import subprocess
from tqdm import trange
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

evaluate = subprocess.Popen('../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
sleep(1)

min_depth = 3
max_depth = 10

vhs = [[[] for _ in range(max_depth - min_depth + 1)] for _ in range(12)]
vds = [[[] for _ in range(max_depth - min_depth + 1)] for _ in range(12)]

vh_vd = []

mpcd = [0,1,0,1,2,3,2,3,4,3,4,3,4,5,6,5,6,5,6,7,6,7,8,9,8,9,10,11,10,11]


def collect_data(num):
    global vhs, vds, vh_vd
    try:
        with open('data/records0/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        print('cannot open')
        return
    for _ in trange(1000):
        datum = data[randrange(0, len(data))]
        board, player, _, _, _, _ = datum.split()
        n_stones = calc_n_stones(board)
        depth = randint(min_depth, max_depth)
        board_proc = player + '\n' + str(mpcd[depth]) + '\n'
        for i in range(hw):
            for j in range(hw):
                board_proc += board[i * hw + j]
            board_proc += '\n'
        #print(board_proc)
        evaluate.stdin.write(board_proc.encode('utf-8'))
        evaluate.stdin.flush()
        vd = float(evaluate.stdout.readline().decode().strip())
        board_proc = player + '\n' + str(depth) + '\n'
        for i in range(hw):
            for j in range(hw):
                board_proc += board[i * hw + j]
            board_proc += '\n'
        #print(board_proc)
        evaluate.stdin.write(board_proc.encode('utf-8'))
        evaluate.stdin.flush()
        vh = float(evaluate.stdout.readline().decode().strip())
        #print(score)
        vhs[(n_stones - 4) // 5][depth - min_depth].append(vh)
        vds[(n_stones - 4) // 5][depth - min_depth].append(vd)

for i in range(10):
    collect_data(i)

start_temp = 1000.0
end_temp   = 10.0
def temperature_x(x):
    #return pow(start_temp, 1 - x) * pow(end_temp, x)
    return start_temp + (end_temp - start_temp) * x

def prob(p_score, n_score, strt, now, tl):
    dis = p_score - n_score
    if dis >= 0:
        return 1.0
    return exp(dis / temperature_x((now - strt) / tl))

a = 1.0
b = 0.0

def f(x):
    return a * x + b
'''
def scoring():
    dv = 0
    for i in range(6):
        dv += len(vhs[i])
    return sum([sum([(vhs[j][i] - f(vds[j][i])) ** 2 for i in range(len(vhs[j]))]) for j in range(1, 6)]) / dv

f_score = scoring()
print(f_score)

tl = 10.0
strt = time()
while time() - strt < tl:
    rnd = random()
    if rnd < 0.5:
        fa = a
        a += random() * 0.02 - 0.01
        score = scoring()
        if prob(f_score, score, strt, time(), tl) > random():
            f_score = score
            #print(f_score)
        else:
            a = fa
    else:
        fb = b
        b += random() * 0.02 - 0.01
        score = scoring()
        if prob(f_score, score, strt, time(), tl) > random():
            f_score = score
            #print(f_score)
        else:
            b = fb

print(f_score)
'''

vh_vd = [[[vhs[i][j][k] - f(vds[i][j][k]) for k in range(len(vhs[i][j]))] for j in range(len(vhs[i]))] for i in range(len(vhs))]
sd = [[round(statistics.stdev(vh_vd[i][j])) for j in range(len(vh_vd[i]))] for i in range(len(vh_vd))]
for each_sd in sd:
    print(str(each_sd).replace('[', '{').replace(']', '}') + ',')
evaluate.kill()
