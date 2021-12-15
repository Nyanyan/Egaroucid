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

min_depth = 9
max_depth = 28

vhs = [[] for _ in range(max_depth - min_depth + 1)]
vds = [[] for _ in range(max_depth - min_depth + 1)]

vh_vd = []

mpcd = [0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9]


def collect_data(num):
    global vhs, vds, vh_vd
    try:
        with open('data/records3/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        print('cannot open')
        return
    for data_idx in trange(len(data)):
        datum = data[data_idx]
        board, player, _, _, _, vh = datum.split()
        #board, player, _, _, _, vh = datum.split()
        n_stones = calc_n_stones(board)
        depth = 64 - n_stones
        if min_depth <= depth <= max_depth:
            board_proc = player + '\n' + str(mpcd[depth]) + '\n'
            for i in range(hw):
                for j in range(hw):
                    board_proc += board[i * hw + j]
                board_proc += '\n'
            #print(board_proc)
            evaluate.stdin.write(board_proc.encode('utf-8'))
            evaluate.stdin.flush()
            vd = float(evaluate.stdout.readline().decode().strip())
            vh = float(vh)
            if player == '1':
                vh = -vh
            vh *= 100
            #print(score)
            vhs[depth - min_depth].append(vh)
            vds[depth - min_depth].append(vd)

for i in range(3):
    collect_data(i)
evaluate.kill()

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

n_all_data = 0
for i in range(len(vhs)):
    n_all_data += len(vhs[i])
print(n_all_data)

def f(x):
    return a * x + b

def scoring():
    score = 0
    for i in range(len(vhs)):
        for j in range(len(vhs[i])):
            score += 1 / n_all_data * ((vhs[i][j] - f(vds[i][j])) ** 2)
    return score

f_score = scoring()
print(f_score)

vh_vd = [[vhs[j][k] - f(vds[j][k]) for k in range(len(vhs[j]))] for j in range(len(vhs))]
sd = [round(statistics.stdev(vh_vd[j])) for j in range(len(vh_vd))]
print(sd)

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
            #print(a, f_score)
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
print(a, b)


vh_vd = [[vhs[j][k] - f(vds[j][k]) for k in range(len(vhs[j]))] for j in range(len(vhs))]
sd = [round(statistics.stdev(vh_vd[j])) for j in range(len(vh_vd))]
print(sd)
