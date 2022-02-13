import subprocess
from random import randrange, randint
from time import time
from tqdm import trange

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

N_MID_WEIGHT = 10

param = [
    [70 for _ in range(N_MID_WEIGHT)],
    [1 for _ in range(N_MID_WEIGHT)],
    [20 for _ in range(N_MID_WEIGHT)],
    [61 for _ in range(N_MID_WEIGHT)],
    [7 for _ in range(N_MID_WEIGHT)],
    [4 for _ in range(N_MID_WEIGHT)]
]

names = [
    'W_CACHE_VALUE',
    'W_CELL_WEIGHT',
    'W_EVALUATE',
    'W_MOBILITY',
    'W_SURROUND',
    'W_PARITY'
]

def print_params():
    for i in range(len(param)):
        print(names[i] + '[N_MID_WEIGHT] = {', end='')
        for j in range(len(param[i]) - 1):
            print(str(param[i][j]) + ', ', end='')
        print(str(param[i][len(param[i]) - 1]) + '};')
    for p in param:
        print(p, ',')

print_params()

egaroucid = subprocess.Popen('egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

test_strt = 0
test_end = 19

def test():
    param_in = ''
    for p in param:
        for elem in p:
            param_in += str(elem) + '\n'
    res = []
    for i in trange(test_strt, test_end + 1):
        with open('testcases/20/' + digit(i, 7) + '.txt', 'r') as f:
            board = f.read()
        #print(param_in + board)
        strt = time()
        egaroucid.stdin.write((param_in + board).encode('utf-8'))
        egaroucid.stdin.flush()
        n = int(egaroucid.stdout.readline().decode())
        #print(i, n, time() - strt)
        res.append(n)
    return res

first_n_nodes = test()
#print(first_n_nodes)
print('avg nodes', sum(first_n_nodes) / len(first_n_nodes))

def scoring(n_nodes):
    '''
    res = 0.0
    for i in range(len(first_n_nodes)):
        res += n_nodes[i] / first_n_nodes[i]
        #print(n_nodes[i], first_n_nodes[i], n_nodes[i] / first_n_nodes[i])
    return res / len(first_n_nodes)
    '''
    return sum(n_nodes) / sum(first_n_nodes)

score = 1.0

for num in range(20):
    idx1 = randrange(0, len(param))
    idx2 = randrange(0, N_MID_WEIGHT)
    f_param = param[idx1][idx2]
    pls = 0
    while pls == 0:
        pls = randint(-20, 20)
    param[idx1][idx2] += pls
    n_nodes = test()
    #print(n_nodes)
    print('avg nodes', sum(n_nodes) / len(n_nodes))
    n_score = scoring(n_nodes)
    print(num, n_score)
    if n_score < 1.0:
        first_n_nodes = [elem for elem in n_nodes]
        print_params()
    else:
        param[idx1][idx2] = f_param

print_params()

egaroucid.kill()