import subprocess
from random import randrange, randint
from time import time

param = [900000000, 800000000, 700000000, 1000000, 10000, 70, 3, 10, 20, 16, 61, 1, 6, 4, 1, 5, 29, 10, 5, 13]

names = [
    'W_BEST1_MOVE',
    'W_BEST2_MOVE',
    'W_BEST3_MOVE',
    'W_CACHE_HIT',
    'W_CACHE_HIGH',
    'W_CACHE_VALUE',
    'W_CELL_WEIGHT',
    'W_EVALUATE1',
    'W_EVALUATE2',
    'EVALUATE_SWITCH_DEPTH',
    'W_MOBILITY',
    'W_STABILITY',
    'W_SURROUND',
    'W_PARITY',
    'W_END_CELL_WEIGHT',
    'W_END_EVALUATE',
    'W_END_MOBILITY',
    'W_END_SURROUND',
    'W_END_STABILITY',
    'W_END_PARITY'
]

def print_params():
    for i in range(len(param)):
        print('#define ' + names[i] + ' ' + str(param[i]))
    print(param)

print_params()

egaroucid = subprocess.Popen('egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

test_strt = 40
test_end = 44

def test():
    param_in = '\n'.join([str(elem) for elem in param]) + '\n'
    res = []
    for i in range(test_strt, test_end + 1):
        with open('testcases/' + str(i) + '.txt', 'r') as f:
            board = f.read()
        #print(param_in + board)
        strt = time()
        egaroucid.stdin.write((param_in + board).encode('utf-8'))
        egaroucid.stdin.flush()
        n = int(egaroucid.stdout.readline().decode())
        print(i, n, time() - strt)
        res.append(n)
    return res

first_n_nodes = test()
print(first_n_nodes)

def scoring(n_nodes):
    res = 0.0
    for i in range(len(first_n_nodes)):
        res += n_nodes[i] / first_n_nodes[i]
        print(n_nodes[i], first_n_nodes[i], n_nodes[i] / first_n_nodes[i])
    return res / len(first_n_nodes)

score = 1.0

for num in range(20):
    idx = randrange(14, len(param))
    f_param = param[idx]
    pls = 0
    while pls == 0:
        pls = randint(-3, 3)
    param[idx] += pls
    n_nodes = test()
    print(n_nodes)
    n_score = scoring(n_nodes)
    print(num, n_score)
    if n_score < 1.0:
        first_n_nodes = [elem for elem in n_nodes]
        print_params()
    else:
        param[idx] = f_param

print_params()

egaroucid.kill()