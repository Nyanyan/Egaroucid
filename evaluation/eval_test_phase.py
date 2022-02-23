from random import random
import subprocess
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

evaluate = subprocess.Popen('../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

maes = []
mses = []

for phase in trange(15):
    with open('eval_testcases/' + str(phase) + '.txt', 'r') as f:
        data = f.read().splitlines()
    error = []
    mae = 0
    mse = 0
    #print('score =', score)
    for datum in data:
        board = datum[:64]
        score = int(datum[64:]) - 64
        std_in = '0\n' + board + '\n0\n0\n10000\n'
        evaluate.stdin.write(std_in.encode('utf-8'))
        evaluate.stdin.flush()
        received_score = -int(evaluate.stdout.readline())
        error.append(received_score - score)
        mae += abs(received_score - score)
        mse += (received_score - score) ** 2
    plt.title('evaluation error score=' + str(score))
    plt.hist(error)
    plt.xlim(right=32)
    plt.xlim(left=-32)
    plt.savefig('eval_testcases/' + str(phase) + '.png', format="png")
    plt.clf()
    mae /= len(data)
    mse /= len(data)
    maes.append(mae)
    mses.append(mse)

evaluate.kill()

with open('eval_testcases/summary.txt', 'w') as f:
    f.write('mae\n')
    for phase in range(15):
        f.write(str(phase) + ': ' + str(maes[phase]) + '\n')
    f.write('mse\n')
    for phase in range(15):
        f.write(str(phase) + ': ' + str(mses[phase]) + '\n')