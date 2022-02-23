from random import random
import subprocess
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

evaluate = subprocess.Popen('../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

maes = {}
mses = {}

for score in trange(-64, 65, 2):
    with open('eval_testcases/' + str(score + 64) + '.txt', 'r') as f:
        data = f.read().splitlines()
    error = []
    mae = 0
    mse = 0
    #print('score =', score)
    for datum in data:
        std_in = '0\n' + datum + '\n0\n0\n10000\n'
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
    plt.savefig('eval_testcases/' + str(score + 64) + '.png', format="png")
    plt.clf()
    mae /= len(data)
    mse /= len(data)
    maes[score] = mae
    mses[score] = mse

evaluate.kill()

with open('eval_testcases/summary.txt', 'w') as f:
    f.write('mae\n')
    for key in maes.keys():
        f.write(str(key) + ': ' + str(maes[key]) + '\n')
    f.write('mse\n')
    for key in mses.keys():
        f.write(str(key) + ': ' + str(mses[key]) + '\n')