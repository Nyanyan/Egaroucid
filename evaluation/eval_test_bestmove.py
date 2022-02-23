from random import sample
import subprocess
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

evaluate = subprocess.Popen('../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


with open('eval_test_data_bestmove.txt', 'r') as f:
    all_data = f.read().splitlines()

n_data = 10000

data = sample(all_data, n_data)

n_error = 0
weighted_n_error = 0

for datum in tqdm(data):
    board = datum[:64]
    best_move = int(datum[64:])
    std_in = board + '\n'
    evaluate.stdin.write(std_in.encode('utf-8'))
    evaluate.stdin.flush()
    received_move = int(evaluate.stdout.readline())
    if best_move != received_move:
        n_error += 1
        weighted_n_error += 1

print(n_error, weighted_n_error, ' ', n_error / n_data, weighted_n_error / n_data)
