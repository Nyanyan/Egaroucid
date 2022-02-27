from cmath import sqrt
from tqdm import trange, tqdm
import math
import matplotlib.pyplot as plt

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

nums = [0 for _ in range(129)]

mu = 0
sigma = 24
N = 10000000

def f(x):
    return int(complex(N * 1 / (sqrt(2 * math.pi) * sigma) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))).real)
    #return 5000

max_nums = [f(i) for i in range(-64, 65)]
print(max_nums)
plt.plot(range(-64, 65), max_nums)
plt.show()

n_reg = 0
for file_idx in tqdm(list(reversed(range(435)))):
    with open('data/records4/' + digit(file_idx, 7) + '.txt', 'r') as f:
        data = f.read().splitlines()
    for datum in data:
        board, player, value = datum.split()
        n_moves = -4
        for i in range(64):
            n_moves += board[i] != '.'
        #if n_moves >= 20:
        value = int(value)
        score_idx = value + 64
        if nums[score_idx] < max_nums[score_idx]:
            with open('data/records5/' + digit(n_reg // 40000, 7) + '.txt', 'a') as f:
                f.write(datum + '\n')
            nums[score_idx] += 1
            n_reg += 1
print(nums)
plt.plot(range(-64, 65), nums)
plt.show()