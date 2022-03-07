import matplotlib.pyplot as plt
from random import randrange, random

min_depth = 2
max_depth = 20

n_phases = 15
n_scores = 17

def calc_phase(x):
    return (x - 4) // 4

def calc_score(x):
    return (x + 64) // 8

depth_width = max_depth - min_depth + 1

vhs = [[[] for _ in range(65)] for _ in range(65)]
vds = [[[] for _ in range(65)] for _ in range(65)]

sd = [[-1 for _ in range(65)] for _ in range(65)]

with open('sigma_calculation.txt', 'r') as f:
    for i in range(65):
        for j in range(65):
            sd[i][j] = float(f.readline())
            #if i <= 20:
            #    sd[i][j] = 1000000.0

plot_y = [0.0 for _ in range(65)]
nums = [0 for _ in range(65)]
for i in range(65):
    for j in range(65):
        if sd[i][j] != 1000000.0:
            plot_y[i] += sd[i][j]
            nums[i] += 1
for i in range(65):
    if nums[i]:
        plot_y[i] /= nums[i]
plt.title('n_stones')
plt.plot(range(65), plot_y)
plt.show()

plot_y = [0.0 for _ in range(65)]
nums = [0 for _ in range(65)]
for i in range(65):
    for j in range(65):
        if sd[i][j] != 1000000.0:
            plot_y[j] += sd[i][j]
            nums[j] += 1
for i in range(65):
    if nums[i]:
        plot_y[i] /= nums[i]
plt.title('depth')
plt.plot(range(65), plot_y)
plt.show()

params = [
    0.12506284069441298, 0.003146851320878552, -3.129294209685033, 0.01836594544092185, 0.42200995669259966, -1.828870503078114, 5.687890822032203
]

def f(n_stones, depth):
    x = params[0] * n_stones + params[1] * depth + params[2]
    return params[3] * x * x * x + params[4] * x * x + params[5] * x + params[6]

def scoring():
    res = 0
    for i in range(65):
        for j in range(65):
            ans = sd[i][j]
            if ans != 1000000.0:
                pred = f(i, j)
                res += (pred - ans) ** 2
    return res

score = scoring()
print(score)
if(input('continue?: ') != 'yes'):
    exit()
while True:
    idx = randrange(0, 7)
    f_param = params[idx]
    params[idx] += random() - 0.5
    n_score = scoring()
    if n_score < score:
        score = n_score
        print(score, params)
    else:
        params[idx] = f_param