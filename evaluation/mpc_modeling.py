import matplotlib.pyplot as plt
from random import randrange, random

vhs = [[[] for _ in range(65)] for _ in range(65)]
vds = [[[] for _ in range(65)] for _ in range(65)]

sd = [[-1 for _ in range(65)] for _ in range(65)]

with open('sigma_calculation_stones_depth_15_vh.txt', 'r') as f:
    for i in range(65):
        for j in range(65):
            sd[i][j] = float(f.readline())
            #if i <= 20:
            #    sd[i][j] = 1000000.0

params = [
    0.12540371630624436, 0.042967909228542545, -4.4119385232059605, 0.03045617597596307, 0.03078555144050721, 0.18196487462695699, 2.2515818894708874
]

def f(n_stones, depth):
    #x = params[0] * n_stones + params[1] * depth + params[2]
    #return params[3] * x * x * x + params[4] * x * x + params[5] * x + params[6]
    return 1.0 + 0.05 * n_stones

plot_y = [0.0 for _ in range(65)]
plot_pred = [0.0 for _ in range(65)]
nums = [0 for _ in range(65)]
for i in range(65):
    for j in range(65):
        if sd[i][j] != 1000000.0:
            plot_y[i] += sd[i][j]
            nums[i] += 1
        plot_pred[i] += f(i, j)
for i in range(65):
    if nums[i]:
        plot_y[i] /= nums[i]
    plot_pred[i] /= 65
plt.title('n_stones')
plt.plot(range(65), plot_y)
plt.plot(range(65), plot_pred)
plt.show()

plot_y = [0.0 for _ in range(65)]
nums = [0 for _ in range(65)]
for j in range(65):
    for i in range(65):
        if sd[i][j] != 1000000.0:
            plot_y[j] += sd[i][j]
            nums[j] += 1
        plot_pred[j] += f(i, j)
for i in range(65):
    if nums[i]:
        plot_y[i] /= nums[i]
    plot_pred[i] /= 65
plt.title('depth')
plt.plot(range(65), plot_y)
plt.plot(range(65), plot_pred)
plt.show()

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