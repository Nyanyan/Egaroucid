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

vhs = [[[[] for _ in range(depth_width)] for _ in range(n_phases)] for _ in range(n_scores)]
vds = [[[[] for _ in range(depth_width)] for _ in range(n_phases)] for _ in range(n_scores)]

sd = [[[-1 for _ in range(depth_width)] for _ in range(n_phases)] for _ in range(n_scores)]

with open('sigma_calculation.txt', 'r') as f:
    for i in range(n_scores):
        for j in range(n_phases):
            for k in range(depth_width):
                sd[i][j][k] = float(f.readline())

plot_y = [0.0 for _ in range(depth_width)]
nums = [0 for _ in range(depth_width)]
for i in range(n_scores):
    for j in range(n_phases):
        for k in range(depth_width):
            if sd[i][j][k] != 1000000.0:
                plot_y[k] += sd[i][j][k]
                nums[k] += 1
for i in range(depth_width):
    plot_y[i] /= nums[i]
plt.plot(range(min_depth, max_depth + 1), plot_y)
plt.show()

params = [-0.01162423449752914, -0.9699640862387829, 0.032504124127035405, 0.11349651033096486, 1.8997087721680037, 11.634261529155676]

def f(depth, phase, score):
    x = params[0] * depth + params[1] * phase + params[2] * score
    return params[3] * x * x + params[4] * x + params[5]

def scoring():
    res = 0
    for i in range(n_scores):
        for j in range(n_phases):
            for k in range(depth_width):
                ans = sd[i][j][k]
                if ans != 1000000.0:
                    pred = f(k, j, i)
                    print(ans, pred)
                    res += (pred - ans) ** 2
    return res

score = scoring()
print(score)
exit()
while True:
    idx = randrange(0, 6)
    f_param = params[idx]
    params[idx] += random() - 0.5
    n_score = scoring()
    if n_score < score:
        score = n_score
        print(score, params)
    else:
        params[idx] = f_param