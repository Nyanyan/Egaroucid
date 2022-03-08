import matplotlib.pyplot as plt
from random import random, randrange

with open('comb_sd_probcut.txt', 'r') as f:
    raw_data = f.read().splitlines()

n_phases = 30

data = [[0.0 for _ in range(n_phases)] for _ in range(n_phases)]

for datum in raw_data:
    i, j, sd = datum.split()
    data[int(i)][int(j)] = float(sd)

for i in range(n_phases):
    print('{', end='')
    for j in range(n_phases - 1):
        print(data[i][j], end=', ')
    print(str(data[i][n_phases - 1]) + '},')

exit()


params = [
    -0.38071038725841744, -0.38055010560956826, 0.0174953712887832, 1.7446117159062204, 33.76336205985707
]

def f(phase1, phase2):
    x = params[0] * phase1 + params[1] * phase2
    return params[2] * x * x + params[3] * x + params[4]

p = 10

plt.plot(range(n_phases), data[p])
plt.plot(range(n_phases), [f(p, i) for i in range(n_phases)])
plt.show()

def scoring():
    res = 0.0
    for phase1 in range(n_phases):
        for phase2 in range(n_phases):
            res += (f(phase1, phase2) - data[phase1][phase2]) ** 2
    return res

score = scoring()
print(score)

while True:
    idx = randrange(0, 5)
    f_param = params[idx]
    params[idx] += random() - 0.5
    n_score = scoring()
    if n_score < score:
        score = n_score
        print(score, params)
    else:
        params[idx] = f_param