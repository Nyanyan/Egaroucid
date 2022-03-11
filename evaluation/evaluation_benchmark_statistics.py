import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit

with open('eval_error_data.txt', 'r') as f:
    raw_data = f.read().splitlines()

n_phases = 30
n_scores = 129

x_phase = []
y_score = []
z_error = []


for datum in raw_data:
    n_stones, ans, pred = [int(float(elem)) for elem in datum.split()]
    x_phase.append((n_stones - 4) // 2)
    y_score.append(pred)
    z_error.append(ans - pred)

score_modify_params_before = [
    1.0 for _ in range(10)
]

def f(xy, score_modify_a, score_modify_b, score_modify_c, score_modify_d, score_modify_e, score_modify_f, score_modify_g, score_modify_h, score_modify_i, score_modify_j):
    x, y = xy
    res = score_modify_a * x * x * x + score_modify_b * x * x * y + score_modify_c * x * y * y + score_modify_d * y * y * y
    res += score_modify_e * x * x + score_modify_f * x * y + score_modify_g * y * y
    res += score_modify_h * x + score_modify_i * y + score_modify_j
    return res

def plot_fit_result(params):
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot(x_phase, y_score, z_error, ms=1, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(30), range(-64, 65))
    ax.plot_wireframe(mx, my, f((mx, my), *params), rstride=5, cstride=5)
    ax.set_xlabel('phase')
    ax.set_ylabel('estimated score')
    ax.set_zlabel('error')
    plt.show()

#plot_fit_result(*score_modify_params_before)

popt, pcov = curve_fit(f, (x_phase, y_score), z_error, np.array(score_modify_params_before))
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define score_modify_' + chr(ord('a') + i), popt[i])
perr = np.sqrt(np.diag(pcov))
plot_fit_result(popt)