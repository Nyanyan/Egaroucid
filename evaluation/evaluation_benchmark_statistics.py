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

probcut_params_before = [
    1.0 for _ in range(10)
]

def f(xy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    x, y = xy
    res = probcut_a * x * x * x + probcut_b * x * x * y + probcut_c * x * y * y + probcut_d * y * y * y
    res += probcut_e * x * x + probcut_f * x * y + probcut_g * y * y
    res += probcut_h * x + probcut_i * y + probcut_j
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

#plot_fit_result(*probcut_params_before)

popt, pcov = curve_fit(f, (x_phase, y_score), z_error, np.array(probcut_params_before))
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define probcut_' + chr(ord('a') + i), popt[i])
perr = np.sqrt(np.diag(pcov))
plot_fit_result(popt)