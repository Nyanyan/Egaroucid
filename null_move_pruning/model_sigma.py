import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

data_file = 'data/mid.txt'

const_weight = 0.0

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

data = []

def model(pass_score):
    return 13.895352711158006 * pass_score / 64 - 1.4594275635787501

data = [[[] for _ in range(61)] for _ in range(65)] # n_discs, depth
for datum in raw_data:
    n_discs, depth, pass_score, val = [int(elem) for elem in datum.split()]
    data[n_discs][depth].append(val - model(pass_score))

x_n_discs = []
y_depth = []
z_error = []
weight = []
for n_discs in range(65):
    if n_discs >= 64 - 13:
        continue
    for depth in range(61):
        if len(data[n_discs][depth]) >= 3:
            mean = statistics.mean(data[n_discs][depth])
            sigma = statistics.stdev(data[n_discs][depth])
            x_n_discs.append(n_discs)
            y_depth.append(depth)
            z_error.append(sigma + const_weight)
            weight.append(1 / len(data[n_discs][depth]))

for depth in range(65):
    x_n_discs.append(4)
    y_depth.append(depth)
    z_error.append(0.2)
    weight.append(0.006)

def f(xy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    x, y = xy
    x = x / 64
    y = y / 60
    res = probcut_a * x + probcut_b * y
    res = probcut_c * res * res * res + probcut_d * res * res + probcut_e * res + probcut_f
    return res

def plot_fit_result(params):
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    ax.plot(x_n_discs, y_depth, z_error, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(65), range(61))
    ax.plot_wireframe(mx, my, f((mx, my), *params), rstride=10, cstride=10)
    ax.set_xlabel('n_discs')
    ax.set_ylabel('depth')
    ax.set_zlabel('error')
    plt.show()

probcut_params_before = [1.0 for _ in range(10)]

popt, pcov = curve_fit(f, (x_n_discs, y_depth), z_error, np.array(probcut_params_before), sigma=weight, absolute_sigma=True)
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define null_move_pruning_' + chr(ord('a') + i), popt[i])
plot_fit_result(popt)
