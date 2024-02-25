import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

#data_files = ['data/probcut_end1.txt', 'data/probcut_end2.txt', 'data/probcut_end3.txt', 'data/probcut_end4.txt']
data_files = ['data/probcut_end5.txt']

data = [[[] for _ in range(61)] for _ in range(65)] # n_discs, depth
for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth, error = [int(elem) for elem in datum.split()]
            data[n_discs][depth].append(error)
    except:
        print('cannot open', data_file)
x_n_discs = []
y_depth = []
z_error = []
weight = []


for n_discs in range(len(data)):
    if n_discs > 60:
        continue
    for depth in range(len(data[n_discs])):
        if len(data[n_discs][depth]) >= 3:
            mean = statistics.mean(data[n_discs][depth])
            sigma = statistics.stdev(data[n_discs][depth])
            print('n_discs', n_discs, 'depth', depth, 'mean', mean, 'sd', sigma, 'n_data', len(data[n_discs][depth]))
            x_n_discs.append(n_discs)
            y_depth.append(depth)
            z_error.append(sigma)
            weight.append(1 / len(data[n_discs][depth]))

for n_discs in range(80):
    depth = 64 - n_discs + 5
    x_n_discs.append(n_discs)
    y_depth.append(depth)
    z_error.append(0.0)
    weight.append(0.1)


for n_discs in range(60):
    depth = 0
    x_n_discs.append(n_discs)
    y_depth.append(depth)
    z_error.append(4.5 - (n_discs - 4 - depth) / 60 * 2.0)
    weight.append(0.01)

'''
for n_discs in range(60):
    depth = (64 - n_discs) / 2
    x_n_discs.append(n_discs)
    y_depth.append(depth)
    z_error.append(3.0 - (n_discs - 4 - depth) / 60 * 0.5)
    weight.append(0.008)
'''

def f(xy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    x, y = xy
    x = x / 64
    y = y / 60
    res = probcut_a * x + probcut_b * y
    res = probcut_c * res * res * res + probcut_d * res * res + probcut_e * res + probcut_f
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    return np.minimum(15.0, np.maximum(-2.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j)))

def plot_fit_result(params):
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    ax.plot(x_n_discs, y_depth, z_error, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(65), range(61))
    ax.plot_wireframe(mx, my, f_max((mx, my), *params), rstride=10, cstride=10)
    ax.set_xlabel('n_discs')
    ax.set_ylabel('search_depth')
    ax.set_zlabel('error')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 15)
    plt.show()

probcut_params_before = [1.0 for _ in range(10)]

probcut_params_old = [1.7618824674282139, 1.6761979186286653, -1.5368514045644108, 5.649224423285564, -12.020118030448877, 14.261869478626625, 1.0, 1.0, 1.0, 1.0]

popt, pcov = curve_fit(f, (x_n_discs, y_depth), z_error, np.array(probcut_params_before), sigma=weight, absolute_sigma=True)
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define probcut_end_' + chr(ord('a') + i), popt[i])

plot_fit_result(popt)
#plot_fit_result(probcut_params_old)
