import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

data_file = 'data/mid.txt'

const_weight = 1.0

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)] # n_discs, depth1, depth2 (depth1 < depth2)

for datum in raw_data:
    n_discs, depth1, depth2, abs_error = [int(elem) for elem in datum.split()]
    data[n_discs][depth1][depth2].append(abs_error)

w_n_discs = []
x_depth1 = []
y_depth2 = []
z_sigma = []
weight = []

for n_discs in range(len(data)):
    for depth1 in range(len(data[n_discs])):
        for depth2 in range(len(data[n_discs][depth1])):
            if len(data[n_discs][depth1][depth2]) >= 3:
                sigma = statistics.stdev(data[n_discs][depth1][depth2])
                w_n_discs.append(n_discs)
                x_depth1.append(depth1)
                y_depth2.append(depth2)
                z_sigma.append(sigma + const_weight)
                weight.append(1 / len(data[n_discs][depth1][depth2]))

for n_discs in range(61):
    for depth1 in range(20, 65):
        #for depth2 in range(depth1 + 2, 65, 2):
        depth2 = depth1 + 2
        w_n_discs.append(n_discs)
        x_depth1.append(depth1)
        y_depth2.append(depth2)
        z_sigma.append(0.5 + 1.0 * (depth2 - depth1) / depth2 + const_weight)
        weight.append(0.05)

def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    w, x, y = wxy
    w = w / 64
    x = x / 60
    y = y / 60
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    return np.minimum(20.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j))

def plot_fit_result(params):
    z_sigma_pred = [f((w, x, y), *params) for w, x, y in zip(w_n_discs, x_depth1, y_depth2)]
    max_z = max(max(z_sigma), max(z_sigma_pred))
    print(max_z)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    cm = plt.cm.get_cmap('gray')
    for w, x, y, z in zip(w_n_discs, x_depth1, y_depth2, z_sigma):
        ax.plot(w, x, y, ms=3, marker="o",color=cm(z / max_z))
    ax.set_xlabel('n_discs')
    ax.set_ylabel('depth1')
    ax.set_zlabel('depth2')
    plt.show()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for w, x, y, z in zip(w_n_discs, x_depth1, y_depth2, z_sigma_pred):
        ax.plot(w, x, y, ms=3, marker="o",color=cm(z / max_z))
    ax.set_xlabel('n_discs')
    ax.set_ylabel('depth1')
    ax.set_zlabel('depth2')
    plt.show()

def plot_fit_result_onephase(n_discs, params):
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    phase = n_discs
    x_depth1_phase = []
    y_depth2_phase = []
    z_sigma_phase = []
    for w, x, y, z in zip(w_n_discs, x_depth1, y_depth2, z_sigma):
        if w == phase:
            x_depth1_phase.append(x)
            y_depth2_phase.append(y)
            z_sigma_phase.append(z)
    ax.plot(x_depth1_phase, y_depth2_phase, z_sigma_phase, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(61), range(61))
    ax.plot_wireframe(mx, my, f_max((n_discs, mx, my), *params), rstride=10, cstride=10)
    ax.set_xlabel('depth1')
    ax.set_ylabel('depth2')
    ax.set_zlabel('sigma')
    plt.show()

probcut_params_before = [
    -0.0027183880227839127,
    -0.005159330980892623,
    0.04069811753963199,
    -3.126668257717306,
    8.513417624696323,
    -9.550971692854511,
    9.03198419537373,
    1.0,
    1.0,
    1.0
    #1.0 for _ in range(10)
]

popt, pcov = curve_fit(f, (w_n_discs, x_depth1, y_depth2), z_sigma, np.array(probcut_params_before), sigma=weight, absolute_sigma=True)
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define probcut_' + chr(ord('a') + i), popt[i])
plot_fit_result_onephase(20, popt)
