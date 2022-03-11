import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit

with open('sigma_data.txt', 'r') as f:
    raw_data = f.read().splitlines()

n_stones_div = 4
depth_div = 4

data = [[[[] for _ in range(0, 33 + 1, depth_div)] for _ in range(0, 33 + 1, depth_div)] for _ in range(0, 64, n_stones_div)]

for datum in raw_data:
    n_stones, depth1, score1, depth2, score2 = [int(float(elem)) for elem in datum.split()]
    data[n_stones // n_stones_div][depth1 // depth_div][depth2 // depth_div].append(score1 - score2)
    data[n_stones // n_stones_div][depth2 // depth_div][depth1 // depth_div].append(score2 - score1)

w_n_stones = []
x_depth1 = []
y_depth2 = []
z_sigma = []

for w in range(len(data)):
    for x in range(len(data[w])):
        for y in range(len(data[w][x])):
            if len(data[w][x][y]) >= 3:
                n_stones = w * n_stones_div + n_stones_div / 2
                depth1 = x * depth_div + depth_div / 2
                depth2 = y * depth_div + depth_div / 2
                w_n_stones.append(n_stones)
                x_depth1.append(depth1)
                y_depth2.append(depth2)
                z_sigma.append(statistics.stdev(data[w][x][y]))

probcut_params_before = [
    1.0 for _ in range(10)
]

def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    w, x, y = wxy
    res = probcut_a * w * w * w + probcut_b * w * w * (x + y) + probcut_c * w * (x * x + y * y) + probcut_d * (x * x * x + y * y * y)
    res += probcut_e * w * w + probcut_f * w * (x + y) + probcut_g * (x * x + y * y)
    res += probcut_h * w + probcut_i * (x + y) + probcut_j
    return res

def to_rgb(x):
    r = min(1.0, x / 20)
    g = 0.0
    b = 0.0
    return [r, g, b]

def plot_fit_result(params):
    z_sigma_pred = [f((w, x, y), *params) for w, x, y in zip(w_n_stones, x_depth1, y_depth2)]
    max_z = max(max(z_sigma), max(z_sigma_pred))
    print(max_z)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    cm = plt.cm.get_cmap('gray')
    for w, x, y, z in zip(w_n_stones, x_depth1, y_depth2, z_sigma):
        ax.plot(w, x, y, ms=3, marker="o",color=cm(z / max_z))
    ax.set_xlabel('n_stones')
    ax.set_ylabel('depth1')
    ax.set_zlabel('depth2')
    plt.show()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for w, x, y, z in zip(w_n_stones, x_depth1, y_depth2, z_sigma_pred):
        ax.plot(w, x, y, ms=3, marker="o",color=cm(z / max_z))
    ax.set_xlabel('n_stones')
    ax.set_ylabel('depth1')
    ax.set_zlabel('depth2')
    plt.show()

popt, pcov = curve_fit(f, (w_n_stones, x_depth1, y_depth2), z_sigma, np.array(probcut_params_before))
print([float(elem) for elem in popt])
print(pcov)
for i in range(len(popt)):
    print('#define probcut_' + chr(ord('a') + i), popt[i])
perr = np.sqrt(np.diag(pcov))
plot_fit_result(popt)