import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation

with open('sigma_data.txt', 'r') as f:
    raw_data = f.read().splitlines()

n_stones_div = 2

data = [[[] for _ in range(0, 33 + 1)] for _ in range(0, 64, n_stones_div)]

for datum in raw_data:
    n_stones, depth1, score1, depth2, score2 = [int(float(elem)) for elem in datum.split()]
    if n_stones + depth1 != 64 and n_stones + depth2 != 64:
        continue
    if n_stones + depth2 == 64:
        depth1, depth2 = depth2, depth1
        score1, score2 = score2, score1
    data[n_stones // n_stones_div][depth2].append(score1 - score2)

x_n_stones = []
y_depth = []
z_sigma = []

for x in range(len(data)):
    for y in range(len(data[x])):
        if len(data[x][y]) >= 3:
            '''
            n_stones = w
            depth1 = x
            depth2 = y
            '''
            n_stones = x * n_stones_div + n_stones_div / 2
            depth = y
            
            x_n_stones.append(n_stones)
            y_depth.append(depth)
            z_sigma.append(statistics.stdev(data[x][y]))

for x in range(10):
    x_n_stones.append(x)
    y_depth.append(0)
    z_sigma.append(7.0)

probcut_end_params_before = [
    1.0 for _ in range(10)
]

def f(xy, probcut_end_a, probcut_end_b, probcut_end_c, probcut_end_d, probcut_end_e, probcut_end_f, probcut_end_g, probcut_end_h, probcut_end_i, probcut_end_j):
    x, y = xy
    res = 0.0
    res = probcut_end_a * x + probcut_end_b * y
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f
    return res

def to_rgb(x):
    r = min(1.0, x / 20)
    g = 0.0
    b = 0.0
    return [r, g, b]

def plot_fit_result_onephase(params):
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    phase = n_stones // n_stones_div
    ax.plot(x_n_stones, y_depth, z_sigma, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(65), range(31))
    ax.plot_wireframe(mx, my, f((mx, my), *params), rstride=10, cstride=10)
    ax.set_xlabel('n_stones')
    ax.set_ylabel('depth')
    ax.set_zlabel('sigma')
    plt.show()

popt, pcov = curve_fit(f, (x_n_stones, y_depth), z_sigma, np.array(probcut_end_params_before))
#popt = probcut_end_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define probcut_end_' + chr(ord('a') + i), popt[i])
perr = np.sqrt(np.diag(pcov))
#plot_fit_result(popt)
plot_fit_result_onephase(popt)
exit()

fig = plt.figure()

def update(i):
    n_stones = i % 61 + 4
    phase = n_stones // n_stones_div

    plt.cla()

    x_depth1_phase = []
    y_depth2_phase = []
    z_sigma_phase = []
    for w, x, y, z in zip(w_n_stones, x_depth1, y_depth2, z_sigma):
        if int(w - n_stones_div / 2) // n_stones_div == phase:
            x_depth1_phase.append(x)
            y_depth2_phase.append(y)
            z_sigma_phase.append(z)
    ax = Axes3D(fig)
    ax.plot(x_depth1_phase, y_depth2_phase, z_sigma_phase, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(61), range(61))
    ax.plot_wireframe(mx, my, f((n_stones, mx, my), *popt), rstride=10, cstride=10)
    plt.title(str(n_stones))
    ax.set_xlabel('depth1')
    ax.set_ylabel('depth2')
    ax.set_zlabel('sigma n_stones=' + str(n_stones))
    ax.set_zlim([-1.0, 20.0])

ani = animation.FuncAnimation(fig, update, interval = 10)
plt.show()