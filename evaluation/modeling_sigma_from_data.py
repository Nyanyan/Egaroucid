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
depth_div = 2

data = [[[[] for _ in range(0, 60 + 1, depth_div)] for _ in range(0, 60 + 1, depth_div)] for _ in range(0, 65, n_stones_div)]

for datum in raw_data:
    n_stones, depth1, score1, depth2, score2 = [int(float(elem)) for elem in datum.split()]
    if n_stones + depth1 == 64 or n_stones + depth2 == 64:
        continue
    if depth1 < depth2:
        depth1, depth2 = depth2, depth1
    data[n_stones // n_stones_div][depth1 // depth_div][depth2 // depth_div].append(abs(score1 - score2))
    #data[n_stones // n_stones_div][depth2 // depth_div][depth1 // depth_div].append(abs(score2 - score1))

w_n_stones = []
x_depth1 = []
y_depth2 = []
z_sigma = []

for w in range(len(data)):
    for x in range(len(data[w])):
        for y in range(len(data[w][x])):
            if len(data[w][x][y]) >= 3:
                '''
                n_stones = w
                depth1 = x
                depth2 = y
                '''
                n_stones = w * n_stones_div + n_stones_div / 2
                depth1 = x * depth_div + depth_div / 2
                depth2 = y * depth_div + depth_div / 2
                
                sigma = statistics.stdev(data[w][x][y])
                
                w_n_stones.append(n_stones)
                x_depth1.append(depth1)
                y_depth2.append(depth2)
                z_sigma.append(sigma * 2)
                '''
                if y == 0 and sigma >= 3.0:
                    for _ in range(8):
                        w_n_stones.append(n_stones)
                        x_depth1.append(depth1)
                        y_depth2.append(depth2)
                        z_sigma.append(sigma)
                '''

for w in range(4, 65):
    w_n_stones.append(w)
    x_depth1.append(60)
    y_depth2.append(0)
    z_sigma.append(10.0)

for w in range(4, 65):
    for x in range(10, 20):
        w_n_stones.append(w)
        x_depth1.append(x)
        y_depth2.append(0)
        z_sigma.append(8.0)

for w in range(4, 65):
    for xy in range(40, 61):
        w_n_stones.append(w)
        x_depth1.append(xy)
        y_depth2.append(xy)
        z_sigma.append(0.0)

probcut_params_before = [
    1.0 for _ in range(10)
]

def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j):
    w, x, y = wxy
    res = 0.0
    #res = probcut_a * w * w * w + probcut_b * w * w * x + probcut_c * w * w * y + probcut_d * w * x * y
    #res += probcut_e * w * w + probcut_f * w * x + probcut_g * w * y
    #res += probcut_h * w + probcut_i * (x - y) + probcut_j
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g
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

def plot_fit_result_onephase(n_stones, params):
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    phase = n_stones // n_stones_div
    x_depth1_phase = []
    y_depth2_phase = []
    z_sigma_phase = []
    for w, x, y, z in zip(w_n_stones, x_depth1, y_depth2, z_sigma):
        if int(w - n_stones_div / 2) // n_stones_div == phase:
            x_depth1_phase.append(x)
            y_depth2_phase.append(y)
            z_sigma_phase.append(z)
    ax.plot(x_depth1_phase, y_depth2_phase, z_sigma_phase, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(61), range(61))
    ax.plot_wireframe(mx, my, f((n_stones, mx, my), *params), rstride=10, cstride=10)
    ax.set_xlabel('depth1')
    ax.set_ylabel('depth2')
    ax.set_zlabel('sigma')
    plt.show()

popt, pcov = curve_fit(f, (w_n_stones, x_depth1, y_depth2), z_sigma, np.array(probcut_params_before))
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define probcut_' + chr(ord('a') + i), popt[i])
perr = np.sqrt(np.diag(pcov))
#plot_fit_result(popt)
plot_fit_result_onephase(30, popt)
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