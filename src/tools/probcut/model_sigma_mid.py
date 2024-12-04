import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

#depth1: short
#depth2: long

#data_files = ['data/probcut_mid9.txt']
#data_files = ['data/probcut_mid10.txt', 'data/probcut_mid11.txt']
#data_files = ['data/probcut_mid14.txt', 'data/probcut_mid15.txt', 'data/probcut_mid16.txt', 'data/probcut_mid17.txt', 'data/probcut_mid18.txt', 'data/probcut_mid19.txt']
#data_files = ['data/probcut_mid0.txt']
#data_files = ['data/20240925_1_7_4/probcut_mid0.txt']
#data_files = ['data/20241118_1_7_5/probcut_mid0.txt']
#data_files = ['data/20241128_1_7_5/probcut_mid0.txt']
data_files = ['data/20241130_1_7_5/probcut_mid0.txt']


data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)] # n_discs, depth1, depth2 (depth1 < depth2)

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            data[n_discs][depth1][depth2].append(error)
    except:
        print('cannot open', data_file)

w_n_discs_sd = []
x_depth1_sd = []
y_depth2_sd = []
z_sd = []
weight_sd = []

w_n_discs_mean = []
x_depth1_mean = []
y_depth2_mean = []
z_mean = []
weight_mean = []

for n_discs in range(len(data)):
    for depth1 in range(len(data[n_discs])):
        for depth2 in range(len(data[n_discs][depth1])):
            if len(data[n_discs][depth1][depth2]) >= 3:
                #mean = statistics.mean(data[n_discs][depth1][depth2])
                #sd = statistics.stdev(data[n_discs][depth1][depth2])
                
                mean = 0.0
                sd = 0.0
                for elem in data[n_discs][depth1][depth2]:
                    sd += elem ** 2
                sd = math.sqrt(sd / len(data[n_discs][depth1][depth2]))
                
                print('n_discs', n_discs, 'depth1', depth1, 'depth2', depth2, 'mean', mean, 'sd', sd, 'n_data', len(data[n_discs][depth1][depth2]))

                w_n_discs_sd.append(n_discs)
                x_depth1_sd.append(depth1)
                y_depth2_sd.append(depth2)
                z_sd.append(sd)
                weight_sd.append(0.001)

                w_n_discs_mean.append(n_discs)
                x_depth1_mean.append(depth1)
                y_depth2_mean.append(depth2)
                z_mean.append(mean)
                weight_mean.append(0.001)

for n_discs in range(61):
    for depth2 in range(30, 31):
        depth1 = 0
        z = 3.0 + 12.0 * ((n_discs - 4) / 60)
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)
        z_sd.append(z)
        weight_sd.append(0.001)


def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g):
    w, x, y = wxy
    w = w / 64
    x = x / 60
    y = y / 60
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g):
    return np.minimum(30.0, np.maximum(-2.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g)))

def plot_fit_result_onephase(w, x, y, z, n_discs, params):
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    phase = n_discs
    x_depth1_phase = []
    y_depth2_phase = []
    z_error_phase = []
    for ww, xx, yy, zz in zip(w, x, y, z):
        if ww == phase:
            x_depth1_phase.append(xx)
            y_depth2_phase.append(yy)
            z_error_phase.append(zz)
    ax.plot(x_depth1_phase, y_depth2_phase, z_error_phase, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(13), range(30))
    ax.plot_wireframe(mx, my, f_max((n_discs, mx, my), *params), rstride=4, cstride=2)
    ax.set_xlabel('depth1_short')
    ax.set_ylabel('depth2_long')
    ax.set_zlabel('error')
    ax.set_xlim((0, 12))
    ax.set_ylim((0, 30))
    ax.set_zlim((0, 12))
    plt.show()

popt_sd, pcov_sd = curve_fit(f, (w_n_discs_sd, x_depth1_sd, y_depth2_sd), z_sd, np.ones(7), sigma=weight_sd, absolute_sigma=True)
print([float(elem) for elem in popt_sd])
for i in range(len(popt_sd)):
    print('constexpr double probcut_' + chr(ord('a') + i) + ' = ' + str(popt_sd[i]) + ';')

for i in [10, 20, 30, 40, 50]:
    plot_fit_result_onephase(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, i, popt_sd)

'''
#popt_mean, pcov_mean = curve_fit(f, (w_n_discs_mean, x_depth1_mean, y_depth2_mean), z_mean, np.ones(7), sigma=weight_mean, absolute_sigma=True)
popt_mean = [0 for _ in range(7)]
print([float(elem) for elem in popt_mean])
for i in range(len(popt_mean)):
    print('#define probcut_mean_' + chr(ord('a') + i), popt_mean[i])

for i in [10, 20, 30, 40, 50]:
    plot_fit_result_onephase(w_n_discs_mean, x_depth1_mean, y_depth2_mean, z_mean, i, popt_mean)
'''
