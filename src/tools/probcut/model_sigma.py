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
#data_files = ['data/20241130_1_7_5/probcut_mid0.txt']
#data_files_end = ['data/20241130_1_7_5/probcut_end0.txt', 'data/20241130_1_7_5/probcut_end1.txt', 'data/20241130_1_7_5/probcut_end2.txt']
#data_files = ['data/20250109_1_7_6/probcut_mid0.txt', 'data/20250109_1_7_6/probcut_mid1.txt', 'data/20250109_1_7_6/probcut_mid2.txt']
#data_files_end = ['data/20250109_1_7_6/probcut_end0.txt', 'data/20250109_1_7_6/probcut_end1.txt']
#data_files = ['data/20250306_1_7_6_20250305_1/probcut_mid0.txt', 'data/20250306_1_7_6_20250305_1/probcut_mid1.txt', 'data/20250306_1_7_6_20250305_1/probcut_mid2.txt']
#data_files_end = ['data/20250306_1_7_6_20250305_1/probcut_end0.txt', 'data/20250306_1_7_6_20250305_1/probcut_end1.txt']
#data_files = ['data/20250402_1_7_6_20250330_1/probcut_mid0.txt']
#data_files_end = ['data/20250402_1_7_6_20250330_1/probcut_end0.txt']
#data_files = ['data/20250514_1_7_7/probcut_mid0.txt', 'data/20250514_1_7_7/probcut_mid1.txt', 'data/20250514_1_7_7/probcut_mid2.txt', 'data/20250514_1_7_7/probcut_mid3.txt', 'data/20250514_1_7_7/probcut_mid4.txt', 'data/20250514_1_7_7/probcut_mid5.txt']
# data_files = ['data/20250625_1_7_7_20250618_2/probcut_mid0.txt', 'data/20250625_1_7_7_20250618_2/probcut_mid1.txt']
#data_files = ['data/20250514_1_7_7/probcut_mid0.txt', 'data/20250514_1_7_7/probcut_mid1.txt', 'data/20250514_1_7_7/probcut_mid2.txt', 'data/20250514_1_7_7/probcut_mid3.txt', 'data/20250514_1_7_7/probcut_mid4.txt']
#data_files_end = ['data/20250514_1_7_7/probcut_end0.txt', 'data/20250514_1_7_7/probcut_end1.txt', 'data/20250514_1_7_7/probcut_end2.txt', 'data/20250514_1_7_7/probcut_end3.txt']
#data_files_end = ['data/20250514_1_7_7/probcut_end0.txt']
# data_files_end = ['data/20250625_1_7_7_20250618_2/probcut_end0.txt']

# data_files = ['data/20250916_7_8_20250915_1/probcut_mid0.txt', 'data/20250916_7_8_20250915_1/probcut_mid1.txt']
# data_files_end = ['data/20250916_7_8_20250915_1/probcut_end0.txt', 'data/20250916_7_8_20250915_1/probcut_end1.txt', 'data/20250916_7_8_20250915_1/probcut_end2.txt']

data_files = ['data/20250918_7_8_20250917_1/probcut_mid0.txt']
data_files_end = ['data/20250918_7_8_20250917_1/probcut_end0.txt', 'data/20250918_7_8_20250917_1/probcut_end1.txt']


data = [[[[] for _ in range(61)] for _ in range(61)] for _ in range(65)] # n_discs, depth1, depth2 (depth1 < depth2)

for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, depth2, error = [int(elem) for elem in datum.split()]
            if depth2 > 1:
                data[n_discs][depth1][depth2].append(error)
    except:
        print('cannot open', data_file)

for data_file in data_files_end:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth1, error = [int(elem) for elem in datum.split()]
            depth2 = 64 - n_discs
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
    #for n_discs in range(4 + 12, 51):
    for depth1 in range(len(data[n_discs])): # short
        for depth2 in range(len(data[n_discs][depth1])): # long
            if len(data[n_discs][depth1][depth2]) >= 3:
                '''
                mean = statistics.mean(data[n_discs][depth1][depth2])
                sd = statistics.stdev(data[n_discs][depth1][depth2])
                #'''
                #'''
                mean = 0.0
                sd = 0.0
                for elem in data[n_discs][depth1][depth2]:
                    sd += elem ** 2
                sd = math.sqrt(sd / len(data[n_discs][depth1][depth2]))
                #'''
                #print('n_discs', n_discs, 'depth1', depth1, 'depth2', depth2, 'mean', mean, 'sd', sd, 'n_data', len(data[n_discs][depth1][depth2]))

                w_n_discs_sd.append(n_discs)
                x_depth1_sd.append(depth1)
                y_depth2_sd.append(depth2)
                z_sd.append(sd)
                if depth2 == 64 - n_discs:
                    weight_sd.append(0.0001)
                else:
                    weight_sd.append(0.001)

                w_n_discs_mean.append(n_discs)
                x_depth1_mean.append(depth1)
                y_depth2_mean.append(depth2)
                z_mean.append(mean)
                weight_mean.append(0.001)

#'''
for n_discs in range(4, 30):
    depth2 = 64 - n_discs
    depth1 = 0
    z = 10.0 + 2.0 * ((n_discs - 4) / 60)
    w_n_discs_sd.append(n_discs)
    x_depth1_sd.append(depth1)
    y_depth2_sd.append(depth2)
    z_sd.append(z)
    weight_sd.append(0.0008)
#'''
'''
for n_discs in range(4, 61):
    for depth2 in range(30, 31):
        depth1 = depth2 - 2
        z = 2.5
        w_n_discs_sd.append(n_discs)
        x_depth1_sd.append(depth1)
        y_depth2_sd.append(depth2)
        z_sd.append(z)
        weight_sd.append(0.001)
'''
def f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j, probcut_k, probcut_l, probcut_m, probcut_o, probcut_p, probcut_q):
    w, x, y = wxy
    w = w / 64 # n_discs
    x = x / 60 # depth 1 short
    y = y / 60 # depth 2 long
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j, probcut_k, probcut_l, probcut_m, probcut_o, probcut_p, probcut_q):
    return np.minimum(30.0, np.maximum(-2.0, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g, probcut_h, probcut_i, probcut_j, probcut_k, probcut_l, probcut_m, probcut_o, probcut_p, probcut_q)))



popt_sd, pcov_sd = curve_fit(f, (w_n_discs_sd, x_depth1_sd, y_depth2_sd), z_sd, np.ones(16), sigma=weight_sd, absolute_sigma=True, maxfev=500000)
print([float(elem) for elem in popt_sd])
for i in range(len(popt_sd)):
    print('constexpr double probcut_' + chr(ord('a') + i) + ' = ' + str(popt_sd[i]) + ';')


def plot_fit_result_allphases(w, x, y, z, params):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n_moves in range(10, 60, 10):
        n_discs = 4 + n_moves
        x_depth1_phase = []
        y_depth2_phase = []
        z_error_phase = []
        for ww, xx, yy, zz in zip(w, x, y, z):
            if ww == n_discs:
                x_depth1_phase.append(xx)
                y_depth2_phase.append(yy)
                z_error_phase.append(zz)
        color = next(ax._get_lines.prop_cycler)['color']  # Get the next color in the cycle
        ax.plot(x_depth1_phase, y_depth2_phase, z_error_phase, ms=5, marker="o", alpha=1.0, linestyle='None', label=f'n_moves={n_moves}', color=color)
        n_remaining_moves = 60 - n_moves
        mx, my = np.meshgrid(range(20), range(n_remaining_moves))
        ax.plot_wireframe(mx, my, f_max((n_discs, mx, my), *params), rstride=4, cstride=2, alpha=0.5, color=color)

    ax.set_xlabel('depth1_short')
    ax.set_ylabel('depth2_long')
    ax.set_zlabel('error')
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 30))
    ax.set_zlim((0, 8))
    ax.legend()
    plt.show()

plot_fit_result_allphases(w_n_discs_sd, x_depth1_sd, y_depth2_sd, z_sd, popt_sd)

'''
popt_mean, pcov_mean = curve_fit(f, (w_n_discs_mean, x_depth1_mean, y_depth2_mean), z_mean, np.ones(7), sigma=weight_mean, absolute_sigma=True)
#popt_mean = [0 for _ in range(7)]
print([float(elem) for elem in popt_mean])
for i in range(len(popt_mean)):
    print('#define probcut_mean_' + chr(ord('a') + i), popt_mean[i])

for i in [10, 20, 30, 40, 50]:
    plot_fit_result_onephase(w_n_discs_mean, x_depth1_mean, y_depth2_mean, z_mean, i, popt_mean)
'''
