import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

#data_files = ['data/probcut_end1.txt', 'data/probcut_end2.txt', 'data/probcut_end3.txt', 'data/probcut_end4.txt']
#data_files = ['data/probcut_end6.txt']
#data_files = ['data/probcut_end8.txt', 'data/probcut_end9.txt', 'data/probcut_end10.txt', 'data/probcut_end11.txt', 'data/probcut_end12.txt']
#data_files = ['data/probcut_end16.txt']
#data_files = ['data/probcut_end19.txt']
#data_files = ['data/probcut_end0.txt']
#data_files = ['data/20240925_1_7_4/probcut_end0.txt']
#data_files = ['data/20241118_1_7_5/probcut_end0.txt']
#data_files = ['data/20241128_1_7_5/probcut_end0.txt']
data_files = ['data/20241130_1_7_5/probcut_end0.txt', 'data/20241130_1_7_5/probcut_end1.txt', 'data/20241130_1_7_5/probcut_end2.txt']


probcut_mid_param = [0.93740805837003, -7.340323137961951, 1.1401695320187872, 0.700567733735339, 2.662003673678691, 3.0554301965778063, 2.0942574977708674]

def mid_f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f, probcut_g):
    w, x, y = wxy
    w = w / 64 # n_discs
    x = x / 60 # depth_short
    y = y / 60 # depth_long (== 64 - n_discs)
    res = probcut_a * w + probcut_b * x + probcut_c * y
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g
    return res

data = [[[] for _ in range(61)] for _ in range(65)] # n_discs, depth, error (exact - predict)
for data_file in data_files:
    try:
        with open(data_file, 'r') as f:
            raw_data = f.read().splitlines()
        for datum in raw_data:
            n_discs, depth, error = [int(elem) for elem in datum.split()]
            data[n_discs][depth].append(error)
    except:
        print('cannot open', data_file)
x_n_discs_sd = []
y_depth_sd = []
z_sd = []
weight_sd = []

x_n_discs_mean = []
y_depth_mean = []
z_mean = []
weight_mean = []


for n_discs in range(len(data)):
    if n_discs >= 64 - 12:
        continue
    for depth in range(3, len(data[n_discs])):
        if len(data[n_discs][depth]) >= 3:
            '''
            mean = statistics.mean(data[n_discs][depth])
            sd = statistics.stdev(data[n_discs][depth])
            '''
            mean = 0.0
            sd = 0.0
            for elem in data[n_discs][depth]:
                sd += (elem - mean) ** 2
            sd = math.sqrt(sd / len(data[n_discs][depth]))
            

            print('n_discs', n_discs, 'depth', depth, 'mean', mean, 'sd', sd, 'n_data', len(data[n_discs][depth]))
            x_n_discs_sd.append(n_discs)
            y_depth_sd.append(depth)
            z_sd.append(sd)
            weight_sd.append(1 / len(data[n_discs][depth]))
            
            x_n_discs_mean.append(n_discs)
            y_depth_mean.append(depth)
            z_mean.append(mean)
            weight_mean.append(1 / len(data[n_discs][depth]))

for n_discs in range(1):
    for depth in range(1):
        x_n_discs_sd.append(n_discs)
        y_depth_sd.append(depth)
        z_sd.append(10.0 - n_discs / 60 * 1.0 - depth * 0.05)
        weight_sd.append(0.01)

def f(xy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f):
    x, y = xy
    x = x / 64
    y = y / 60
    res = probcut_a * x + probcut_b * y
    res = probcut_c * res * res * res + probcut_d * res * res + probcut_e * res + probcut_f
    return res

def f_max(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f):
    return np.minimum(10.0, np.maximum(-0.5, f(wxy, probcut_a, probcut_b, probcut_c, probcut_d, probcut_e, probcut_f)))

def plot_fit_result(x, y, z, params):
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(xs, ys, zs, ms=3, marker="o",linestyle='None')
    #ax.plot(sdxs, sdys, sdzs, ms=3, marker="o",linestyle='None')
    ax.plot(x, y, z, ms=3, marker="o",linestyle='None')
    mx, my = np.meshgrid(range(65), range(16))
    ax.plot_wireframe(mx, my, f_max((mx, my), *params), rstride=5, cstride=5)
    # midgame MPC
    x_mid_mpc = []
    y_mid_mpc = []
    for xx in range(64 - 13):
        for yy in range(15):
            x_mid_mpc.append(xx)
            y_mid_mpc.append(yy)
    z_mid_mpc = []
    for xx, yy in zip(x_mid_mpc, y_mid_mpc):
        zz = mid_f([xx, yy, 64 - xx], *probcut_mid_param)
        z_mid_mpc.append(zz)
    ax.plot(x_mid_mpc, y_mid_mpc, z_mid_mpc, ms=1, marker="o",linestyle='None')
    ax.set_xlabel('n_discs(=64-long_depth)')
    ax.set_ylabel('search_depth')
    ax.set_zlabel('error')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 16)
    plt.show()

popt_sd, pcov_sd = curve_fit(f, (x_n_discs_sd, y_depth_sd), z_sd, np.ones(6), sigma=weight_sd, absolute_sigma=True)
print([float(elem) for elem in popt_sd])
for i in range(len(popt_sd)):
    print('constexpr double probcut_end_' + chr(ord('a') + i) + ' = ' + str(popt_sd[i]) + ';')

'''
popt_mean, pcov_mean = curve_fit(f, (x_n_discs_mean, y_depth_mean), z_mean, np.ones(6), sigma=weight_mean, absolute_sigma=True)
print([float(elem) for elem in popt_mean])
for i in range(len(popt_sd)):
    print('#define probcut_end_mean_' + chr(ord('a') + i), popt_sd[i])
'''


plot_fit_result(x_n_discs_sd, y_depth_sd, z_sd, popt_sd)
#plot_fit_result(x_n_discs_mean, y_depth_mean, z_mean, popt_mean)
