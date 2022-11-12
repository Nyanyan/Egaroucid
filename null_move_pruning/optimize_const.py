import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import animation
import math

data_file = 'data/mid.txt'

with open(data_file, 'r') as f:
    raw_data = f.read().splitlines()

data = []

x_pass_scores = [[], []]
y_vals = [[], []]
for datum in raw_data:
    n_discs, pass_score, val = [int(elem) for elem in datum.split()]
    if n_discs >= 64 - 13:
        continue
    y_vals[n_discs % 2].append(val)
    x_pass_scores[n_discs % 2].append(pass_score)

def f(x, const_a, const_b):
    return const_a * x + const_b

def y_x(x):
    return -x

def plot_fit_result(params1, params2):
    plt.scatter(x_pass_scores[0], y_vals[0], color='blue')
    plt.scatter(x_pass_scores[1], y_vals[1], color='cyan')
    x = np.array(range(-64, 65))
    x_small = np.array(range(-40, 41))
    plt.plot(x, f(x, *params1), color='red')
    plt.plot(x, f(x, *params2), color='orange')
    plt.plot(x_small, y_x(x_small), color='green')
    plt.show()

probcut_params_before = [1.0 for _ in range(2)]

popt = [[], []]
for parity in [0, 1]:
    popt[parity], pcov = curve_fit(f, x_pass_scores[parity], y_vals[parity], np.array(probcut_params_before))
    #popt = probcut_params_before
for i in range(len(popt[0])):
    print('constexpr double null_move_const_' + chr(ord('a') + i) + '[2] = {' + str(popt[0][i]) + ', ' + str(popt[1][i]) + '};')
plot_fit_result(popt[0], popt[1])
