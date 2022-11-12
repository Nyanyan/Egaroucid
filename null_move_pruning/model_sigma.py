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
    return 12.249025096921958 * pass_score / 64 - 1.0806086731648539

data = [[] for _ in range(65)] # n_discs
for datum in raw_data:
    n_discs, depth, pass_score, val = [int(elem) for elem in datum.split()]
    data[n_discs].append(val - model(pass_score))

x_n_discs = []
y_error = []
weight = []
for n_discs in range(len(data)):
    if n_discs >= 64 - 13:
        continue
    if len(data[n_discs]) >= 3:
        mean = statistics.mean(data[n_discs])
        sigma = statistics.stdev(data[n_discs])
        x_n_discs.append(n_discs)
        y_error.append(sigma + const_weight)
        weight.append(1 / len(data[n_discs]))

def f(x, const_a, const_b, const_c):
    x = x / 64
    return const_a * x * x + const_b * x + const_c

def plot_fit_result(params):
    plt.scatter(x_n_discs, y_error)
    x = np.array(range(65))
    plt.plot(x, f(x, *params))
    plt.show()

probcut_params_before = [1.0 for _ in range(3)]

popt, pcov = curve_fit(f, x_n_discs, y_error, np.array(probcut_params_before), sigma=weight, absolute_sigma=True)
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define null_move_pruning_' + chr(ord('a') + i), popt[i])
plot_fit_result(popt)
