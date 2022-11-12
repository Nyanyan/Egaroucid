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

x_pass_scores = []
y_vals = []
for datum in raw_data:
    n_discs, depth, pass_score, val = [int(elem) for elem in datum.split()]
    y_vals.append(val)
    x_pass_scores.append(pass_score)

def f(x, const_a, const_b):
    return const_a * x / 64 + const_b

def plot_fit_result(params):
    plt.scatter(x_pass_scores, y_vals)
    x = np.array(range(-64, 65))
    plt.plot(x, f(x, *params))
    plt.show()

probcut_params_before = [1.0 for _ in range(2)]

popt, pcov = curve_fit(f, x_pass_scores, y_vals, np.array(probcut_params_before))
#popt = probcut_params_before
print([float(elem) for elem in popt])
for i in range(len(popt)):
    print('#define null_move_const_' + chr(ord('a') + i), popt[i])
plot_fit_result(popt)
