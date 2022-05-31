import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random, randrange
import numpy as np
from scipy.optimize import curve_fit
from math import exp

s = '''#40 depth 20 value 38 policy a2 nodes 21704057 time 169 nps 128426372
#41 depth 22 value 0 policy h4 nodes 37837262 time 312 nps 121273275
#42 depth 22 value 6 policy g2 nodes 52567779 time 416 nps 126364853
#43 depth 23 value -12 policy c7 nodes 99650314 time 711 nps 140155153
#44 depth 23 value -14 policy d2 nodes 14090661 time 255 nps 55257494
#45 depth 24 value 6 policy b2 nodes 651509880 time 3439 nps 189447478
#46 depth 24 value -8 policy b3 nodes 101776083 time 1027 nps 99100372
#47 depth 25 value 4 policy g2 nodes 27548765 time 439 nps 62753451
#48 depth 25 value 28 policy f6 nodes 172289411 time 1794 nps 96036460
#49 depth 26 value 16 policy e1 nodes 521842137 time 4820 nps 108266003
#50 depth 26 value 10 policy d8 nodes 1307711930 time 10151 nps 128825921
#51 depth 27 value 6 policy e2 nodes 320536084 time 3273 nps 97933420
#52 depth 27 value 0 policy a3 nodes 503956547 time 4654 nps 108284603
#53 depth 28 value -2 policy d8 nodes 5616606395 time 35521 nps 158120728
#54 depth 28 value -2 policy c7 nodes 9965457602 time 48561 nps 205215246
#55 depth 29 value 0 policy g6 nodes 23482530340 time 150200 nps 156341746
#56 depth 29 value 2 policy h5 nodes 1373034500 time 13802 nps 99480836
#57 depth 30 value -10 policy a6 nodes 1882921932 time 16681 nps 112878240
#58 depth 30 value 4 policy g1 nodes 2302712191 time 18231 nps 126307508
'''

points = [[int(line.split()[2]), float(line.split()[10]) / 1000.0 / 60.0] for line in s.splitlines()]

print(points)

points_x = [arr[0] for arr in points]
points_y = [arr[1] for arr in points]

def f(x, a):
    return a * np.exp(x) - a

params_before = [1.0]

popt, pcov = curve_fit(f, points_x, points_y, np.array(params_before))
print([float(elem) for elem in popt])
plt.scatter(points_x, points_y)
pred_x = range(40)
pred_y = [f(x, popt[0]) for x in pred_x]
print(pred_y[38])
plt.plot(pred_x, pred_y)
plt.show()