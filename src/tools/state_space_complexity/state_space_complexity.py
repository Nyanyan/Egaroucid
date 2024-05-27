import matplotlib.pyplot as plt
from scipy.special import comb
import math

n_state_arr_exact = [1, 1, 3, 14, 60, 322, 1773, 10649, 67245, 434029, 2958586, 19786627, 137642461]

def nCr(n, r):
    return comb(n, r, exact=True)

def f0_upper_const():
    return (2 ** 4) * 3 ** 60

def f1(p):
    return (2 ** 4) * nCr(60, p) * (2 ** p)

def e(n):
    embedded_values = [
        0, 0, 0, 0, 1, 3, 5, 6, 9, 11, 
        12, 14, 15, 16, 18, 20, 21, 29, 30, 31, 
        32 
    ]
    if n < len(embedded_values):
        return embedded_values[n]
    return min(33, 64 - n)

def f2(p):
    if p < len(n_state_arr_exact):
        return n_state_arr_exact[p]
    prod = n_state_arr_exact[-1]
    for n in range(4 + len(n_state_arr_exact), 4 + p + 1):
        prod *= e(n)
    return (2 ** 4) * prod * (2 ** p)

p_arr = list(range(61))

n_state_arr_upper_const = [1 for _ in range(61)]
for p in range(61):
    n_state_arr_upper_const[p] = f0_upper_const()

n_state_arr = [1 for _ in range(61)]
for p in range(61):
    n_state_arr[p] = f1(p)

n_state_arr2 = [1 for _ in range(61)]
for p in range(61):
    n_state_arr2[p] = f2(p)

plt.plot(p_arr, n_state_arr_upper_const, label=r'$2^4 \times 3^{60}$')
plt.plot(p_arr, n_state_arr, label=r'$2^4 \times {}_{60}\mathrm{C}_p \times 2^p$')
plt.plot(p_arr, n_state_arr2, label=r'$2^4 \times \prod_{n=4}^{4+p}e(n) \times 2^p$')
plt.plot(range(len(n_state_arr_exact)), n_state_arr_exact, label='exact')
plt.yscale('log')
plt.grid()
plt.xlabel('n_moves')
plt.ylabel('n_states')
plt.legend()
plt.show()
