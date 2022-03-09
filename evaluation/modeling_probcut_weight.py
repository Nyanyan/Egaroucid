from math import exp
import matplotlib.pyplot as plt

s = 0.9999999
e = 0.4


def f(x):
    return pow(s, 1 - x / 60) * pow(e, x / 60)

arr = []
for i in range(61):
    arr.append(f(i))

for i in range(61):
    print(arr[i], end=', ')
    if i % 10 == 9:
        print('')
print('')
plt.plot(range(61), arr)
plt.show()