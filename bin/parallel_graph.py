import matplotlib.pyplot as plt
import sys

file = sys.argv[1]

with open(file, 'r') as f:
    raw_data = f.read().splitlines()

x = []
y = []
for line in raw_data:
    x_elem = float(line.split()[0])
    if len(x) == 0 or x[-1] != x_elem:
        x.append(x_elem)
        y.append(float(line.split()[1]))
    if len(y) >= 50 and sum(y[-50:]) == max(y) * 50:
        x = x[:-50]
        y = y[:-50]
        break

print('avg', sum(y) / len(y))

plt.plot(x, y)
plt.minorticks_on()
plt.grid()
plt.show()
