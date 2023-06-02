import matplotlib.pyplot as plt

with open('par.txt', 'r') as f:
    raw_data = f.read().splitlines()

x = []
y = []
for line in raw_data:
    x.append(float(line.split()[0]))
    y.append(float(line.split()[1]))
    if x[-1] >= 1450:
        break

plt.plot(x, y)
plt.grid()
plt.show()