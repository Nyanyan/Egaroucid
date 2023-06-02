import matplotlib.pyplot as plt

with open('par_20230602_4.txt', 'r') as f:
    raw_data = f.read().splitlines()

x = []
y = []
for line in raw_data:
    x.append(float(line.split()[0]))
    y.append(float(line.split()[1]))
    if x[-1] >= 1150:
        break

print('avg', sum(y) / len(y))

plt.plot(x, y)
plt.minorticks_on()
plt.grid()
plt.show()