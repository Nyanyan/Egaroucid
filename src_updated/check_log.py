import matplotlib.pyplot as plt

data = []
with open('log.txt', 'r') as f:
    data = [[int(elem) for elem in line.split()] for line in f.read().splitlines()]
x = [datum[0] for datum in data]
y = [datum[1] for datum in data]
plt.plot(x, y)
plt.show()