import matplotlib.pyplot as plt

data = []
with open('log.txt', 'r') as f:
    data = [[int(elem) for elem in line.split()] for line in f.read().splitlines()]
x = []
y = []
div = []
div2 = []
for xx, yy in data:
    if yy >= 0:
        x.append(xx)
        y.append(yy)
    elif yy == -1:
        div.append(xx)
    elif yy == -2:
        div2.append(xx)
counts = [0 for _ in range(max(y) + 1)]
for i in range(len(x) - 1):
    counts[y[i]] += 1 #min(1, x[i + 1] - x[i]) #max(10, min(1, x[i + 1] - x[i]))
counts = [elem for elem in counts]
print(counts)
counts = [elem / sum(counts) for elem in counts]
print(counts)
plt.plot(x, y)
plt.scatter(div, [-0.1 for _ in div])
plt.scatter(div2, [-0.2 for _ in div2])
plt.show()