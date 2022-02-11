import matplotlib.pyplot as plt

data = []
with open('log.txt', 'r') as f:
    data = [[int(elem) for elem in line.split()] for line in f.read().splitlines()]
x = [datum[0] for datum in data]
y = [datum[1] for datum in data]
counts = [0 for _ in range(max(y) + 1)]
for i in range(len(x) - 1):
    counts[y[i]] += min(1, x[i + 1] - x[i]) #max(10, min(1, x[i + 1] - x[i]))
counts = [elem for elem in counts]
print(counts)
counts = [elem / sum(counts) for elem in counts]
print(counts)
plt.plot(x, y)
plt.show()