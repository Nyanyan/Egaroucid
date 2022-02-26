import matplotlib.pyplot as plt
import sys

with open(sys.argv[1], 'r') as f:
    data = f.read().splitlines()

graph_x = list(range(-64, 65, 2))
graph_y = [0 for _ in range(65)]
for datum in data:
    score, num = [int(elem) for elem in datum.split()]
    graph_y[(score + 64) // 2] = num
plt.bar(graph_x, graph_y, width=2.0)
plt.show()