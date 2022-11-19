import matplotlib.pyplot as plt

s = '1519 15963 779063 5112562 33783053' # ffo42@single thread

data = [int(elem) for elem in s.split()]
data.append(sum(data))

plt.plot(range(len(data)), data)
plt.show()