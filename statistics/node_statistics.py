import matplotlib.pyplot as plt

s = '1519 15963 779044 5112464 3808551 7515956 9511446 12958575' # ffo42@single thread
labels = ['mid', 'mid_last', 'end', 'end_fast', 'last4', 'last3', 'last2', 'last1', 'last4-1', 'all']

data = [int(elem) for elem in s.split()]
l = sum(data[-4:])
a = sum(data)
data.append(l)
data.append(a)

plt.bar(range(len(data)), data)
plt.xticks(range(len(data)), labels, fontsize=8)
plt.show()