import matplotlib.pyplot as plt

# ffo42 @ single thread
'''
ss = ['1519 15963 779044 5112464 3808551 7515956 9511446 12958575', # without stab cut optimize
      '1519 15963 774612 4759788 3454902 6707636 8534342 11676201' # stab cut optimize
]
'''
ss = ['0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 52 261 811 1635 4160 8743 15475 8026 23361 22186 73038 90338 253334 320173 853412 1065070 2840687 3455052 6707866 8534715 11676678'
]
#labels = ['mid', 'mid_last', 'end', 'end_fast', 'last4', 'last3', 'last2', 'last1', 'last4-1', 'all']

for s in ss:
    data = [int(elem) for elem in s.split()]
    l = sum(data[-4:])
    a = sum(data)
    #data.append(l)
    #data.append(a)

    plt.bar(range(len(data)), data)
labels = [str(elem) for elem in range(len(data))]
plt.xticks(range(len(data)), labels, fontsize=8)
plt.show()