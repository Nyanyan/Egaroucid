import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

''' # input format

    // 14 fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}},
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}},
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}},
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}},

    // 15 anvil
    {10, {COORD_C6, COORD_D6, COORD_D7, COORD_D8, COORD_C8, COORD_F8, COORD_E8, COORD_E7, COORD_E6, COORD_F6}},
    {10, {COORD_C3, COORD_C4, COORD_B4, COORD_A4, COORD_A3, COORD_A6, COORD_A5, COORD_B5, COORD_C5, COORD_C6}},
    {10, {COORD_F3, COORD_E3, COORD_E2, COORD_E1, COORD_F1, COORD_C1, COORD_D1, COORD_D2, COORD_D3, COORD_C3}},
    {10, {COORD_F6, COORD_F5, COORD_G5, COORD_H5, COORD_H6, COORD_H3, COORD_H4, COORD_G4, COORD_F4, COORD_F3}}
    quit

'''

s_arr = []
while True:
    s = input()
    for i in range(30):
        s = s.replace('{' + str(i) + ',', '')
    s = s.replace('{', '')
    s = s.replace('},', '')
    s = s.replace('}', '')
    s = s.replace('//', ', //')
    s = s.replace(' ', '')
    if s == 'quit':
        break
    if s == '' or s.find('COORD') == -1:
        continue
    #print(s)
    s_arr.append(s)


print('n_features', len(s_arr))

duplication = [[0 for _ in range(8)] for _ in range(8)]
fig = plt.figure(figsize=(11, 10))
for idx, s in enumerate(s_arr):
    coords = s.split(',')
    ax = fig.add_subplot(9, 8, idx + 1)
    ax.set_aspect("equal", adjustable="box")
    for x in range(9):
        p = plt.Line2D(xdata=(x, x), ydata=(0, 8), color='k', linewidth=1)
        ax.add_line(p)
    for y in range(9):
        p = plt.Line2D(xdata=(0, 8), ydata=(y, y), color='k', linewidth=1)
        ax.add_line(p)
    num = 0
    for coord in coords:
        try:
            x = ord(coord[-2]) - ord('A')
            y = int(coord[-1]) - 1
            #print(coord, x, y)
            p = Rectangle(xy=(x, 7 - y), height=1, width=1, facecolor='k')
            ax.add_patch(p)
            ax.text(x + 0.5, 7 - y + 0.5, num, color='w', fontsize=7, horizontalalignment='center', verticalalignment='center')
            num += 1
            duplication[y][x] += 1
            '''
            duplication[7 - y][x] += 1
            duplication[y][7 - x] += 1
            duplication[7 - y][7 - x] += 1
            duplication[x][y] += 1
            duplication[7 - x][y] += 1
            duplication[x][7 - y] += 1
            duplication[7 - x][7 - y] += 1
            '''
        except:
            pass
    ax.set_xlim((0, 8))
    ax.set_ylim((0, 8))
    ax.axis("off")

for arr in duplication:
    print(arr)
ax = fig.add_subplot(9, 8, len(s_arr) + 1)
ax.set_aspect("equal", adjustable="box")
heatmap = ax.pcolor(np.flipud(np.array(duplication)), cmap=plt.cm.Blues)
for y in range(8):
    for x in range(8):
        ax.text(x + 0.5, 7 - y + 0.5, duplication[y][x], color='w', fontsize=7, horizontalalignment='center', verticalalignment='center')
ax.set_xlim((0, 8))
ax.set_ylim((0, 8))
ax.axis("off")
plt.tight_layout()
plt.show()
