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

def yx_to_idx(y, x):
    return y * 8 + x

def idx_to_yx(yx):
    return yx // 8, yx % 8

def rotate_all(yx_list):
    res = [set([]) for _ in range(8)]
    for yx in yx_list:
        y, x = idx_to_yx(yx)
        res[0].add(yx_to_idx(y, x))
        res[1].add(yx_to_idx(y, 7 - x))
        res[2].add(yx_to_idx(7 - y, x))
        res[3].add(yx_to_idx(7 - y, 7 - x))
        res[4].add(yx_to_idx(x, y))
        res[5].add(yx_to_idx(x, 7 - y))
        res[6].add(yx_to_idx(7 - x, y))
        res[7].add(yx_to_idx(7 - x, 7 - y))
    return res

feature_list = []
pattern_list = []
for s in s_arr:
    coords = s.split(',')
    yx_list = []
    for coord in coords:
        try:
            x = ord(coord[-2]) - ord('A')
            y = int(coord[-1]) - 1
            yx_list.append(yx_to_idx(y, x))
        except:
            pass
    feature_list.append(yx_list)
    rotated_lists = rotate_all(yx_list)
    ignore_this_pattern = False
    for rotated in rotated_lists:
        for pattern in pattern_list:
            if rotated == set(pattern):
                ignore_this_pattern = True
    if not ignore_this_pattern:
        pattern_list.append(yx_list)

print('n_features', len(feature_list))
print('n_patterns', len(pattern_list))

# feature image
fig = plt.figure(figsize=(10, 10))
for idx, yx_list in enumerate(feature_list):
    ax = fig.add_subplot((len(feature_list) + 7) // 8, 8, idx + 1)
    ax.set_aspect("equal", adjustable="box")
    for x in range(9):
        p = plt.Line2D(xdata=(x, x), ydata=(0, 8), color='k', linewidth= 2 if x == 0 or x == 8 else 1)
        ax.add_line(p)
    for y in range(9):
        p = plt.Line2D(xdata=(0, 8), ydata=(y, y), color='k', linewidth= 2 if y == 0 or y == 8 else 1)
        ax.add_line(p)
    num = 0
    for yx in yx_list:
        y, x = idx_to_yx(yx)
        p = Rectangle(xy=(x, 7 - y), height=1, width=1, facecolor='k')
        ax.add_patch(p)
        ax.text(x + 0.5, 7 - y + 0.5, num, color='w', fontsize=7, horizontalalignment='center', verticalalignment='center')
        num += 1
    ax.set_xlim((0, 8))
    ax.set_ylim((0, 8))
    ax.axis("off")
plt.tight_layout()
plt.show()
plt.clf()

# pattern image
fig = plt.figure(figsize=(10, 10))
for idx, yx_list in enumerate(pattern_list):
    if len(pattern_list) % 4 == 0 or idx < len(pattern_list) - 4:
        ax = fig.add_subplot((len(pattern_list) + 3) // 4, 4, idx + 1)
    elif len(pattern_list) % 4 == 2:
        if idx < len(pattern_list) - 2:
            ax = fig.add_subplot((len(pattern_list) + 3) // 4, 4, idx + 1)
        else:
            ax = fig.add_subplot((len(pattern_list) + 3) // 4, 4, idx + 2)
    ax.set_aspect("equal", adjustable="box")
    for x in range(9):
        p = plt.Line2D(xdata=(x, x), ydata=(0, 8), color='k', linewidth= 3 if x == 0 or x == 8 else 1)
        ax.add_line(p)
    for y in range(9):
        p = plt.Line2D(xdata=(0, 8), ydata=(y, y), color='k', linewidth= 3 if y == 0 or y == 8 else 1)
        ax.add_line(p)
    num = 0
    for yx in yx_list:
        y, x = idx_to_yx(yx)
        p = Rectangle(xy=(x, 7 - y), height=1, width=1, facecolor='k')
        ax.add_patch(p)
        #ax.text(x + 0.5, 7 - y + 0.5, num, color='w', fontsize=7, horizontalalignment='center', verticalalignment='center')
        num += 1
    ax.set_xlim((0, 8))
    ax.set_ylim((0, 8))
    ax.axis("off")
plt.tight_layout()
plt.show()
plt.clf()

# heatmap
fig = plt.figure(figsize=(10, 10))
duplication = [[0 for _ in range(8)] for _ in range(8)]
for idx, yx_list in enumerate(feature_list):
    for yx in yx_list:
        y, x = idx_to_yx(yx)
        duplication[y][x] += 1
max_dup = 0
min_dup = 10000
for arr in duplication:
    max_dup = max(max_dup, max(arr))
    min_dup = min(min_dup, min(arr))
    print(arr)
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal", adjustable="box")
heatmap = ax.pcolor(np.flipud(np.array(duplication)), cmap=plt.cm.Blues)
for y in range(8):
    for x in range(8):
        if duplication[y][x] < (max_dup + min_dup) // 2:
            color = 'black'
        else:
            color = 'white'
        ax.text(x + 0.5, 7 - y + 0.5, duplication[y][x], color=color, fontsize=30, horizontalalignment='center', verticalalignment='center')
ax.set_xlim((0, 8))
ax.set_ylim((0, 8))
ax.axis("off")
plt.tight_layout()
plt.show()
plt.clf()
