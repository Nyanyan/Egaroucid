import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


while True:
    s = input()
    if s == '':
        break
    if s[0] != 'C':
        coords = s.split()[1:]
    else:
        coords = s.split()
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for x in range(9):
        p = plt.Line2D(xdata=(x, x), ydata=(0, 8), color='k', linewidth=2)
        ax.add_line(p)
    for y in range(9):
        p = plt.Line2D(xdata=(0, 8), ydata=(y, y), color='k', linewidth=2)
        ax.add_line(p)
    for coord in coords:
        coord = coord.replace(',', '')
        x = ord(coord[-2]) - ord('A')
        y = int(coord[-1]) - 1
        print(coord, x, y)
        p = Rectangle(xy=(7 - x, 7 - y), height=1, width=1, facecolor='k')
        ax.add_patch(p)
    plt.xlim((0, 8))
    plt.ylim((0, 8))
    plt.tight_layout()
    plt.show()
