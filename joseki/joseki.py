from othello_py import *

with open('data/joseki_data.txt', 'r', encoding='utf-8') as f:
    data = [datum.split() for datum in f.read().splitlines()]

joseki = {}

for name, record in data:
    o = othello()
    o.check_legal()
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('A')
        if x >= hw:
            x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        o.move(y, x)
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        s = ''
        for i in range(hw):
            for j in range(hw):
                if o.grid[i][j] == 0:
                    s += '0'
                elif o.grid[i][j] == 1:
                    s += '1'
                else:
                    s += '.'
        if not (s in joseki.keys()):
            joseki[s] = name

print(len(joseki))
with open('learned_data/joseki.txt', 'w', encoding='utf-8') as f:
    for board in joseki.keys():
        f.write(board + ' ' + joseki[board] + '\n')