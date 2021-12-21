import subprocess
from copy import deepcopy
from collections import deque
from othello_py import *

ai_exe = subprocess.Popen('./../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

book = {}

with open('learned_data/book.txt') as f:
    data = f.read().splitlines()
for datum in data:
    board, val = datum.split()
    val = float(val)
    book[board] = val

val_threshold = 7
move_threshold = 0

que = deque([])

def calc_value(o):
    o.print_info()
    grid_str = ''
    for i in range(hw):
        for j in range(hw):
            grid_str += '0' if o.grid[i][j] == 0 else '1' if o.grid[i][j] == 1 else '.'
        grid_str += '\n'
    #print(grid_str)
    if grid_str.replace('\n', '') in book:
        val = book[grid_str.replace('\n', '')]
    else:
        ai_exe.stdin.write((str(o.player) + '\n' + grid_str).encode('utf-8'))
        ai_exe.stdin.flush()
        _, _, val = [float(elem) for elem in ai_exe.stdout.readline().decode().split()]
        if o.player == white:
            val = -val
        print(val)
        book[grid_str.replace('\n', '')] = val
        with open('learned_data/book.txt', 'a') as f:
            f.write(grid_str.replace('\n', '') + ' ' + str(val) + '\n')
    if sum(o.n_stones) > move_threshold + 4 and abs(val) > abs(val_threshold):
        return
    for i in range(hw):
        for j in range(hw):
            if o.grid[i][j] == legal:
                next_o = deepcopy(o)
                next_o.move(i, j)
                if not next_o.check_legal():
                    next_o.player = 1 - next_o.player
                    if not next_o.check_legal():
                        continue
                que.append(next_o)

o = othello()
o.check_legal()
o.move(4, 5)
o.check_legal()
que.append(o)
while que:
    calc_value(que.popleft())