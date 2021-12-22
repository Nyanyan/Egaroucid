from os import kill
import subprocess
from copy import deepcopy
from collections import deque
from othello_py import *

n_parallel = 14

ai_exe = [subprocess.Popen('./../src/egaroucid5.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) for _ in range(n_parallel)]

all_chars = [
    '!', '#', '$', '&', "'", '(', ')', '*', 
    '+', ',', '-', '.', '/', '0', '1', '2', 
    '3', '4', '5', '6', '7', '8', '9', ':', 
    ';', '<', '=', '>', '?', '@', 'A', 'B', 
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '[', ']', '^', '_', '`', 'a', 'b', 'c', 
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

with open('learned_data/book_change.txt', 'r') as f:
    data = f.read()
killer_boards = []
idx = 0
o = othello()
o.check_legal()
o.move(4, 5)
o.check_legal()
while idx < len(data):
    print('\r', idx, end='')
    if data[idx] == ' ':
        coord = all_chars.index(data[idx + 1])
        o.move(coord // hw, coord % hw)
        o.check_legal()
        killer_boards.append(o)
        idx += 2
        o = othello()
        o.check_legal()
        o.move(4, 5)
        o.check_legal()
    else:
        coord = all_chars.index(data[idx])
        o.move(coord // hw, coord % hw)
        o.check_legal()
        idx += 1

book = {}

with open('learned_data/book.txt') as f:
    data = f.read().splitlines()
for datum in data:
    board, val = datum.split()
    val = float(val)
    book[board] = val

val_threshold = 5
move_threshold = 0

que = deque([])

def start_calc_value(o, ai_num):
    grid_str = ''
    for i in range(hw):
        for j in range(hw):
            grid_str += '0' if o.grid[i][j] == 0 else '1' if o.grid[i][j] == 1 else '.'
        grid_str += '\n'
    #print(grid_str)
    if grid_str.replace('\n', '') in book:
        print('in book')
        return False
    flag = False
    for i in range(hw):
        for j in range(hw):
            flag = flag or (o.grid[i][j] == legal)
    if not flag:
        print('no legal')
        return False
    #print(o.player)
    #print(grid_str)
    ai_exe[ai_num].stdin.write((str(o.player) + '\n' + grid_str).encode('utf-8'))
    ai_exe[ai_num].stdin.flush()
    return True

def calc_value(o, ai_num):
    #print(grid_str)
    #if grid_str.replace('\n', '') in book:
    #    print('in book')
    #    return
    #ai_exe[ai_num].stdin.write((str(o.player) + '\n' + grid_str).encode('utf-8'))
    #ai_exe[ai_num].stdin.flush()
    _, _, val = [float(elem) for elem in ai_exe[ai_num].stdout.readline().decode().split()]
    if o.player == white:
        val = -val
    return val

for i in range(0, len(killer_boards), n_parallel):
    print(len(killer_boards), i)
    lst = []
    for j in range(i, min(len(killer_boards), i + n_parallel)):
        lst.append(start_calc_value(killer_boards[j], j - i))
    for j in range(i, min(len(killer_boards), i + n_parallel)):
        print(j)
        if lst[j - i]:
            val = calc_value(killer_boards[j], j - i)
            killer_boards[j].print_info()
            grid_str = ''
            for k in range(hw):
                for l in range(hw):
                    grid_str += '0' if killer_boards[j].grid[k][l] == 0 else '1' if killer_boards[j].grid[k][l] == 1 else '.'
            book[grid_str] = val
            with open('learned_data/book.txt', 'a') as f:
                f.write(grid_str + ' ' + str(val) + '\n')

for i in range(n_parallel):
    ai_exe[i].kill()