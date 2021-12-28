from tqdm import trange, tqdm
from othello_py import *

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

xs = 'abcdefgh'
ys = '12345678'

char_translate = {}
for i in range(64):
    char_translate[all_chars[i]] = xs[i % 8] + ys[i // 8]

char_coord = {}
for i in range(64):
    char_coord[all_chars[i]] = i

def translate(r):
    return ''.join([char_translate[i] for i in r])

hw2 = 64

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

record_all = {}

def append_data(record, score):
    raw_record = ''
    for i in range(0, len(record), 2):
        raw_record += record[i + 1]
        if not raw_record in record_all:
            record_all[raw_record] = [1, score]
        else:
            record_all[raw_record][0] += 1
            record_all[raw_record][1] += score

'''
rec = ''
rec_num = ''
for i in range(0, len(rec), 2):
    x = ord(rec[i]) - ord('a')
    y = int(rec[i + 1]) - 1
    yy = 7 - x
    xx = 7 - y
    rec_num += all_chars[yy * 8 + xx]
print(translate(rec_num))
exit()
'''
black_win = 0
white_win = 0

for i in trange(2, 124):
    with open('data/' + digit(i, 7) + '.txt', 'r') as f:
        records = f.read().splitlines()
    for datum in records:
        record, score = datum.split()
        score = int(score)
        append_data(record, score)
        if score > 0:
            black_win += 1
        elif score < 0:
            white_win += 1
print(len(record_all))

book = {}

num_threshold1 = 10

inf = 100000000

def calc_value(r):
    #if translate(r) in hand_book:
    #    return inf
    if r in record_all:
        if record_all[r][0] < num_threshold1:
            return -inf
        val = record_all[r][1] / record_all[r][0]
        #val += 0.01 * record_all[r][0]
        return round(val)
    return -inf

def create_board(record):
    res = []
    o = othello()
    for i in range(len(record)):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        coord = char_coord[record[i]]
        if not o.move(coord // hw, coord % hw):
            return []
    board = ''
    for y in range(hw):
        for x in range(hw):
            if o.grid[y][x] == 0:
                board += '0'
            elif o.grid[y][x] == 1:
                board += '1'
            else:
                board += '.'
    res.append(board)
    o = othello()
    for i in range(len(record)):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        y = char_coord[record[i]] // hw
        x = char_coord[record[i]] % hw
        coord = (7 - y) * hw + (7 - x)
        if not o.move(coord // hw, coord % hw):
            return []
    board = ''
    for y in range(hw):
        for x in range(hw):
            if o.grid[y][x] == 0:
                board += '0'
            elif o.grid[y][x] == 1:
                board += '1'
            else:
                board += '.'
    res.append(board)
    o = othello()
    for i in range(len(record)):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        y = char_coord[record[i]] // hw
        x = char_coord[record[i]] % hw
        coord = (7 - x) * hw + (7 - y)
        if not o.move(coord // hw, coord % hw):
            return []
    board = ''
    for y in range(hw):
        for x in range(hw):
            if o.grid[y][x] == 0:
                board += '0'
            elif o.grid[y][x] == 1:
                board += '1'
            else:
                board += '.'
    res.append(board)
    o = othello()
    for i in range(len(record)):
        if not o.check_legal():
            o.player = 1 - o.player
            o.check_legal()
        y = char_coord[record[i]] // hw
        x = char_coord[record[i]] % hw
        coord = x * hw + y
        if not o.move(coord // hw, coord % hw):
            return []
    board = ''
    for y in range(hw):
        for x in range(hw):
            if o.grid[y][x] == 0:
                board += '0'
            elif o.grid[y][x] == 1:
                board += '1'
            else:
                board += '.'
    res.append(board)
    return res

def create_book(record):
    for i in range(64):
        val = calc_value(record + all_chars[i])
        if val != -inf:
            boards = create_board(record + all_chars[i])
            if boards:
                contain = False
                for board in boards:
                    if board in book.keys():
                        contain = True
                if not contain:
                    book[board] = val
                create_book(record + all_chars[i])

book = {}
with open('learned_data/before_book.txt', 'r') as f:
    data = f.read().splitlines()
for datum in data:
    board, value = datum.split()
    value = int(value)
    book[board] = value
print(len(data))

create_book('')
print(len(book))
#if (input('sure?: ') == 'yes'):
with open('learned_data/book.txt', 'w') as f:
    for board in book.keys():
        f.write(board + ' ' + str(round(book[board])) + '\n')