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

for i in trange(236):
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

'''
inf = 1000000000
with open('third_party/records4.txt', 'r') as f:
    records = f.read().splitlines()
for record in records:
    record_proc = ''
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        record_proc += all_chars[y * 8 + x]
        if not record_proc in record_all:
            record_all[record_proc] = [100, inf]
        else:
            record_all[record_proc][0] += 100
            record_all[record_proc][1] += inf
print(len(record_all))
print(black_win, white_win)

hand_book = set()
with open('third_party/records5.txt', 'r') as f:
    dat_handbook = f.read().splitlines()
for elem in dat_handbook:
    hand_book.add(elem)

with open('third_party/records6.txt', 'r') as f:
    records6 = f.read().splitlines()
print(len(records6))
for record in records6:
    record_proc = ''
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        record_proc += all_chars[y * 8 + x]
        if i % 4 == 0:
            if not record_proc in record_all:
                record_all[record_proc] = [1000, inf * 1000]
            else:
                record_all[record_proc][0] += 1000
                record_all[record_proc][1] += inf * 1000
        else:
            if not record_proc in record_all:
                record_all[record_proc] = [0, 0]

with open('third_party/records7.txt', 'r') as f:
    records7 = f.read().splitlines()
print(len(records7))
for record in records7:
    record_proc = ''
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        record_proc += all_chars[y * 8 + x]
        if i % 4 == 2:
            if not record_proc in record_all:
                record_all[record_proc] = [1000, inf * 1000]
            else:
                record_all[record_proc][0] += 1000
                record_all[record_proc][1] += inf * 1000
        else:
            if not record_proc in record_all:
                record_all[record_proc] = [0, 0]

with open('third_party/records8.txt', 'r') as f:
    records8 = f.read().splitlines()
print(len(records8))
for record in records8:
    record_proc = ''
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        record_proc += all_chars[y * 8 + x]
        if not record_proc in record_all:
            record_all[record_proc] = [1000, inf * 1000]
        else:
            record_all[record_proc][0] += 1000
            record_all[record_proc][1] += inf * 1000
'''

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
        return val
    return -inf

def create_board(record):
    o = othello()
    for i in range(len(record)):
        if not o.check_legal():
            o.player = 1 - o.player
        coord = char_coord[record[i]]
        if not o.move(coord // hw, coord % hw):
            return ''
    res = ''
    for y in range(hw):
        for x in range(hw):
            if o.grid[y][x] == 0:
                res += '0'
            elif o.grid[y][x] == 1:
                res += '1'
            else:
                res += '.'
    return res

def create_book(record):
    for i in range(64):
        val = calc_value(record + all_chars[i])
        if val != -inf:
            board = create_board(record + all_chars[i])
            if board != '':
                if not (board in book.keys()):
                    book[board] = val
                    create_book(record + all_chars[i])

book = {}
create_book('')
print(len(book))
#if (input('sure?: ') == 'yes'):
with open('learned_data/book_record.txt', 'w') as f:
    for board in book.keys():
        f.write(board + ' ' + str(round(book[board])) + '\n')