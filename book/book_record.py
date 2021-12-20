from tqdm import trange, tqdm

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

def win_data(record, player):
    raw_record = ''
    for i in range(0, len(record), 2):
        p = int(record[i])
        raw_record += record[i + 1]
        if p == player:
            if not raw_record in record_all:
                record_all[raw_record] = [1, 1]
            else:
                record_all[raw_record][0] += 1
                record_all[raw_record][1] += 1


def lose_data(record, player):
    raw_record = ''
    for i in range(0, len(record), 2):
        p = int(record[i])
        raw_record += record[i + 1]
        if p == player:
            if not raw_record in record_all:
                record_all[raw_record] = [1, -1]
            else:
                record_all[raw_record][0] += 1
                record_all[raw_record][1] -= 1

def draw_data(record, player):
    raw_record = ''
    for i in range(0, len(record), 2):
        p = int(record[i])
        raw_record += record[i + 1]
        if p == player:
            if not raw_record in record_all:
                record_all[raw_record] = [1, 0]
            else:
                record_all[raw_record][0] += 1
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

for i in trange(231):
    try:
        with open('data/' + digit(i, 7) + '.txt', 'r') as f:
            records = f.read().splitlines()
        for datum in records:
            record, score = datum.split()
            score = int(score)
            if score == 0:
                draw_data(record, 0)
                draw_data(record, 1)
            elif score > 0:
                win_data(record, 0)
                lose_data(record, 1)
                black_win += 1
            else:
                lose_data(record, 0)
                win_data(record, 1)
                white_win += 1
    except:
        print('cannot open', i)
        continue
print(len(record_all))


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


book = {}

max_ln = 45

num_threshold1 = 6

inf = 100000000

def calc_value(r):
    if translate(r) in hand_book:
        return inf
    if r in record_all:
        if record_all[r][0] < num_threshold1:
            return -inf
        val = record_all[r][1] / record_all[r][0]
        val += 0.01 * record_all[r][0]
        return val
    return -inf

def create_book(record):
    if len(record) > max_ln:
        return
    policy = -1
    max_val = -inf
    for i in range(hw2):
        r = record + all_chars[i]
        val = calc_value(r)
        if max_val < val:
            max_val = val
            policy = i
    if policy != -1:
        book[record] = [all_chars[policy]]
        for i in range(hw2):
            r = record + all_chars[policy] + all_chars[i]
            if r in record_all:
                create_book(r)

'''
book = {}
create_book(all_chars[37])
print(len(book))
create_book(all_chars[37] + all_chars[43])
create_book(all_chars[37] + all_chars[45])
create_book(all_chars[37] + all_chars[29])
print(len(book))
if (input('sure?: ') == 'yes'):
    with open('learned_data/book.txt', 'w') as f:
        for record in book.keys():
            f.write(record[1:] + ' ' + book[record][0])


val_threshold = 0.0

def create_book_change(record):
    if len(record) > max_ln:
        return
    max_val = -inf
    max_val_move = -1
    flag = False
    for i in range(hw2):
        r = record + all_chars[i]
        val = calc_value(r)
        if val >= val_threshold:
            flag = True
            if record in book:
                book[record].append(all_chars[i])
            else:
                book[record] = [all_chars[i]]
            for j in range(hw2):
                rr = r + all_chars[j]
                if rr in record_all:
                    create_book(rr)
        elif val > max_val:
            max_val = val
            max_val_move = i
    if not flag:
        r = record + all_chars[max_val_move]
        if record in book:
            book[record].append(all_chars[max_val_move])
        else:
            book[record] = [all_chars[max_val_move]]
        for j in range(hw2):
            rr = r + all_chars[j]
            if rr in record_all:
                create_book(rr)

book = {}
create_book_change(all_chars[37])
print(len(book))
create_book_change(all_chars[37] + all_chars[43])
create_book_change(all_chars[37] + all_chars[45])
create_book_change(all_chars[37] + all_chars[29])
print(len(book))
if (input('sure?: ') == 'yes'):
    with open('learned_data/book_change.txt', 'w') as f:
        for record in book.keys():
            for elem in book[record]:
                f.write(record[1:] + ' ' + elem)
'''
big_book_val_threshold = 0.2

def create_big_book(record):
    for i in range(64):
        if calc_value(record + all_chars[i]) >= big_book_val_threshold:
            if record in book:
                book[record].append(all_chars[i])
            else:
                book[record] = [all_chars[i]]

book = {}
for key in tqdm(record_all.keys()):
    create_big_book(key)
print(len(book))
if (input('sure?: ') == 'yes'):
    with open('learned_data/book_big.txt', 'w') as f:
        for record in book.keys():
            for elem in book[record]:
                f.write(record[1:] + ' ' + elem)