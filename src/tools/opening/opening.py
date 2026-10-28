from othello_py import *
from pathlib import Path


def opening_board_key(board):
    # The same four symmetries as the GUI: preserve the initial black/white discs.
    diagonal = ''.join(board[x * hw + y] for y in range(hw) for x in range(hw))
    return min(board, board[::-1], diagonal, diagonal[::-1])


data = []
with Path(__file__).with_name('openings_japanese.txt').open(encoding='utf-8') as f:
    for datum in f.read().splitlines():
        if len(datum.replace(' ', '').replace('　', '')):
            if datum.replace(' ', '').replace('　', '')[:2] == '//':
                continue
            datum = datum.replace(' =', '=').replace('= ', '=').replace('　', '')
            name, record = datum.split('=')
            record = record.lower()
            print(name, record)
            data.append([name, record, []])

data.sort(key=lambda x: len(x[1]))

for i in range(len(data)):
    name, record, _ = data[i]
    for cname, crecord, _ in data:
        if name != cname and record == crecord[:len(record)]:
            data[i][2].append(cname)

for i in range(len(data)):
    name, record, children = data[i]
    n_children = set(children)
    for child in children:
        for cname, crecord, cchildren in data:
            if child == cname:
                n_children -= set(cchildren)
    n_children = sorted(n_children)
    data[i][2] = n_children
    if record == 'F5F6':
        print(name, record, children, n_children)
    #print(name, record, n_children)


joseki = {}
joseki_many = {}

for name, record, children in data:
    o = othello()
    o.check_legal()
    for i in range(0, len(record), 2):
        x = ord(record[i]) - ord('a')
        y = int(record[i + 1]) - 1
        if not o.move(y, x):
            raise ValueError(f'Illegal move in {name}: {record[:i + 2]}')
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
        if not s in joseki_many:
            joseki_many[s] = [name]
        else:
            joseki_many[s].append(name)
    joseki[s] = name

for board in joseki_many.keys():
    joseki_children = set(joseki_many[board])
    for child in joseki_many[board]:
        for name, record, children in data:
            if name == child:
                joseki_children -= set(children)
    joseki_many[board] = sorted(joseki_children)

print(len(joseki))
print(len(joseki_many))
Path('output').mkdir(exist_ok=True)
with open('output/openings.txt', 'w', encoding='utf-8') as f:
    written = {}
    for board in joseki.keys():
        key = opening_board_key(board)
        if key in written:
            if written[key] != joseki[board]:
                raise ValueError(f'Conflicting names for symmetric boards: {written[key]} / {joseki[board]}')
            continue
        written[key] = joseki[board]
        f.write(board + ' ' + joseki[board] + '\n')
with open('output/openings_fork.txt', 'w', encoding='utf-8') as f:
    for board in joseki_many.keys():
        f.write(board + ' ' + '|'.join(joseki_many[board]) + '\n')
