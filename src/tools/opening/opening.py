from othello_py import *

data = []
with open('data/openings_japanese.txt', 'r', encoding='utf-8') as f:
    for datum in f.read().splitlines():
        if len(datum.replace(' ', '').replace('　', '')):
            if datum.replace(' ', '').replace('　', '')[:2] == '//':
                continue
            n_spaces = 0
            for i in range(100):
                if datum[i] != ' ' and datum[i] != '　':
                    n_spaces = i
                    break
            datum = datum.replace(' ', '').replace('　', '')
            name, record = datum.split('=')
            print(n_spaces, name, record)
            data.append([n_spaces, name, record])

data.sort(key=lambda x: x[2])

joseki = {}
joseki_many = {}

for n_spaces, name, record in data:
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
        flag = not (s in joseki_many)
        if (not flag):
            if joseki_many[s][0] == n_spaces:
                joseki_many[s].append(name)
        else:
            joseki_many[s] = [n_spaces, name]
    #if not (s in joseki.keys()):
    #    joseki[s] = name
    joseki[s] = name

print(len(joseki))
print(len(joseki_many))
with open('output/openings.txt', 'w', encoding='utf-8') as f:
    for board in joseki.keys():
        f.write(board + ' ' + joseki[board] + '\n')
with open('output/openings_fork.txt', 'w', encoding='utf-8') as f:
    for board in joseki_many.keys():
        f.write(board + ' ' + '|'.join(joseki_many[board][1:]) + '\n')