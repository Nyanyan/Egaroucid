import sys

dr = sys.argv[1]

N = 62

TAB = '    '

def fillspace(s, d):
    return ' ' * (d - len(s)) + s

for i in range(N):
    file = dr + '/' + str(i) + '.txt'
    with open(file, 'r') as f:
        data = f.read().splitlines()
    data2 = []
    for i in range(20):
        lst = []
        for j in range(8):
            lst.append(data[i * 8 + j])
        data2.append(lst)
    print('{')
    for i in range(20):
        data_join = ''
        if i % 5 == 0:
            data_join += TAB
        data_join += '{' + ', '.join([fillspace(elem, 4) for elem in data2[i]]) + '}'
        if i < 19:
            data_join += ', '
        print(data_join, end='')
        if i % 5 == 4:
            print('')
    print('},')