s = '''
    // 0 hv1
    {8, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1}},
    {8, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8}},
    {8, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8}},
    {8, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1}},

    // 1 hv2
...
'''

s = input('input definition:\n')

ss = s.splitlines()
s = ''
for sss in ss:
    sss = sss.replace('    ', '')
    if len(sss):
        if sss[0] == '{':
            s += sss + '\n'

s = s.replace('{10, {', '').replace('{9, {', '').replace('{8, {', '').replace('{8,  {', '').replace('{7,  {', '').replace('{7, {', '').replace('{6, {', '').replace('{5, {', '').replace('{4, {', '').replace('{0, {', '').replace('\n\n', '\n').replace('}', '').replace('    ', '')
for num in reversed(range(100)):
    s = s.replace(', // ' + str(num), '')
    s = s.replace('  // ' + str(num), '')
    ss = '// ' + str(num) + ' '
    idx = s.find(ss)
    if idx >= 0:
        end = idx
        for i in range(idx, len(s)):
            if s[i] == '\n':
                end = i
                break
        s = s.replace(s[idx:end + 1], '')

s = s.replace(',', '')
s = s.replace('COORD_NO', '')

print(s)

ss = [line.split() for line in s.splitlines()]

def cell_to_coord(cell):
    cell = 63 - cell
    x = cell % 8
    y = cell // 8
    return 'COORD_' + chr(ord('A') + x) + str(y + 1)

def digit_space(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = ' ' + n
    return n

#for cell in range(64):
#    print('#define COORD_' + cell_to_coord(cell) + ' ' + str(cell))

res = ''
for cell in range(64):
    coord = cell_to_coord(cell)
    tmp_arr = []
    for i in range(len(ss)):
        if coord in ss[i]:
            ii = i
            #if i >= 26:
            #    ii += 2
            tmp = '{' + digit_space(ii, 2) + ', P3' + str(len(ss[i]) - 1 - ss[i].index(coord)) + '}'
            tmp_arr.append(tmp)
    len_main = len(tmp_arr)
    for i in range(16 - len_main):
        tmp = '{ 0, PNO}'
        tmp_arr.append(tmp)
    res_tmp = '{' + digit_space(len_main, 2) + ', {' + ', '.join(tmp_arr) + '}}, // ' + coord
    res += res_tmp + '\n'
print(res)
