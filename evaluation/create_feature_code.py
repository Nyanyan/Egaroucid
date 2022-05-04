s = '''A2, B2, C2, D2, E2, F2, G2, H2
B1, B2, B3, B4, B5, B6, B7, B8
A7, B7, C7, D7, E7, F7, G7, H7
G1, G2, G3, G4, G5, G6, G7, G8
A3, B3, C3, D3, E3, F3, G3, H3
C1, C2, C3, C4, C5, C6, C7, C8
A6, B6, C6, D6, E6, F6, G6, H6
F1, F2, F3, F4, F5, F6, F7, F8
A4, B4, C4, D4, E4, F4, G4, H4
D1, D2, D3, D4, D5, D6, D7, D8
A5, B5, C5, D5, E5, F5, G5, H5
E1, E2, E3, E4, E5, E6, E7, E8
D1, E2, F3, G4, H5
E1, D2, C3, B4, A5
A4, B5, C6, D7, E8
H4, G5, F6, E7, D8
C1, D2, E3, F4, G5, H6
F1, E2, D3, C4, B5, A6
A3, B4, C5, D6, E7, F8
H3, G4, F5, E6, D7, C8
B1, C2, D3, E4, F5, G6, H7
G1, F2, E3, D4, C5, B6, A7
A2, B3, C4, D5, E6, F7, G8
H2, G3, F4, E5, D6, C7, B8
A1, B2, C3, D4, E5, F6, G7, H8
H1, G2, F3, E4, D5, C6, B7, A8
B2, A1, B1, C1, D1, E1, F1, G1, H1, G2
B2, A1, A2, A3, A4, A5, A6, A7, A8, B7
B7, A8, B8, C8, D8, E8, F8, G8, H8, G7
G2, H1, H2, H3, H4, H5, H6, H7, H8, G7
A1, B1, C1, D1, A2, B2, C2, A3, B3, A4
H1, G1, F1, E1, H2, G2, F2, H3, G3, H4
A8, B8, C8, D8, A7, B7, C7, A6, B6, A5
H8, G8, F8, E8, H7, G7, F7, H6, G6, H5
A1, C1, D1, E1, F1, H1, C2, D2, E2, F2
A1, A3, A4, A5, A6, A8, B3, B4, B5, B6
A8, C8, D8, E8, F8, H8, C7, D7, E7, F7
H1, H3, H4, H5, H6, H8, G3, G4, G5, G6
A1, B2, C3, D4, B1, C2, D3, A2, B3, C4
H1, G2, F3, E4, G1, F2, E3, H2, G3, F4
A8, B7, C6, D5, B8, C7, D6, A7, B6, C5
H8, G7, F6, E5, G8, F7, E6, H7, G6, F5
A1, B1, C1, A2, B2, C2, A3, B3, C3
H1, G1, F1, H2, G2, F2, H3, G3, F3
A8, B8, C8, A7, B7, C7, A6, B6, C6
H8, G8, F8, H7, G7, F7, H6, G6, F6
C2, A1, B1, C1, D1, E1, F1, G1, H1, F2
B3, A1, A2, A3, A4, A5, A6, A7, A8, B6
C7, A8, B8, C8, D8, E8, F8, G8, H8, F7
G3, H1, H2, H3, H4, H5, H6, H7, H8, G6
A1, B1, C1, D1, E1, A2, B2, A3, A4, A5
H1, G1, F1, E1, D1, H2, G2, H3, H4, H5
A8, B8, C8, D8, E8, A7, B7, A6, A5, A4
H8, G8, F8, E8, D8, H7, G7, H6, H5, H4
A1, B1, A2, B2, C2, D2, B3, C3, B4, D4
H1, G1, H2, G2, F2, E2, G3, F3, G4, E4
A8, B8, A7, B7, C7, D7, B6, C6, B5, D5
H8, G8, H7, G7, F7, E7, G6, F6, G5, E5
A1, B1, A2, B2, C2, D2, E2, B3, B4, B5
H1, G1, H2, G2, F2, E2, D2, G3, G4, G5
A8, B8, A7, B7, C7, D7, E7, B6, B5, B4
H8, G8, H7, G7, F7, E7, D7, G6, G5, G4'''


ss = [line.split(', ') for line in s.splitlines()]

def cell_to_coord(cell):
    #cell = 63 - cell
    x = cell % 8
    y = cell // 8
    return chr(ord('A') + x) + str(y + 1)

def digit_space(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = ' ' + n
    return n

res = ''
for cell in range(64):
    coord = cell_to_coord(cell)
    tmp_arr = []
    for i in range(len(ss)):
        if coord in ss[i]:
            tmp = '{' + digit_space(i, 2) + ', P3' + str(len(ss[i]) - 1 - ss[i].index(coord)) + '}'
            tmp_arr.append(tmp)
    res_tmp = '{' + digit_space(len(tmp_arr), 2) + ', {' + ', '.join(tmp_arr) + '}}, // ' + coord
    res += res_tmp + '\n'
print(res)