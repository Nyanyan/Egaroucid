#s = 'COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4, COORD_B4'
s = input()

s = s.split(', ')

N_SYMMETRY = 8

res = [[] for _ in range(N_SYMMETRY)]

def xy_to_coord(x, y):
    return 'COORD_' + chr(ord('A') + x) + str(y + 1)

for elem in s:
    if elem[-2:] == 'NO':
        for i in range(N_SYMMETRY):
            res[i].append('COORD_NO')
    else:
        x = ord(elem[-2]) - ord('A')
        y = int(elem[-1]) - 1
        
        if N_SYMMETRY == 8:
            res[0].append(xy_to_coord(x, y))
            res[1].append(xy_to_coord(7 - y, 7 - x)) # black
            res[2].append(xy_to_coord(y, x)) # white
            res[3].append(xy_to_coord(7 - x, 7 - y)) # 180
            res[4].append(xy_to_coord(7 - x, y)) # v mirror
            res[5].append(xy_to_coord(x, 7 - y)) # h mirror
            res[6].append(xy_to_coord(7 - y, x)) # v black
            res[7].append(xy_to_coord(y, 7 - x)) # v white
        elif N_SYMMETRY == 4:
            res[0].append(xy_to_coord(x, y))
            res[1].append(xy_to_coord(7 - y, x)) # 90 deg
            res[2].append(xy_to_coord(7 - x, 7 - y)) # 180 deg
            res[3].append(xy_to_coord(y, 7 - x)) # 270 deg

for arr in res:
    if len(arr) == len(s):
        l = 0
        for elem in arr:
            if elem[-2:] != 'NO':
                l += 1
        arr_s = '{' + str(l) + ', {' + ', '.join(arr) + '}},'
        print(arr_s)