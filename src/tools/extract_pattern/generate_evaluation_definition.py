s = '''230.794 COORD_C1, COORD_C2, COORD_F2, COORD_B4, COORD_A5, COORD_A6, COORD_A8, COORD_C8, COORD_E8, COORD_H8
275.783 COORD_C1, COORD_C3, COORD_D4, COORD_A5, COORD_C6, COORD_D6, COORD_C7, COORD_F7, COORD_H7, COORD_H8
284.238 COORD_C2, COORD_A4, COORD_H4, COORD_A6, COORD_A7, COORD_B8, COORD_C8, COORD_E8, COORD_G8, COORD_H8
295.82 COORD_D4, COORD_B5, COORD_A6, COORD_B6, COORD_D6, COORD_E6, COORD_F6, COORD_H6, COORD_G8, COORD_H8
301.621 COORD_F1, COORD_B2, COORD_C2, COORD_D2, COORD_G3, COORD_G5, COORD_H5, COORD_H7, COORD_E8, COORD_H8
312.024 COORD_D1, COORD_E2, COORD_E3, COORD_D4, COORD_H4, COORD_G5, COORD_D6, COORD_C7, COORD_G7, COORD_H8
312.671 COORD_F1, COORD_G1, COORD_B2, COORD_D2, COORD_G3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8
333.084 COORD_C2, COORD_B3, COORD_A4, COORD_B4, COORD_E4, COORD_D5, COORD_F5, COORD_E6, COORD_F6, COORD_B8
350.17 COORD_B1, COORD_C5, COORD_D6, COORD_E6, COORD_C7, COORD_E7, COORD_G7, COORD_H7, COORD_C8, COORD_G8
349.88 COORD_D1, COORD_C3, COORD_B4, COORD_A6, COORD_B6, COORD_A7, COORD_B7, COORD_F7, COORD_G7, COORD_H8
364.97 COORD_B2, COORD_F3, COORD_C4, COORD_D4, COORD_B5, COORD_D5, COORD_A6, COORD_D6, COORD_E6, COORD_B8
378.317 COORD_H2, COORD_C3, COORD_B5, COORD_F5, COORD_G5, COORD_G6, COORD_E7, COORD_G7, COORD_H7, COORD_C8'''

s = s.splitlines()

def xy_to_coord(x, y):
    return 'COORD_' + chr(ord('A') + x) + str(y + 1)

def rotate_feature(ss, n):

    res = [[] for _ in range(8)]

    for elem in ss:
        x = ord(elem[-2]) - ord('A')
        y = int(elem[-1]) - 1
        
        res[0].append(xy_to_coord(x, y))
        res[1].append(xy_to_coord(7 - y, 7 - x)) # black
        res[2].append(xy_to_coord(y, x)) # white
        res[3].append(xy_to_coord(7 - x, 7 - y)) # 180
        res[4].append(xy_to_coord(7 - x, y)) # v mirror
        res[5].append(xy_to_coord(x, 7 - y)) # h mirror
        res[6].append(xy_to_coord(7 - y, x)) # v black
        res[7].append(xy_to_coord(y, 7 - x)) # v white

    res_str = ''
    for arr in res:
        arr_s = '{' + str(len(arr)) + ', {' + ', '.join(arr) + '}}, // ' + str(n)
        res_str += arr_s + '\n'
        n += 1
    return res_str

n = 0
for idx, line in enumerate(s):
    line = line.replace(',', '').split()[1:]
    features = rotate_feature(line, n)
    n += 8
    print('// ' + str(idx))
    print(features)