n_flip_pre_calc = [
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 3, 4, 5], [1, 0, 0, 0, 1, 2, 3, 4], [1, 0, 1, 0, 1, 2, 3, 4], [0, 0, 0, 0, 1, 2, 3, 4], [0, 0, 0, 0, 1, 2, 3, 4],
    [2, 1, 0, 0, 0, 1, 2, 3], [2, 1, 1, 2, 0, 1, 2, 3], [0, 1, 0, 1, 0, 1, 2, 3], [0, 1, 0, 1, 0, 1, 2, 3], [1, 0, 0, 0, 0, 1, 2, 3], [1, 0, 1, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 1, 2, 3],
    [3, 2, 1, 0, 0, 0, 1, 2], [3, 2, 2, 2, 3, 0, 1, 2], [0, 2, 1, 1, 2, 0, 1, 2], [0, 2, 1, 1, 2, 0, 1, 2], [1, 0, 1, 0, 1, 0, 1, 2], [1, 0, 2, 0, 1, 0, 1, 2], [0, 0, 1, 0, 1, 0, 1, 2], [0, 0, 1, 0, 1, 0, 1, 2],
    [2, 1, 0, 0, 0, 0, 1, 2], [2, 1, 1, 2, 0, 0, 1, 2], [0, 1, 0, 1, 0, 0, 1, 2], [0, 1, 0, 1, 0, 0, 1, 2], [1, 0, 0, 0, 0, 0, 1, 2], [1, 0, 1, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 2],
    [4, 3, 2, 1, 0, 0, 0, 1], [4, 3, 3, 3, 3, 4, 0, 1], [0, 3, 2, 2, 2, 3, 0, 1], [0, 3, 2, 2, 2, 3, 0, 1], [1, 0, 2, 1, 1, 2, 0, 1], [1, 0, 3, 1, 1, 2, 0, 1], [0, 0, 2, 1, 1, 2, 0, 1], [0, 0, 2, 1, 1, 2, 0, 1],
    [2, 1, 0, 1, 0, 1, 0, 1], [2, 1, 1, 3, 0, 1, 0, 1], [0, 1, 0, 2, 0, 1, 0, 1], [0, 1, 0, 2, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0, 1],
    [3, 2, 1, 0, 0, 0, 0, 1], [3, 2, 2, 2, 3, 0, 0, 1], [0, 2, 1, 1, 2, 0, 0, 1], [0, 2, 1, 1, 2, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 2, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1],
    [2, 1, 0, 0, 0, 0, 0, 1], [2, 1, 1, 2, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1],
    [5, 4, 3, 2, 1, 0, 0, 0], [5, 4, 4, 4, 4, 4, 5, 0], [0, 4, 3, 3, 3, 3, 4, 0], [0, 4, 3, 3, 3, 3, 4, 0], [1, 0, 3, 2, 2, 2, 3, 0], [1, 0, 4, 2, 2, 2, 3, 0], [0, 0, 3, 2, 2, 2, 3, 0], [0, 0, 3, 2, 2, 2, 3, 0],
    [2, 1, 0, 2, 1, 1, 2, 0], [2, 1, 1, 4, 1, 1, 2, 0], [0, 1, 0, 3, 1, 1, 2, 0], [0, 1, 0, 3, 1, 1, 2, 0], [1, 0, 0, 2, 1, 1, 2, 0], [1, 0, 1, 2, 1, 1, 2, 0], [0, 0, 0, 2, 1, 1, 2, 0], [0, 0, 0, 2, 1, 1, 2, 0],
    [3, 2, 1, 0, 1, 0, 1, 0], [3, 2, 2, 2, 4, 0, 1, 0], [0, 2, 1, 1, 3, 0, 1, 0], [0, 2, 1, 1, 3, 0, 1, 0], [1, 0, 1, 0, 2, 0, 1, 0], [1, 0, 2, 0, 2, 0, 1, 0], [0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 0, 2, 0, 1, 0],
    [2, 1, 0, 0, 1, 0, 1, 0], [2, 1, 1, 2, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0],
    [4, 3, 2, 1, 0, 0, 0, 0], [4, 3, 3, 3, 3, 4, 0, 0], [0, 3, 2, 2, 2, 3, 0, 0], [0, 3, 2, 2, 2, 3, 0, 0], [1, 0, 2, 1, 1, 2, 0, 0], [1, 0, 3, 1, 1, 2, 0, 0], [0, 0, 2, 1, 1, 2, 0, 0], [0, 0, 2, 1, 1, 2, 0, 0],
    [2, 1, 0, 1, 0, 1, 0, 0], [2, 1, 1, 3, 0, 1, 0, 0], [0, 1, 0, 2, 0, 1, 0, 0], [0, 1, 0, 2, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0], [3, 2, 2, 2, 3, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 2, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0], [2, 1, 1, 2, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    [6, 5, 4, 3, 2, 1, 0, 0], [6, 5, 5, 5, 5, 5, 5, 6], [0, 5, 4, 4, 4, 4, 4, 5], [0, 5, 4, 4, 4, 4, 4, 5], [1, 0, 4, 3, 3, 3, 3, 4], [1, 0, 5, 3, 3, 3, 3, 4], [0, 0, 4, 3, 3, 3, 3, 4], [0, 0, 4, 3, 3, 3, 3, 4],
    [2, 1, 0, 3, 2, 2, 2, 3], [2, 1, 1, 5, 2, 2, 2, 3], [0, 1, 0, 4, 2, 2, 2, 3], [0, 1, 0, 4, 2, 2, 2, 3], [1, 0, 0, 3, 2, 2, 2, 3], [1, 0, 1, 3, 2, 2, 2, 3], [0, 0, 0, 3, 2, 2, 2, 3], [0, 0, 0, 3, 2, 2, 2, 3],
    [3, 2, 1, 0, 2, 1, 1, 2], [3, 2, 2, 2, 5, 1, 1, 2], [0, 2, 1, 1, 4, 1, 1, 2], [0, 2, 1, 1, 4, 1, 1, 2], [1, 0, 1, 0, 3, 1, 1, 2], [1, 0, 2, 0, 3, 1, 1, 2], [0, 0, 1, 0, 3, 1, 1, 2], [0, 0, 1, 0, 3, 1, 1, 2],
    [2, 1, 0, 0, 2, 1, 1, 2], [2, 1, 1, 2, 2, 1, 1, 2], [0, 1, 0, 1, 2, 1, 1, 2], [0, 1, 0, 1, 2, 1, 1, 2], [1, 0, 0, 0, 2, 1, 1, 2], [1, 0, 1, 0, 2, 1, 1, 2], [0, 0, 0, 0, 2, 1, 1, 2], [0, 0, 0, 0, 2, 1, 1, 2],
    [4, 3, 2, 1, 0, 1, 0, 1], [4, 3, 3, 3, 3, 5, 0, 1], [0, 3, 2, 2, 2, 4, 0, 1], [0, 3, 2, 2, 2, 4, 0, 1], [1, 0, 2, 1, 1, 3, 0, 1], [1, 0, 3, 1, 1, 3, 0, 1], [0, 0, 2, 1, 1, 3, 0, 1], [0, 0, 2, 1, 1, 3, 0, 1],
    [2, 1, 0, 1, 0, 2, 0, 1], [2, 1, 1, 3, 0, 2, 0, 1], [0, 1, 0, 2, 0, 2, 0, 1], [0, 1, 0, 2, 0, 2, 0, 1], [1, 0, 0, 1, 0, 2, 0, 1], [1, 0, 1, 1, 0, 2, 0, 1], [0, 0, 0, 1, 0, 2, 0, 1], [0, 0, 0, 1, 0, 2, 0, 1],
    [3, 2, 1, 0, 0, 1, 0, 1], [3, 2, 2, 2, 3, 1, 0, 1], [0, 2, 1, 1, 2, 1, 0, 1], [0, 2, 1, 1, 2, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1], [1, 0, 2, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 0, 1],
    [2, 1, 0, 0, 0, 1, 0, 1], [2, 1, 1, 2, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1],
    [5, 4, 3, 2, 1, 0, 0, 0], [5, 4, 4, 4, 4, 4, 5, 0], [0, 4, 3, 3, 3, 3, 4, 0], [0, 4, 3, 3, 3, 3, 4, 0], [1, 0, 3, 2, 2, 2, 3, 0], [1, 0, 4, 2, 2, 2, 3, 0], [0, 0, 3, 2, 2, 2, 3, 0], [0, 0, 3, 2, 2, 2, 3, 0],
    [2, 1, 0, 2, 1, 1, 2, 0], [2, 1, 1, 4, 1, 1, 2, 0], [0, 1, 0, 3, 1, 1, 2, 0], [0, 1, 0, 3, 1, 1, 2, 0], [1, 0, 0, 2, 1, 1, 2, 0], [1, 0, 1, 2, 1, 1, 2, 0], [0, 0, 0, 2, 1, 1, 2, 0], [0, 0, 0, 2, 1, 1, 2, 0],
    [3, 2, 1, 0, 1, 0, 1, 0], [3, 2, 2, 2, 4, 0, 1, 0], [0, 2, 1, 1, 3, 0, 1, 0], [0, 2, 1, 1, 3, 0, 1, 0], [1, 0, 1, 0, 2, 0, 1, 0], [1, 0, 2, 0, 2, 0, 1, 0], [0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 0, 2, 0, 1, 0],
    [2, 1, 0, 0, 1, 0, 1, 0], [2, 1, 1, 2, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0],
    [4, 3, 2, 1, 0, 0, 0, 0], [4, 3, 3, 3, 3, 4, 0, 0], [0, 3, 2, 2, 2, 3, 0, 0], [0, 3, 2, 2, 2, 3, 0, 0], [1, 0, 2, 1, 1, 2, 0, 0], [1, 0, 3, 1, 1, 2, 0, 0], [0, 0, 2, 1, 1, 2, 0, 0], [0, 0, 2, 1, 1, 2, 0, 0],
    [2, 1, 0, 1, 0, 1, 0, 0], [2, 1, 1, 3, 0, 1, 0, 0], [0, 1, 0, 2, 0, 1, 0, 0], [0, 1, 0, 2, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0], [3, 2, 2, 2, 3, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0], [0, 2, 1, 1, 2, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [1, 0, 2, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0], [2, 1, 1, 2, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]
]

rev_n_flip_pre_calc = []

for i in range(len(n_flip_pre_calc)):
    rev_n_flip_pre_calc.append([])
    for j in range(len(n_flip_pre_calc[i])):
        rev_n_flip_pre_calc[i].append(-1)

for i in range(len(n_flip_pre_calc)):
    for j in range(len(n_flip_pre_calc[i])):
        p = i
        put_bit = 1 << j
        if p & put_bit: # invalid
            n_flip_pre_calc[i][j] = 0
            rev_n_flip_pre_calc[i][j] = 0
            continue
        o = 0xff ^ p ^ put_bit
        rev_n_flip_pre_calc[i][j] = n_flip_pre_calc[o][j]

print(sum([sum(elem) for elem in n_flip_pre_calc]))
print(sum([sum(elem) for elem in rev_n_flip_pre_calc]))

for i in range(len(n_flip_pre_calc)):
    print('{', end='')
    for j in range(len(n_flip_pre_calc[i])):
        print('0x0' + str(rev_n_flip_pre_calc[i][j]) + '0' + str(n_flip_pre_calc[i][j]), end='')
        #print(rev_n_flip_pre_calc[i][j] << 8 | n_flip_pre_calc[i][j], end='')
        if j < len(n_flip_pre_calc[i]) - 1:
            print(', ', end='')
    print('}, ', end='')
    if i % 2 == 1:
        print('')