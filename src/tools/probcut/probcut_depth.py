for i in range(61):
    n = i // 4 * 2
    if i % 2:
        n += 1
    print(n, end=', ')
    if i % 10 == 9:
        print('')