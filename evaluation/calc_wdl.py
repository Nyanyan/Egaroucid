
black_win = 0
draw = 0
white_win = 0
with open('big_data.txt', 'r') as f:
    t = 0
    while True:
        try:
            t += 1
            score = int(f.readline().split()[-1])
            if score > 0:
                black_win += 1
            elif score < 0:
                white_win += 1
            else:
                draw += 1
            if t % 10000 == 0:
                print('\r', black_win, draw, white_win, end='')
        except:
            break

print('\r', black_win, draw, white_win)