import pyperclip

while True:
    raw_in = input().split()
    player_got = False
    for elem in raw_in:
        elem = elem.replace(',', '')
        try:
            elem_int = int(elem)
            if not player_got:
                player = elem_int
                player_got = True
            else:
                opponent = elem_int
        except:
            pass
    #player = int(input())
    #opponent = int(input())
    black_or_white = int(input()) # 0: X for black 1: X for white

    def fill_digit(s, n):
        while len(s) < n:
            s = '0' + s
        return s

    player_bin = bin(player)[2:]
    opponent_bin = bin(opponent)[2:]

    player_bin = fill_digit(player_bin, 64)
    opponent_bin = fill_digit(opponent_bin, 64)

    print(player_bin)
    print(opponent_bin)

    res = ''
    for i in range(64):
        if player_bin[i] == '1':
            res += 'X'
        elif opponent_bin[i] == '1':
            res += 'O'
        else:
            res += '-'
    if black_or_white == 0:
        res += ' X'
    else: # X is white
        res = res.replace('X', 'T')
        res = res.replace('O', 'X')
        res = res.replace('T', 'O')
        res += ' O'

    print(res)

    pyperclip.copy(res)