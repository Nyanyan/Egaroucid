import subprocess
from othello_py import *
import sys

with open('problem/xot_small_shuffled.txt', 'r') as f:
    tactic = [elem for elem in f.read().splitlines()]
print(len(tactic), 'openings found')

#shuffle(tactic)

level = int(sys.argv[1])
n_games = int(sys.argv[2])

eval0 = sys.argv[3]
eval1 = sys.argv[4]

print('0 eval', eval0)
print('1 eval', eval1)

egaroucids = [
    subprocess.Popen(('Egaroucid_for_Console.exe -quiet -nobook -level ' + str(level) + ' -eval ' + eval0).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL),
    subprocess.Popen(('Egaroucid_for_Console.exe -quiet -nobook -level ' + str(level) + ' -eval ' + eval1).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
]

results = [[0, 0], [0, 0], [0, 0]] # 0 win (0 plays black / white), 0 lose (0 plays black / white), draw (0 plays black / white)

print('level', level)

max_num = min(len(tactic), n_games)
smpl = range(len(tactic))
print('play', max_num, 'games')

def create_res_str(arr):
    res = 'level: ' + str(level) + ' '
    res += '(0W-D-0L) 0=black: '
    res += str(arr[0][0]) + '-' + str(arr[2][0]) + '-' + str(arr[1][0])
    res += ' ' + str(round((arr[0][0] + arr[2][0] * 0.5) / max(1, arr[0][0] + arr[1][0] + arr[2][0]) * 100, 2)) + '% '
    res += ' 0=white: '
    res += str(arr[0][1]) + '-' + str(arr[2][1]) + '-' + str(arr[1][1])
    res += ' ' + str(round((arr[0][1] + arr[2][1] * 0.5) / max(1, arr[0][1] + arr[1][1] + arr[2][1]) * 100, 2)) + '% '
    res += ' all ' + str(round((arr[0][0] + arr[0][1]) / max(1, arr[0][0] + arr[0][1] + arr[1][0] + arr[1][1]) * 100, 2)) + '%'
    return res

for num in range(max_num):
    tactic_idx = smpl[num % len(tactic)]
    for player in range(2): # which plays black
        record = ''
        boards = []
        o = othello()
        for i in range(0, len(tactic[tactic_idx]), 2):
            if not o.check_legal():
                o.player = 1 - o.player
                o.check_legal()
            x = ord(tactic[tactic_idx][i].lower()) - ord('a')
            y = int(tactic[tactic_idx][i + 1]) - 1
            record += tactic[tactic_idx][i] + tactic[tactic_idx][i + 1]
            o.move(y, x)
        while True:
            if not o.check_legal():
                o.player = 1 - o.player
                if not o.check_legal():
                    break
            grid_str = 'setboard '
            for yy in range(hw):
                for xx in range(hw):
                    if o.grid[yy][xx] == black:
                        grid_str += 'b'
                    elif o.grid[yy][xx] == white:
                        grid_str += 'w'
                    else:
                        grid_str += '.'
            if o.player == black:
                grid_str += ' b\n'
            else:
                grid_str += ' w\n'
            player_idx = o.player ^ player
            egaroucids[player_idx].stdin.write(grid_str.encode('utf-8'))
            egaroucids[player_idx].stdin.flush()
            egaroucids[player_idx].stdin.write('go\n'.encode('utf-8'))
            egaroucids[player_idx].stdin.flush()
            line = ''
            while line == '':
                line = egaroucids[player_idx].stdout.readline().decode().replace('\r', '').replace('\n', '')
            coord = line
            try:
                y = int(coord[1]) - 1
                x = ord(coord[0]) - ord('a')
            except:
                print('error')
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                for i in range(2):
                    egaroucids[i].stdin.write('quit\n'.encode('utf-8'))
                    egaroucids[i].stdin.flush()
                exit()
            record += chr(ord('a') + x) + str(y + 1)
            if not o.move(y, x):
                o.print_info()
                print(grid_str[:-1])
                print(o.player, player)
                print(coord)
                print(y, x)
        wdl_idx = 0 if o.n_stones[player] > o.n_stones[1 - player] else 1 if o.n_stones[player] < o.n_stones[1 - player] else 2
        results[wdl_idx][player] += 1
        print('\r', num, max_num, ' ', create_res_str(results), end='          ')
        for i in range(2):
            egaroucids[i].stdin.write('clearcache\n'.encode('utf-8'))
            egaroucids[i].stdin.flush()

for i in range(2):
    egaroucids[i].stdin.write('quit\n'.encode('utf-8'))
    egaroucids[i].stdin.flush()

print('')

print(create_res_str(results))
