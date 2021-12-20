from tqdm import trange

hw = 8
hw2 = 64
board_index_num = 38
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]

all_chars = [
    '!', '#', '$', '&', "'", '(', ')', '*', 
    '+', ',', '-', '.', '/', '0', '1', '2', 
    '3', '4', '5', '6', '7', '8', '9', ':', 
    ';', '<', '=', '>', '?', '@', 'A', 'B', 
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '[', ']', '^', '_', '`', 'a', 'b', 'c', 
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

char_translate = {}
for i in range(64):
    char_translate[all_chars[i]] = i

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def empty(grid, y, x):
    return grid[y][x] == -1 or grid[y][x] == 2

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

def check(grid, player, y, x):
    res_grid = [[False for _ in range(hw)] for _ in range(hw)]
    res = 0
    for dr in range(8):
        ny = y + dy[dr]
        nx = x + dx[dr]
        if not inside(ny, nx):
            continue
        if empty(grid, ny, nx):
            continue
        if grid[ny][nx] == player:
            continue
        #print(y, x, dr, ny, nx)
        plus = 0
        flag = False
        for d in range(hw):
            nny = ny + d * dy[dr]
            nnx = nx + d * dx[dr]
            if not inside(nny, nnx):
                break
            if empty(grid, nny, nnx):
                break
            if grid[nny][nnx] == player:
                flag = True
                break
            #print(y, x, dr, nny, nnx)
            plus += 1
        if flag:
            res += plus
            for d in range(plus):
                nny = ny + d * dy[dr]
                nnx = nx + d * dx[dr]
                res_grid[nny][nnx] = True
    return res, res_grid

def pot_canput_line(arr):
    res_p = 0
    res_o = 0
    for i in range(len(arr) - 1):
        if arr[i] == -1 or arr[i] == 2:
            if arr[i + 1] == 0:
                res_o += 1
            elif arr[i + 1] == 1:
                res_p += 1
    for i in range(1, len(arr)):
        if arr[i] == -1 or arr[i] == 2:
            if arr[i - 1] == 0:
                res_o += 1
            elif arr[i - 1] == 1:
                res_p += 1
    return res_p, res_o

class reversi:
    def __init__(self):
        self.grid = [[-1 for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = 1
        self.grid[3][4] = 0
        self.grid[4][3] = 0
        self.grid[4][4] = 1
        self.player = 0 # 0: 黒 1: 白
        self.nums = [2, 2]

    def move(self, y, x):
        plus, plus_grid = check(self.grid, self.player, y, x)
        if (not empty(self.grid, y, x)) or (not inside(y, x)) or not plus:
            print('Please input a correct move')
            return 1
        self.grid[y][x] = self.player
        for ny in range(hw):
            for nx in range(hw):
                if plus_grid[ny][nx]:
                    self.grid[ny][nx] = self.player
        self.nums[self.player] += 1 + plus
        self.nums[1 - self.player] -= plus
        self.player = 1 - self.player
        return 0
    
    def check_pass(self):
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == 2:
                    self.grid[y][x] = -1
        res = True
        for y in range(hw):
            for x in range(hw):
                if not empty(self.grid, y, x):
                    continue
                plus, _ = check(self.grid, self.player, y, x)
                if plus:
                    res = False
                    self.grid[y][x] = 2
        if res:
            #print('Pass!')
            self.player = 1 - self.player
        return res

    def output(self):
        print('  ', end='')
        for i in range(hw):
            print(chr(ord('a') + i), end=' ')
        print('')
        for y in range(hw):
            print(str(y + 1) + ' ', end='')
            for x in range(hw):
                print('○' if self.grid[y][x] == 0 else '●' if self.grid[y][x] == 1 else '* ' if self.grid[y][x] == 2 else '. ', end='')
            print('')
    
    def output_file(self):
        res = ''
        for y in range(hw):
            for x in range(hw):
                res += '*' if self.grid[y][x] == 0 else 'O' if self.grid[y][x] == 1 else '-'
        res += ' *'
        return res

    def end(self):
        if min(self.nums) == 0:
            return True
        res = True
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] == -1 or self.grid[y][x] == 2:
                    res = False
        return res
    
    def judge(self):
        if self.nums[0] > self.nums[1]:
            #print('Black won!', self.nums[0], '-', self.nums[1])
            return 0
        elif self.nums[1] > self.nums[0]:
            #print('White won!', self.nums[0], '-', self.nums[1])
            return 1
        else:
            #print('Draw!', self.nums[0], '-', self.nums[1])
            return -1

def get_diff(p, n):
    for i in range(64):
        if p[i] == '.' and n[i] != '.':
            return all_chars[i] #chr(i % hw + ord('a')) + str(i // hw + 1) #
    return -1

def collect_data(num, boards):
    records = []
    prev = ''
    for idx in trange(len(boards)):
        if boards[idx][0] == '...........................10......01...........................':
            if idx > 0:
                records[-1] += ' ' + str(boards[idx - 1][5])
            records.append('')
        else:
            policy = get_diff(prev, boards[idx][0])
            records[-1] += boards[idx - 1][1] + policy
        prev = boards[idx][0]
    records[-1] += ' ' + str(boards[len(boards) - 1][5])
    with open('data/' + digit(num, 7) + '.txt', 'a') as f:
        for record in records:
            raw_record, score = record.split()
            if raw_record[1] == all_chars[26]:
                rec_num = ''
                for i in range(1, len(raw_record), 2):
                    coord = char_translate[raw_record[i]]
                    x = coord % 8
                    y = coord // 8
                    yy = 7 - y
                    xx = 7 - x
                    rec_num += raw_record[i - 1] + all_chars[yy * 8 + xx]
                record = rec_num + ' ' + score
            elif raw_record[0] == all_chars[19]:
                rec_num = ''
                for i in range(1, len(raw_record), 2):
                    coord = char_translate[raw_record[i]]
                    x = coord % 8
                    y = coord // 8
                    yy = 7 - x
                    xx = 7 - y
                    rec_num += raw_record[i - 1] + all_chars[yy * 8 + xx]
                record = rec_num + ' ' + score
            elif raw_record[0] == all_chars[44]:
                rec_num = ''
                for i in range(1, len(raw_record), 2):
                    coord = char_translate[raw_record[i]]
                    x = coord % 8
                    y = coord // 8
                    yy = x
                    xx = y
                    rec_num += raw_record[i - 1] + all_chars[yy * 8 + xx]
                record = rec_num + ' ' + score
            f.write(record + '\n')


games = []

for idx in range(104): #range(1111111, 1111112):
    raw_data = ''
    with open('third_party/self_play/' + digit(idx, 7) + '.txt', 'r') as f:
        raw_data = f.read()
    collect_data(127 + idx, [list(i.split()) for i in raw_data.splitlines()])
