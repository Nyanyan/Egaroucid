hw = 8
hw2 = 64
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]
black = 0
white = 1
legal = 2
vacant = 3

def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

class othello:

    def __init__(self):
        self.grid = [[vacant for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = white
        self.grid[3][4] = black
        self.grid[4][3] = black
        self.grid[4][4] = white
        self.player = black
        self.n_stones = [2, 2]
    
    def __lt__(self, other):
        return sum(self.n_stones) < sum(other.n_stones)

    def check_legal(self):
        for ny in range(hw):
            for nx in range(hw):
                if self.grid[ny][nx] == legal:
                    self.grid[ny][nx] = vacant
        have_legal = False
        for y in range(hw):
            for x in range(hw):
                if self.grid[y][x] != vacant:
                    continue
                legal_flag = False
                for dr in range(8):
                    dr_legal_flag1 = False
                    dr_legal_flag2 = False
                    ny = y
                    nx = x
                    for _ in range(hw - 1):
                        ny += dy[dr]
                        nx += dx[dr]
                        if not inside(ny, nx):
                            dr_legal_flag1 = False
                            break
                        elif self.grid[ny][nx] == vacant or self.grid[ny][nx] == legal:
                            dr_legal_flag1 = False
                            break
                        elif self.grid[ny][nx] != self.player:
                            dr_legal_flag1 = True
                        elif self.grid[ny][nx] == self.player:
                            dr_legal_flag2 = True
                            break
                    if dr_legal_flag1 and dr_legal_flag2:
                        legal_flag = True
                        break
                if legal_flag:
                    self.grid[y][x] = legal
                    have_legal = True
        return have_legal

    def move(self, y, x):
        if not inside(y, x):
            print('out of range')
            return False
        if self.grid[y][x] != legal:
            print('illegal move')
            return False
        n_flipped = 0
        for dr in range(8):
            dr_legal_flag = False
            dr_n_flipped = 0
            ny = y
            nx = x
            for d in range(hw - 1):
                ny += dy[dr]
                nx += dx[dr]
                if not inside(ny, nx):
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] == vacant or self.grid[ny][nx] == legal:
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] != self.player:
                    dr_legal_flag = True
                elif self.grid[ny][nx] == self.player:
                    dr_n_flipped = d
                    break
            if dr_legal_flag:
                n_flipped += dr_n_flipped
                for d in range(dr_n_flipped):
                    ny = y + dy[dr] * (d + 1)
                    nx = x + dx[dr] * (d + 1)
                    self.grid[ny][nx] = self.player
        self.grid[y][x] = self.player
        self.n_stones[self.player] += n_flipped + 1
        self.n_stones[1 - self.player] -= n_flipped
        self.player = 1 - self.player
        return True
    
    def move_stdin(self):
        coord = input(('黒' if self.player == black else '白') + ' 着手: ')
        try:
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('A')
            if not inside(y, x):
                x = ord(coord[0]) - ord('a')
                if not inside(y, x):
                    print('please input like A1 or f5')
                    self.move_stdin()
                    return
            if not self.move(y, x):
                self.move_stdin()
        except:
            print('please input like A1 or f5')
            self.move_stdin()

    def print_info(self):
        print('  A B C D E F G H')
        for y in range(hw):
            print(y + 1, end=' ')
            for x in range(hw):
                if self.grid[y][x] == black:
                    print('X', end=' ')
                elif self.grid[y][x] == white:
                    print('O', end=' ')
                elif self.grid[y][x] == legal:
                    print('*', end=' ')
                else:
                    print('.', end=' ')
            print('')
        print('Black X ', self.n_stones[0], '-', self.n_stones[1], ' O White')