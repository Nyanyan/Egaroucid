def print_log(*args, **kwargs):
    print(*args, **kwargs)

# 定数
HW = 8
HW2 = 64
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]
BLACK = 0
WHITE = 1
EMPTY = 2

# s = "..OOOOOO..XXXXOOXXXXOOXO..XOOOXO..XXOXXO..XXOOXO..XOXXOO..XXXXXO"
# mapping = {'.': EMPTY, 'X': BLACK, 'O': WHITE}
# board_arr = [[mapping[c] for c in s[i*8:(i+1)*8]] for i in range(8)]
# print_log(board_arr)
# player = BLACK
# exit(0)

# 座標(y, x)が盤面内にあるかを見る関数
def inside(y, x):
    return 0 <= y < HW and 0 <= x < HW

# オセロのクラス
class Othello:
    
    # 初期化
    def __init__(self, board_arr=None, player=None):
        if board_arr != None and player != None:
            
            # 盤面の状態 0: 黒 1: 白 2: 空
            self.grid = board_arr

            # プレーヤー情報 0: 黒 1: 白 -1: 終局
            self.player = player
            
            # 石数 n_discs[0]: 黒 n_discs[1]: 白
            self.n_discs = [0, 0]
            for y in range(HW):
                for x in range(HW):
                    if self.grid[y][x] == BLACK:
                        self.n_discs[0] += 1
                    elif self.grid[y][x] == WHITE:
                        self.n_discs[1] += 1
        else:
            # 盤面の状態 0: 黒 1: 白 2: 空
            self.grid = [[EMPTY for _ in range(HW)] for _ in range(HW)]
            self.grid[3][3] = WHITE
            self.grid[3][4] = BLACK
            self.grid[4][3] = BLACK
            self.grid[4][4] = WHITE
            
            # プレーヤー情報 0: 黒 1: 白 -1: 終局
            self.player = BLACK
            
            # 石数 n_discs[0]: 黒 n_discs[1]: 白
            self.n_discs = [2, 2]
    
    def is_legal(self, y, x):
        # すでに石が置いてあれば必ず非合法
        if self.grid[y][x] != EMPTY:
            return False
        
        # 8方向それぞれ合法か見ていく
        legal_flag = False
        for dr in range(8):
            dr_legal_flag1 = False
            dr_legal_flag2 = False
            ny = y
            nx = x
            for _ in range(HW - 1):
                ny += dy[dr]
                nx += dx[dr]
                if not inside(ny, nx):
                    dr_legal_flag1 = False
                    break
                elif self.grid[ny][nx] == EMPTY:
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
        return legal_flag
    
    def is_end(self):
        if self.has_legal():
            return False
        res = True
        # 手番の更新
        self.player = 1 - self.player
        if self.has_legal():
            res = False
        # 手番の更新
        self.player = 1 - self.player
        return res

    # 合法手生成 合法手の位置をTrueにした配列を返す
    def get_legal(self):
        # 合法手をTrueにする配列
        legal_moves = [[False for _ in range(HW)] for _ in range(HW)]

        # 各マスについて合法かどうかチェック
        for y in range(HW):
            for x in range(HW):
                if self.is_legal(y, x):
                    legal_moves[y][x] = True
        
        return legal_moves
    
    def get_legal_moves(self):
        legal_moves = self.get_legal()
        moves = []
        for y in range(HW):
            for x in range(HW):
                if legal_moves[y][x]:
                    moves.append((y, x))
        return moves

    def has_legal(self):
        legal_moves = self.get_legal()
        for y in range(HW):
            for x in range(HW):
                if legal_moves[y][x]:
                    return True
        return False

    # 返る石を返す    
    def get_flipped(self, y, x):
        # 置けるかの判定
        if not inside(y, x):
            print_log('盤面外です')
            return None
        legal_moves = self.get_legal()
        if not legal_moves[y][x]:
            print_log('非合法手です')
            return None

        flipped = [[False for _ in range(HW)] for _ in range(HW)]
        # 8方向それぞれ合法か見ていき、合法ならひっくり返す
        for dr in range(8):
            dr_legal_flag = False
            dr_n_flipped = 0
            ny = y
            nx = x
            for d in range(HW - 1):
                ny += dy[dr]
                nx += dx[dr]
                if not inside(ny, nx):
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] == EMPTY:
                    dr_legal_flag = False
                    break
                elif self.grid[ny][nx] != self.player:
                    dr_legal_flag = True
                elif self.grid[ny][nx] == self.player:
                    dr_n_flipped = d
                    break
            if dr_legal_flag:
                for d in range(dr_n_flipped):
                    ny = y + dy[dr] * (d + 1)
                    nx = x + dx[dr] * (d + 1)
                    flipped[ny][nx] = True

        return flipped
    
    # 着手 着手成功ならTrueが、失敗したらFalseが返る
    def move(self, put_y, put_x):
        flipped = self.get_flipped(put_y, put_x)
        if flipped is None:
            return False
        
        # 着手
        n_flipped = 0
        for y in range(HW):
            for x in range(HW):
                if flipped[y][x]:
                    self.grid[y][x] = self.player
                    n_flipped += 1
        
        # 着手部分の更新
        self.grid[put_y][put_x] = self.player
        
        # 石数の更新
        self.n_discs[self.player] += n_flipped + 1
        self.n_discs[1 - self.player] -= n_flipped
        
        # 手番の更新
        self.player = 1 - self.player

        return True
    
    def move_pass(self):
        # 手番の更新
        self.player = 1 - self.player

    # 標準入力からの入力で着手を行う
    def move_stdin(self):
        coord = input(('黒' if self.player == BLACK else '白') + ' 着手: ')
        try:
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('A')
            if not inside(y, x):
                x = ord(coord[0]) - ord('a')
                if not inside(y, x):
                    print_log('座標を A1 や c5 のように入力してください')
                    self.move_stdin()
                    return
            if not self.move(y, x):
                self.move_stdin()
        except:
            print_log('座標を A1 や c5 のように入力してください')
            self.move_stdin()
    
    
    # 盤面などの情報を表示
    def print_log(self):
        #盤面表示 X: 黒 O: 白 *: 合法手 .: 非合法手
        print_log('  A B C D E F G H')
        for y in range(HW):
            print_log(y + 1, end=' ')
            for x in range(HW):
                if self.grid[y][x] == BLACK:
                    print_log('X', end=' ')
                elif self.grid[y][x] == WHITE:
                    print_log('O', end=' ')
                else:
                    print_log('.', end=' ')
            print_log('')
        
        # 石数表示
        print_log('黒 X ', self.n_discs[0], '-', self.n_discs[1], ' O 白')
    
    def get_board_str(self):
        board_str = ''
        for y in range(HW):
            for x in range(HW):
                if self.grid[y][x] == BLACK:
                    board_str += 'X'
                elif self.grid[y][x] == WHITE:
                    board_str += 'O'
                else:
                    board_str += '.'
        board_str += ' '
        board_str += 'X' if self.player == BLACK else 'O'
        return board_str

    def count(self, color):
        res = 0
        for y in range(HW):
            for x in range(HW):
                if self.grid[y][x] == color:
                    res += 1
        return res