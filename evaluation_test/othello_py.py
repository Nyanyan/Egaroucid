# 定数
hw = 8
hw2 = 64
dy = [0, 1, 0, -1, 1, 1, -1, -1]
dx = [1, 0, -1, 0, 1, -1, 1, -1]
black = 0
white = 1
legal = 2
vacant = 3

# 座標(y, x)が盤面内にあるかを見る関数
def inside(y, x):
    return 0 <= y < hw and 0 <= x < hw

# オセロのクラス
class othello:
    
    # 初期化
    def __init__(self):
        
        # 盤面の状態 0: 黒 1: 白 2: 合法手 3: 非合法手
        self.grid = [[vacant for _ in range(hw)] for _ in range(hw)]
        self.grid[3][3] = white
        self.grid[3][4] = black
        self.grid[4][3] = black
        self.grid[4][4] = white
        
        # プレーヤー情報 0: 黒 1: 白 -1: 終局
        self.player = black
        
        # 石数 n_stones[0]: 黒 n_stones[1]: 白
        self.n_stones = [2, 2]
    
    
    # 合法手生成 合法手が1つ以上あればTrueを、なければFalseを返す
    def check_legal(self):
        
        # 盤面の合法手表示をなくす
        for ny in range(hw):
            for nx in range(hw):
                if self.grid[ny][nx] == legal:
                    self.grid[ny][nx] = vacant
        
        # 返す値
        have_legal = False
        
        # 各マスについて合法かどうかチェック
        for y in range(hw):
            for x in range(hw):
                # すでに石が置いてあれば必ず非合法
                if self.grid[y][x] != vacant:
                    continue
                
                # 8方向それぞれ合法か見ていく
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
                
                # 合法だったらgridの値を更新
                if legal_flag:
                    self.grid[y][x] = legal
                    have_legal = True
        
        # 合法手が1つ以上あるかを返す
        return have_legal
    
    
    # 着手 着手成功ならTrueが、失敗したらFalseが返る
    def move(self, y, x):
        # 置けるかの判定
        if not inside(y, x):
            print('盤面外です')
            return False
        if self.grid[y][x] != legal:
            print('非合法手です')
            return False

        # ひっくり返した枚数(着手したぶんはカウントしない)
        n_flipped = 0
        # 8方向それぞれ合法か見ていき、合法ならひっくり返す
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
        
        # 着手部分の更新
        self.grid[y][x] = self.player
        
        # 石数の更新
        self.n_stones[self.player] += n_flipped + 1
        self.n_stones[1 - self.player] -= n_flipped
        
        # 手番の更新
        self.player = 1 - self.player
        
        # ひっくり返したのでTrueを返す
        return True
    
    # 標準入力からの入力で着手を行う
    def move_stdin(self):
        coord = input(('黒' if self.player == black else '白') + ' 着手: ')
        try:
            y = int(coord[1]) - 1
            x = ord(coord[0]) - ord('A')
            if not inside(y, x):
                x = ord(coord[0]) - ord('a')
                if not inside(y, x):
                    print('座標を A1 や c5 のように入力してください')
                    self.move_stdin()
                    return
            if not self.move(y, x):
                self.move_stdin()
        except:
            print('座標を A1 や c5 のように入力してください')
            self.move_stdin()
    
    
    # 盤面などの情報を表示
    def print_info(self):
        
        #盤面表示 X: 黒 O: 白 *: 合法手 .: 非合法手
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
        
        # 石数表示
        print('黒 X ', self.n_stones[0], '-', self.n_stones[1], ' O 白')