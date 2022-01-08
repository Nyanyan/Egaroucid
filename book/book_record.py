from tqdm import tqdm
import glob

hw = 8
hw2 = 64

# 0 > 1 > .
def compare(a, b):
    for i in range(hw2):
        if a[i] != '.' and b[i] == '.':
            return False
        elif a[i] == '.' and b[i] != '.':
            return True
        elif a[i] == '0' and b[i] == '1':
            return False
        elif a[i] == '1' and b[i] == '0':
            return True
    return False

def rotate_board(board):
    res = board
    
    n_board = ''
    for y in range(hw):
        for x in range(hw):
            n_board += board[hw * x + y]
    if compare(res, n_board):
        res = n_board
    
    n_board = ''
    for y in range(hw):
        for x in range(hw):
            n_board += board[hw * (7 - y) + (7 - x)]
    if compare(res, n_board):
        res = n_board
    
    n_board = ''
    for y in range(hw):
        for x in range(hw):
            n_board += board[hw * (7 - x) + (7 - y)]
    if compare(res, n_board):
        res = n_board
    
    return res

data_dict = {}
files = glob.glob('third_party/records3/*')
for file in tqdm(files):
    with open(file, 'r') as f:
        data = f.read().splitlines()
    for datum in data:
        board, player, score = datum.split()
        player = int(player)
        score = int(score)
        board = rotate_board(board)
        if board in data_dict:
            data_dict[board][0] += 1
            data_dict[board][1] += score
        else:
            data_dict[board] = [1, score, player]

use_threshold = 250
n_boards = 0
with open('learned_data/book.txt', 'w') as f:
    for board in data_dict.keys():
        if data_dict[board][0] >= use_threshold:
            f.write(board + ' ' + str(data_dict[board][2]) + ' ' + str(round(data_dict[board][1] / data_dict[board][0])) + '\n')
            n_boards += 1
print(n_boards)
