from tqdm import trange, tqdm

with open('third_party/mybook.txt', 'r') as f:
    data = f.read().splitlines()

with open('learned_data/book.txt', 'w') as f:
    for datum in tqdm(data):
        board, player, score = datum.split()
        score = int(score) + 64
        score = max(0, min(128, score))
        board = board.replace('.', '2')
        f.write(board + player + chr(ord('!') + score // 16) + chr(ord('!') + score % 16) + '\n')