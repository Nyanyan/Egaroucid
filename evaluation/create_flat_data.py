from tqdm import trange, tqdm

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

nums = [0 for _ in range(129)]

max_nums = 5000
for file_idx in tqdm(list(reversed(range(435)))):
    with open('data/records4/' + digit(file_idx, 7) + '.txt', 'r') as f:
        data = f.read().splitlines()
    for datum in data:
        board, player, value = datum.split()
        n_moves = -4
        for i in range(64):
            n_moves += board[i] != '.'
        #if n_moves >= 20:
        value = int(value)
        score_idx = value + 64
        if nums[score_idx] < max_nums:
            with open('data/records5/0000000.txt', 'a') as f:
                f.write(datum + '\n')
            nums[score_idx] += 1
print(nums)