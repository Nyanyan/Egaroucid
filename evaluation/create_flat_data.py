from tqdm import trange, tqdm

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

nums = [0 for _ in range(16)]

max_nums = 65000
for file_idx in tqdm(list(reversed(range(438)))):
    with open('data/records3/' + digit(file_idx, 7) + '.txt', 'r') as f:
        data = f.read().splitlines()
    for datum in data:
        board, player, value = datum.split()
        value = int(value)
        score_idx = min(15, (value + 64) // 8)
        if nums[score_idx] < max_nums:
            with open('data/records5/0000000.txt', 'a') as f:
                f.write(datum + '\n')
            nums[score_idx] += 1
print(nums)