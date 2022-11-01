
line_strt = 39
file = './../statistics/statistics/end.txt'
all_scores = []
with open(file, 'r') as f:
    for i in range(line_strt - 1):
        f.readline()
    for i in range(4):
        line = f.readline()
        scores = [float(elem) for elem in line.split()]
        for score in scores:
            all_scores.append(score)
print(all_scores)
print('')
cells_10_64 = [
    [0, 7, 56, 63],
    [1, 6, 8, 15, 48, 55, 57, 62],
    [2, 5, 23, 16, 40, 47, 58, 61],
    [3, 4, 24, 31, 32, 39, 59, 60],
    [9, 14, 49, 54],
    [10, 13, 17, 22, 41, 46, 50, 53],
    [11, 12, 25, 30, 33, 38, 51, 52],
    [18, 21, 42, 45],
    [19, 20, 26, 29, 34, 37, 43, 44],
    [27, 28, 35, 36]
]

res = [0 for _ in range(64)]
for i in range(10):
    for j in cells_10_64[i]:
        res[j] = round(all_scores[i] * 20)

for i in range(8):
    print(res[i * 8:(i + 1) * 8])
