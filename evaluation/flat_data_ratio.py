with open('diversity_cut.txt', 'r') as f:
    data = [[int(i) for i in j.split()] for j in f.read().splitlines()]

x = [elem[0] for elem in data]
y = [elem[1] for elem in data]
min_y = min(y)
ratio = [min_y / elem for elem in y]

for elem in ratio:
    print(elem, ', ')