from random import shuffle

with open('problem/openingssmall.txt', 'r') as f:
    data = list(f.read().splitlines())

shuffle(data)

with open('problem/xot_small_shuffled.txt', 'w') as f:
    for elem in data:
        f.write(elem + '\n')