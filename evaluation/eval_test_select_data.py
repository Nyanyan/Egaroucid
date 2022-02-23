from random import sample
from tqdm import tqdm


depth = 0
n_testcases = 500

all_testcases = [[] for _ in range(65)]

with open('eval_test_data.txt', 'r') as f:
    data = f.read().splitlines()
for datum in tqdm(data):
    score = int(datum[64:])
    if score % 2 == 0:
        all_testcases[score // 2].append(datum[:64])

print([len(elem) for elem in all_testcases])

for i in range(65):
    score = i * 2 - 64
    testcases = sample(all_testcases[i], min(len(all_testcases[i]), n_testcases))
    with open('eval_testcases/' + str(score + 64) + '.txt', 'w') as f:
        for datum in testcases:
            f.write(datum + '\n')