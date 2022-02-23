from random import sample
from tqdm import tqdm

all_testcases = [[[] for _ in range(65)] for _ in range(15)]

with open('eval_test_data.txt', 'r') as f:
    data = f.read().splitlines()
for datum in tqdm(data):
    score = int(datum[64:])
    if score % 2 == 0:
        n_moves = 59 - datum[:64].count('.')
        all_testcases[n_moves // 4][score // 2].append(datum)

min_phase_score_data = 100000
for i in range(15):
    for j in range(65):
        min_phase_score_data = min(min_phase_score_data, len(all_testcases[i][j]))
        print(len(all_testcases[i][j]), end=' ')
    print('')
print(min_phase_score_data)

max_n_data = 50

for phase in range(15):
    testcases = []
    for score_proc in range(65):
        testcases.extend(sample(all_testcases[phase][score_proc], min(len(all_testcases[phase][score_proc]), max_n_data)))
    with open('eval_testcases/' + str(phase) + '.txt', 'w') as f:
        for datum in testcases:
            f.write(datum + '\n')