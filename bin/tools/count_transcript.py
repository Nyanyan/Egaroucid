import glob
from tqdm import trange

dr = './../../train_data/transcript/records34/*.txt'
print(dr)
files = glob.glob(dr)

n_games = 0
for file in files:
    with open(file, 'r') as f:
        n_games += len(f.read().splitlines())
print(n_games, 'games')