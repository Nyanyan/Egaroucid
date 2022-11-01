import glob
import subprocess
from tqdm import tqdm

n_games = 0
for dir_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 9999999]:
    dr = 'third_party/records' + str(dir_num)
    files = glob.glob(dr + '/*.txt')
    n_games_bef = n_games
    for file in tqdm(files):
        file = file.replace('\\', '/')
        n_games += int(subprocess.check_output(['wc', '-l', file]).decode().split(' ')[0])
    print(dr, len(files), n_games - n_games_bef, n_games)
print('n_games', n_games)

'''
n_games = 0
for dir_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99]:
    dr = 'data/records' + str(dir_num)
    files = glob.glob(dr + '/*.txt')
    n_games_bef = n_games
    for file in tqdm(files):
        file = file.replace('\\', '/')
        n_games += int(subprocess.check_output(['wc', '-l', file]).decode().split(' ')[0])
    print(dr, len(files), n_games - n_games_bef, n_games)
print('n_boards', n_games)
'''


