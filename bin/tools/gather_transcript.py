import glob
import shutil

path_to_transcript = './../transcript/'

joined_path = './../transcript/gathered/'

dirs = [str(elem) for elem in range(10, 20)]

files = []
for dr in dirs:
    path = path_to_transcript + dr + '/*.txt'
    files.extend(glob.glob(path))
print(files)

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

n_games = 0
for idx, file in enumerate(files):
    with open(file, 'r') as f:
        n_games_file = len(f.read().splitlines())
    to = joined_path + fill0(idx, 7) + '.txt'
    print(file, to, n_games_file)
    shutil.copy(file, to)
    n_games += n_games_file
print(n_games)