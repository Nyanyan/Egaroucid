import sys
import glob

d = 'third_party/' + sys.argv[1] + '/*.txt'
print('dir', d)
files = glob.glob(d)
print(len(files), 'files found')

res = 0
for file in files:
    with open(file) as f:
        while f.readline():
            res += 1
print(sys.argv[1], res, 'games found')