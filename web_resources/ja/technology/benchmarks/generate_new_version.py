import sys
import shutil
import glob

version_dot = sys.argv[1]
version_underbar = version_dot.replace('.', '_')

new_path = shutil.copytree('format', version_underbar)

files = glob.glob(version_underbar + '/*.md')
files.extend(glob.glob(version_underbar + '/*.txt'))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data.replace('!!VERSION!!', version_dot))

with open('tasks.txt', 'r') as f:
    data = f.read().splitlines()
if not version_underbar in data:
    with open('tasks.txt', 'a') as f:
        f.write(version_underbar + '\n')