import glob
import sys

try:
    new_year = sys.argv[1]
except:
    print('please input new year')
    exit()

files = glob.glob('./**/*.cpp', recursive=True)
files.extend(glob.glob('./**/*.hpp', recursive=True))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if '@date' in line:
            lines[i] = lines[i][:lines[i].find('-') + 1] + new_year
    new_code = '\n'.join(lines)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(new_code)
