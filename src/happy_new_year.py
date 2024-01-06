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
        if '#define EGAROUCID_DATE ' in line:
            spl = line.split()
            n_line = ''
            for j in range(len(spl)):
                if '-' in spl[j] :
                    sspl = spl[j].split('-')
                    n_line += sspl[0] + '-' + new_year
                else:
                    n_line += spl[j]
            line = n_line
        if '@date' in line and '-' in line:
            lines[i] = lines[i][:lines[i].find('-') + 1] + new_year
    new_code = '\n'.join(lines)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(new_code)
