import glob

str_replaces = [
    ['MPC_78_LEVEL', 'MPC_70_LEVEL'],
    ['MPC_81_LEVEL', 'MPC_80_LEVEL'],
    ['MPC_95_LEVEL', 'MPC_93_LEVEL'],
]

files = glob.glob('./**/*.cpp', recursive=True)
files.extend(glob.glob('./**/*.hpp', recursive=True))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    for str_from, str_to in str_replaces:
        data = data.replace(str_from, str_to)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data)
