with open('trained/opt_log.txt', 'r') as f:
    s = f.read()

DATA_IDX = 6

s = s.splitlines()

n_data = 0
n_count_phases = 0

for line in s:
    try:
        n = int(line.split()[DATA_IDX])
        n_data += n
        n_count_phases += 1
    except:
        continue

result = str(n_data) + ' data used for optimization of ' + str(n_count_phases) + ' phases'
print(result)

if input('write to info.txt?: ') == 'y':
    with open('trained/info.txt', 'a') as f:
        f.write(result + '\n')
    print('written')
