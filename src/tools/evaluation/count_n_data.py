with open('trained/opt_log.txt', 'r') as f:
    s = f.read()

DATA_IDX = 6
VAL_DATA_IDX = 8

s = s.splitlines()

n_data = 0
n_val_data = 0
n_count_phases = 0

for line in s:
    try:
        n_data += int(line.split()[DATA_IDX])
        n_val_data += int(line.split()[VAL_DATA_IDX])
        n_count_phases += 1
    except:
        continue

result = str(n_data) + ' data used for training, ' + str(n_val_data) + ' data used for validation, sum ' + str(n_data + n_val_data) +  ' for ' + str(n_count_phases) + ' phases'
print(result)

if input('write to info.txt?: ') == 'y':
    with open('trained/info.txt', 'a') as f:
        f.write(result + '\n')
    print('written')
