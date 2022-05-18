import sys

dense_pattern = int(sys.argv[1])

data_size = 804572
zeros = ''
for _ in range(data_size):
    zeros += '0\n'
data = ''
for i in range(30):
    try:
        #with open('learned_data/' + str(i) + '.txt', 'r') as f:
        if i >= 28:
            with open('learned_data/' + str(i) + '_' + str(dense_pattern) + '_model.txt', 'r') as f:
                tmp = f.read()
                print(i, len(tmp.splitlines()))
                data += tmp
        else:
            with open('data/' + str(i) + '.txt', 'r') as f:
                tmp = f.read()
                print(i, len(tmp.splitlines()))
                data += tmp
    except:
        print(i, 'add 0')
        data += zeros
with open('learned_data/param.txt', 'w') as f:
    f.write(data)