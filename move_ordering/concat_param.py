data_size = 349504
zeros = ''
for _ in range(data_size):
    zeros += '0\n'
data = ''

delta = 10

for i in range(0, 60, delta):
    try:
        with open('learned_data/' + str(i) + '_' + str(i + delta) + '.txt', 'r') as f:
            tmp = f.read()
            print(i, len(tmp.splitlines()))
            data += tmp
    except:
        print(i, 'add 0')
        data += zeros
with open('learned_data/param.txt', 'w') as f:
    f.write(data)
