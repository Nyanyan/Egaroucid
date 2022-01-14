data_size = 535444 #417346
zeros = ''
for _ in range(data_size):
    zeros += '0\n'
data = ''
for i in range(15):
    for j in range(2):
        try:
            #with open('learned_data/' + str(i) + '.txt', 'r') as f:
            with open('learned_data/' + str(i) + '_' + str(j) + '.txt', 'r') as f:
                tmp = f.read()
                print(i, j, len(tmp.splitlines()))
                data += tmp
        except:
            print(i, j, 'add 0')
            data += zeros
with open('learned_data/param.txt', 'w') as f:
    f.write(data)