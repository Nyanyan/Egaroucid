data_size = 457203 #804572 #694743
zeros = ''
for _ in range(data_size):
    zeros += '0\n'
data = ''
for i in range(30):
    try:
        #with open('learned_data/' + str(i) + '.txt', 'r') as f:
        with open('learned_data/' + str(i) + '.txt', 'r') as f:
            tmp = f.read()
            print(i, len(tmp.splitlines()))
            data += tmp
    except:
        print(i, 'add 0')
        data += zeros
with open('learned_data/param.txt', 'w') as f:
    f.write(data)