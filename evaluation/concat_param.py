data = ''
for i in range(10):
    for j in range(2):
        with open('learned_data/' + str(i) + '_' + str(j) + '.txt', 'r') as f:
            tmp = f.read()
            print(len(tmp.splitlines()))
            data += tmp
with open('learned_data/param.txt', 'w') as f:
    f.write(data)