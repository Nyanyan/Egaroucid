data = ''
for i in range(15):
    with open('learned_data/' + str(i) + '.txt', 'r') as f:
        data += f.read()
with open('param.txt', 'w') as f:
    f.write(data)