for i in range(15):
    for j in range(2):
        with open('learned_data/' + str(i) + '_' + str(j) + '.txt', 'r') as f:
            data = f.read()
        with open('learned_data/' + str(i) + '_' + str(j) + '.txt', 'w') as f:
            f.write(data + '0\n')
    