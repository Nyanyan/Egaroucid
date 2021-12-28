data = ''
with open('big_data1.txt', 'r') as f:
    data += f.read()
with open('data.txt', 'r') as f:
    data += f.read()

with open('big_data.txt', 'w') as f:
    f.write(data)