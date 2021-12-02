import sys


data = ''
for elem in sys.argv[1:]:
    with open('learned_data/' + elem, 'r') as f:
        data += f.read()

with open('param/param.txt', 'w') as f:
    f.write(data)