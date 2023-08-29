with open('param.txt', 'r') as f:
    data = f.read().splitlines()

with open('param_cpp.cpp', 'w') as f:
    f.write('{')
    for elem in data[:-1]:
        f.write(elem + ',')
    f.write(data[-1])
    f.write('};\n')