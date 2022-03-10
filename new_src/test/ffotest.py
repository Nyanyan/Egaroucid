import subprocess

egaroucid = subprocess.Popen('a.exe'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res_str = ''
tim = 0
stim = 0
nodes = 0
for i in range(40, 50):
    print('#', i)
    with open('./../../benchmark/ffotest/' + str(i) + '.txt', 'r') as f:
        s = f.read()
    egaroucid.stdin.write(s.encode('utf-8'))
    egaroucid.stdin.flush()
    result = egaroucid.stdout.readline().decode()
    res_str += '#' + str(i) + ' ' + result
    tim += int(result.split()[10])
    stim += int(result.split()[13])
    nodes += int(result.split()[7])
egaroucid.kill()

print('done')
print(res_str, end='')
print(tim / 1000, 'sec')
print(stim / 1000, 'sec search')
print(nodes, 'nodes')
print(nodes / stim * 1000, 'nps')