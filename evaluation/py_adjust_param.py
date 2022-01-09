import subprocess
import sys

if len(sys.argv) != 3:
    print('arg err')
    exit()
phase = str(sys.argv[1])
player = str(sys.argv[2])
'''
with open(phase + '_' + player + '.txt', 'r') as f:
    data = f.read().replace('\n\n', '\n')
with open(phase + '_' + player + '.txt', 'w') as f:
    f.write(data)
exit()
'''
cmd = 'adjust_param.out ' + phase + ' ' + player + ' learned_data/' + phase + '_' + player + '.txt'
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '_' + player + '.txt', 'w') as f:
    f.write(param)
