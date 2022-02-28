import subprocess
import sys

phase = str(sys.argv[1])
player = str(sys.argv[2])
if len(sys.argv) > 3:
    hour = str(sys.argv[3])
    minute = str(sys.argv[4])
    second = str(sys.argv[5])
    beta = str(sys.argv[6])
else:
    hour = '0'
    minute = '10'
    second = '0'
    beta = '0.002'

cmd = 'adjust_param.out ' + phase + ' ' + player + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' learned_data/' + phase + '_' + player + '.txt'
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '_' + player + '.txt', 'w') as f:
    f.write(param)
