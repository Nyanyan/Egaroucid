import subprocess
import sys

if len(sys.argv) != 3:
    print('arg err')
    exit()
s_n_moves = str(sys.argv[1])
e_n_moves = str(sys.argv[2])
'''
with open(phase + '_' + player + '.txt', 'r') as f:
    data = f.read().replace('\n\n', '\n')
with open(phase + '_' + player + '.txt', 'w') as f:
    f.write(data)
exit()
'''
cmd = 'adjust_param.out ' + s_n_moves + ' ' + e_n_moves + ' learned_data/' + s_n_moves + ' ' + e_n_moves + '.txt'
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(s_n_moves + '_' + e_n_moves + '.txt', 'w') as f:
    f.write(param)
