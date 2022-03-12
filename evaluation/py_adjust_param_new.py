import subprocess
import sys

phase = str(sys.argv[1])
if len(sys.argv) > 3:
    hour = str(sys.argv[2])
    minute = str(sys.argv[3])
    second = str(sys.argv[4])
    beta = str(sys.argv[5])
else:
    hour = '0'
    minute = '1'
    second = '0'
    beta = '0.003'
#additional_params = ' big_data_new_4.dat big_data_new_6.dat big_data_new_7.dat big_data_new_8.dat big_data_new_9.dat big_data_new_10.dat'
additional_params = ' big_data_new_1.dat big_data_new_2.dat big_data_new_3.dat'
#additional_params = ' big_data.dat'

cmd = 'adjust_param_new.out ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' learned_data/' + phase + '.txt' + additional_params
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '.txt', 'w') as f:
    f.write(param)
