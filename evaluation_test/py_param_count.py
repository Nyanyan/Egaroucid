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
    minute = '20'
    second = '0'
    beta = '0.005'
if int(phase) >= 10:
    additional_params = ' big_data_new_4.dat big_data_new_6.dat big_data_new_7.dat big_data_new_8.dat big_data_new_9.dat big_data_new_10.dat'
else:
    additional_params = ' big_data_new_1.dat big_data_new_2.dat big_data_new_3.dat'
#additional_params = ' big_data_new_3.dat big_data_new_14.dat'
#additional_params = ' big_data.dat'

cmd = 'param_count.out ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' data/' + phase + '.txt' + additional_params
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open('data/' + phase + '_count.txt', 'w') as f:
    f.write(param)
