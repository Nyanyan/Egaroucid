import subprocess
import sys

phase = str(sys.argv[1])
if len(sys.argv) > 3:
    hour = str(sys.argv[2])
    minute = str(sys.argv[3])
    second = str(sys.argv[4])
    beta = str(sys.argv[5])
else:
    if int(phase) >= 10:
        hour = '0'
        minute = '20'
        second = '0'
        beta = '0.0075'
    else:
        hour = '0'
        minute = '5'
        second = '0'
        beta = '0.0035'
if int(phase) >= 10:
    additional_params = ' data2_04.dat data2_06.dat data2_07.dat data2_08.dat data2_09.dat data2_10.dat data2_11.dat'
else:
    additional_params = ' data2_01.dat data2_02.dat data2_03.dat'
#additional_params = ' big_data2_new_3.dat big_data2_new_14.dat'
#additional_params = ' big_data.dat'

cmd = 'adjust_param_new2.out ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' learned_data/' + phase + '.txt' + additional_params
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '.txt', 'w') as f:
    f.write(param)
