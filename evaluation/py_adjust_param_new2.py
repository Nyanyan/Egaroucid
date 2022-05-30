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
        beta = '0.005'
    else:
        hour = '0'
        minute = '20'
        second = '0'
        beta = '0.005'
if int(phase) >= 10:
    additional_params = ' data2_0000004.dat data2_0000006.dat data2_0000007.dat data2_0000008.dat data2_0000009.dat data2_0000010.dat'
else:
    additional_params = ' data2_0000001.dat data2_0000002.dat data2_0000003.dat'
#additional_params = ' big_data2_new_3.dat big_data2_new_14.dat'
#additional_params = ' big_data.dat'

cmd = 'adjust_param_new2.out ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' learned_data/' + phase + '.txt' + additional_params
print(cmd)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '.txt', 'w') as f:
    f.write(param)
