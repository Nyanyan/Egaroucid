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
    minute = '5'
    second = '0'
    beta = '0.0113'

if int(phase) < 20:
    train_data = [
        'data5_01.dat',
        'data5_02.dat',
        'data5_04.dat',
        'data5_06.dat',
        'data5_07.dat',
        'data5_08.dat',
        'data5_09.dat',
        'data5_10.dat',
        'data5_11.dat',
        'data5_15.dat',
        'data5_16.dat',
        'data5_99.dat'
    ]
else:
    train_data = [
        #'data5_01.dat',
        #'data5_02.dat',
        'data5_04.dat',
        'data5_06.dat',
        'data5_07.dat',
        'data5_08.dat',
        'data5_09.dat',
        'data5_10.dat',
        'data5_11.dat',
        'data5_15.dat',
        'data5_16.dat',
        'data5_99.dat'
    ]

train_dir = './../../../train_data/bin_data/20230202/'
#model_dir = './../../../model/20230202_2/'
model_dir = './../../../model/nomodel/'

additional_params = ''
for tfile in train_data:
    additional_params += ' ' + train_dir + tfile

#executable = 'adjust_param.out'
executable = 'adjust_param_cuda.exe'

cmd = executable + ' ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' ' + model_dir + phase + '.txt' + additional_params
print(cmd, file=sys.stderr)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
result = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
print(result)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '.txt', 'w') as f:
    f.write(param)
