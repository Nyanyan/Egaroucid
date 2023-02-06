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
    minute = '10'
    second = '0'
    beta = '0.002'

if int(phase) < 20:
    train_data_nums = [1, 2, 3, 4, 6, 7, 8, 9, 11, 15, 16, 99]
else:
    train_data_nums = [3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 99]

train_data = [str(elem) + '.dat' for elem in train_data_nums]

train_dir = './../../../train_data/bin_data/20230205/' + str(4 + int(phase)) + '/'
model_dir = './../../../model/20230205_2/'
#model_dir = './../../../model/nomodel/'

additional_params = ''
for tfile in train_data:
    additional_params += ' ' + train_dir + tfile

#executable = 'gd_eval.out'
executable = 'sgd_eval_cuda.exe'

cmd = executable + ' ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + beta + ' ' + model_dir + phase + '.txt' + additional_params
print(cmd, file=sys.stderr)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
result = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
print(result)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open(phase + '.txt', 'w') as f:
    f.write(param)
