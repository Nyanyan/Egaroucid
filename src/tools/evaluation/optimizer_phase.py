import subprocess
import sys

phase = str(sys.argv[1])
if len(sys.argv) > 3:
    hour = str(sys.argv[2])
    minute = str(sys.argv[3])
    second = str(sys.argv[4])
    alpha = str(sys.argv[5])
else:
    hour = '0'
    minute = '0'
    second = '10'
    alpha = '200'

if int(phase) <= 10:
    train_data_nums = [23] # book data
    n_patience = '20'
elif int(phase) <= 18:
    train_data_nums = [20, 21, 22] # begins with all first11
    n_patience = '5'
else:
    train_data_nums = [3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 99] # no random moves!
    n_patience = '3'

train_data = [str(elem) + '.dat' for elem in train_data_nums]

train_root_dir = './../../../train_data/bin_data/20240223_1/'

#train_dirs = [train_root_dir + str(elem) + '/' for elem in range(int(phase) * 2, int(phase) * 2 + 2)]
train_dirs = [train_root_dir + str(int(phase)) + '/']

model_dir = './../../../model/nomodel/'

#model_dir = './../../../model/20240223_2/'

additional_params = ''
for tfile in train_data:
    for train_dir in train_dirs:
        additional_params += ' ' + train_dir + tfile

executable = 'eval_optimizer_cuda_12_2_0.exe'

cmd = executable + ' ' + phase + ' ' + hour + ' ' + minute + ' ' + second + ' ' + alpha + ' ' + n_patience + ' ' + model_dir + phase + '.txt' + additional_params
print(cmd, file=sys.stderr)
p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
result = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
print(result)
param = p.stdout.read().decode().replace('\r\n', '\n')
with open('trained/' + phase + '.txt', 'w') as f:
    f.write(param)
