import subprocess
import sys

phase = str(sys.argv[1])

# 7.0
train_data_nums = [26, 27, 28, 29, 30, 31, 34, 35]
if int(phase) <= 11:
    train_data_nums = [26] # use book only
if 31 <= int(phase):
    train_data_nums.extend([15, 16, 17, 18, 19, 20, 21, 24, 25]) # use more data
train_data_nums.sort()
train_root_dir = './../../../train_data/bin_data/20240223_1/'


train_data = [str(elem) + '.dat' for elem in train_data_nums]
train_dirs = [train_root_dir + str(int(phase)) + '/']

cmd = 'histgram.out '
for tfile in train_data:
    for train_dir in train_dirs:
        cmd += ' ' + train_dir + tfile

print(cmd, file=sys.stderr)
p = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
output = p.stdout.decode()
print(output)