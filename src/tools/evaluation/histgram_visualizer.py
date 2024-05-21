import subprocess
import sys
import matplotlib.pyplot as plt

phase = str(sys.argv[1])
CHECK_PATTERN = 5

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
#print(output)

output = output.splitlines()

pattern_data = [[] for _ in range(18)]
for line in output:
    line_sp = line.split()
    if line_sp[0] == 'score':
        pass
    elif line_sp[0] == 'pattern':
        pattern, n_appear, n = [int(elem) for elem in line_sp[1:]]
        if pattern == CHECK_PATTERN:
            while len(pattern_data[pattern]) < n_appear:
                pattern_data[pattern].append(0)
            if n_appear == 0:
                pattern_data[pattern].append(0)
            else:
                pattern_data[pattern].append(n)

plt.plot(range(len(pattern_data[CHECK_PATTERN])), pattern_data[CHECK_PATTERN])
plt.xlabel('index duplication <-unique common->')
plt.ylabel('number of such pattern')
plt.show()