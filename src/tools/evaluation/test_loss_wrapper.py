import subprocess

N_PHASES = 60
data_root_dir = './../../../train_data/bin_data/20240223_1/'
exe = 'test_loss.out'
eval_file = 'trained/eval.egev'
data_nums = []
for _ in range(12):
    data_nums.append([26])
for _ in range(12, 61):
    data_nums.append([32])

res = ''
for phase in range(N_PHASES):
    cmd = exe + ' ' + eval_file + ' ' + str(phase)
    for num in data_nums[phase]:
        cmd += ' ' + data_root_dir + str(phase) + '/' + str(num) + '.dat'
    print(cmd)
    p = subprocess.run(cmd, stdout=subprocess.PIPE)
    out = p.stdout.decode().replace('\r', '').replace('\n', '')
    res += out + '\n'
print('')
print('all done')
print(res)
with open('trained/loss.txt', 'a') as f:
    f.write(res)