import subprocess

for i in range(-62, 62, 2):
    in_file = 'a'
    cmd = 'SGD_move_ordering_end.exe ' + str(i) + ' 0 1 0 512.0 '
    cmd += in_file + ' '
    cmd += './../../../train_data/bin_mo_data/20230211/15.dat '
    cmd += './../../../train_data/bin_mo_data/20230211/16.dat '
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    log = p.stdout.readline().decode().replace('\r\n', '\n').replace('\n', '')
    with open('opt_log.txt', 'a') as f:
        f.write(log + '\n')
    param = p.stdout.read().decode().replace('\r\n', '\n')
    with open(str((i + 62) // 2) + '.txt', 'w') as f:
        f.write(param)