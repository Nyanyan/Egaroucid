import subprocess

N_PHASES = 30

for phase in range(N_PHASES):
    print('optimizing phase', phase)
    cmd = 'python py_adjust_param.py ' + str(phase)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    line = p.stdout.readline()
    with open('opt_log.txt', 'a') as f:
        f.write(line + '\n')