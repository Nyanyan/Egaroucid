import subprocess
import sys

N_PHASES = 60

try:
    strt_phase = int(sys.argv[1])
    end_phase = int(sys.argv[2])
except:
    strt_phase = 0
    end_phase = N_PHASES

for phase in range(strt_phase, end_phase):
    print('optimizing phase', phase)
    cmd = 'python py_adjust_param.py ' + str(phase)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    line = p.stdout.readline().decode().replace('\r', '').replace('\n', '')
    with open('opt_log.txt', 'a') as f:
        f.write(line + '\n')