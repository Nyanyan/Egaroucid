import subprocess
import sys

strt_phase = int(sys.argv[1])
end_phase = int(sys.argv[2])

for phase in range(strt_phase, end_phase + 1):
    print('optimizing phase', phase)
    cmd = 'python eval_optimizer_phase.py ' + str(phase)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    line = p.stdout.readline().decode().replace('\r', '').replace('\n', '')
    with open('trained/opt_log.txt', 'a') as f:
        f.write(line + '\n')