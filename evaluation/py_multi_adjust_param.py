import subprocess

for phase in reversed(range(5, 15)):
    for player in range(2):
        cmd = 'python py_adjust_param.py ' + str(phase) + ' ' + str(player)
        subprocess.run(cmd, shell=True, check=True)