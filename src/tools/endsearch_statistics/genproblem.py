import subprocess

cmd = 'Egaroucid_for_console.exe -l 1'
p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

#tasks = [[d, 100] for d in range(5, 25)]
#tasks = [[d, 50] for d in range(25, 30)]

#tasks = [[d, 50] for d in range(30, 40)]
tasks = [[d, 50] for d in range(25, 30)]

for n_empties, n_problems in tasks:
    cmd_str = 'genproblem ' + str(n_empties) + ' ' + str(n_problems) + '\n'
    p.stdin.write(cmd_str.encode('utf-8'))
    p.stdin.flush()