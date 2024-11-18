import subprocess
from glob import glob

files = glob('problems/*_problem.txt')
files.sort(key=lambda x: int(x.split('\\')[1].split('_')[0]))
print(files)

tasks = [[d, 50] for d in range(26, 40)]
depth_to_93p_level = {
    26: 13, 
    27: 13, 
    28: 20, 
    29: 21, 
    30: 21,
    31: 23, 
    32: 23, 
    33: 27, 
    34: 27, 
    35: 30, 
    36: 30, 
    37: 34, 
    38: 34, 
    39: 37, 
}


task_idx = 0
for file in files:
    d = tasks[task_idx][0]
    n = tasks[task_idx][1]
    level = depth_to_93p_level[d]
    cmd = 'Egaroucid_for_console.exe -l ' + str(level) + ' -nobook -solve ' + file
    p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print(d, n)
    p.stdout.readline() # header
    avg_n_nodes = 0
    for _ in range(n):
        line = p.stdout.readline().decode().replace('\r', '').replace('\n', '')
        n_nodes = int(line.split()[6][:-1])
        nps = int(line.split()[7][:-1])
        with open('endgame_search.csv', 'a') as f:
            f.write(str(d) + ',' + str(n_nodes) + ',' + str(nps) + '\n')
        avg_n_nodes += n_nodes
    avg_n_nodes /= n
    print(d, 'avg', avg_n_nodes)
    task_idx += 1
    p.kill()
