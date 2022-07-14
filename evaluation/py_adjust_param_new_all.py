import subprocess

#task_phases = [1, 0]
task_phases = list(reversed(range(58)))
#task_phases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

n_parallel = 1

def do_task(i, i_e):
    tasks = []
    for j in range(i, i_e):
        tasks.append(subprocess.Popen(('python py_adjust_param_new.py ' + str(task_phases[j])).split(), stderr=None, stdout=subprocess.PIPE))
    for j in range(i, i_e):
        result = tasks[j - i].stdout.read().decode().replace('\r', '').replace('\n', '')
        with open('auto_learn_log.txt', 'a') as f:
            f.write(str(task_phases[j]) + '\t' + result + '\n')

for i in range(0, len(task_phases), n_parallel):
    e = i + n_parallel
    if e > len(task_phases):
        e = len(task_phases)
    print(task_phases[i], 'to', task_phases[e - 1])
    do_task(i, e)
