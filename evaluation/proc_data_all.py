import subprocess

cmds = [
    'proc_data_idx_new2.out records1 0 127 data3_01.dat',
    'proc_data_idx_new2.out records2 0 56 data3_02.dat',
    'proc_data_idx_new2.out records3 0 435 data3_03.dat',
    'proc_data_idx_new2.out records4 0 435 data3_04.dat',
    'proc_data_idx_new2.out records6 0 48 data3_06.dat',
    'proc_data_idx_new2.out records7 0 46 data3_07.dat',
    'proc_data_idx_new2.out records8 0 5 data3_08.dat',
    'proc_data_idx_new2.out records9 0 173 data3_09.dat',
    'proc_data_idx_new2.out records10 0 6 data3_10.dat',
    'proc_data_idx_new2.out records11 0 3 data3_11.dat',
    'proc_data_idx_new2.out records15 0 270 data3_15_0.dat',
    'proc_data_idx_new2.out records15 270 540 data3_15_1.dat',
    'proc_data_idx_new2.out records9999999 0 12 data3_99.dat',
]

tasks = []
for cmd in cmds:
    tasks.append(subprocess.Popen(cmd.split(), stderr=None, stdout=subprocess.DEVNULL))
for task in tasks:
    task.wait()
