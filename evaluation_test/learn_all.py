import subprocess

n_dense_pattern = 64
n_dense_additional = 8

for use_phase in reversed(range(30)):
    if use_phase < 10:
        subprocess.run('python learn.py ' + str(n_dense_pattern) + ' ' + str(n_dense_additional) + ' ' + str(use_phase) + ' data_0000001.dat data_0000002.dat data_0000003.dat')
    else:
        subprocess.run('python learn.py ' + str(n_dense_pattern) + ' ' + str(n_dense_additional) + ' ' + str(use_phase) + ' data_0000004.dat data_0000006.dat data_0000007.dat data_0000008.dat data_0000009.dat data_0000010.dat')
