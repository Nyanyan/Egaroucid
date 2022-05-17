import subprocess

n_dense_pattern = 32
n_dense_additional = 8

for use_phase in range(30):
    if use_phase < 10:
        subprocess.run('python learn.py ' + str(n_dense_pattern) + ' ' + str(n_dense_additional) + ' ' + str(use_phase) + ' big_data_new_1.dat big_data_new_2.dat big_data_new_3.dat')
    else:
        subprocess.run('python learn.py ' + str(n_dense_pattern) + ' ' + str(n_dense_additional) + ' ' + str(use_phase) + ' big_data_new_4.dat big_data_new_6.dat big_data_new_7.dat big_data_new_8.dat big_data_new_9.dat big_data_new_10.dat')
