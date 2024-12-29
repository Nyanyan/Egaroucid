import subprocess


for n_random_moves in [5, 4, 3, 2, 1]:
    cmd = 'python autoplay.py ' + str(n_random_moves) + ' 0 0'
    print(cmd)
    subprocess.run(cmd)
