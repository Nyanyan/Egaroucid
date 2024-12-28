import subprocess


for n_random_moves in [6, 5, 4, 3, 2, 1, 0]:
    cmd = 'python autoplay.py ' + str(n_random_moves) + ' 0 0'
    print(cmd)
    subprocess.run(cmd)
