import subprocess


for n_random_moves in [9, 10, 11]:
    cmd = 'python autoplay.py ' + str(n_random_moves) + ' 0 0'
    print(cmd)
    subprocess.run(cmd)
