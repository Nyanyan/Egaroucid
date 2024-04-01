import subprocess

tasks = [
    [1, 1000],
    [5, 1000],
    [10, 1000],
    #[15, 250],
    [21, 100]
]

res = ''
for level, n_games in tasks:
    cmd = 'python egaroucid_vs_edax.py ' + str(level) + ' ' + str(n_games)
    print(cmd)
    subprocess.run(cmd.split())
    #p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    #res += p.stdout.readline().decode()
print(res)
