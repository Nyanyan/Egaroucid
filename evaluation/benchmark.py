import subprocess

ans = []
for phase in range(15):
    for player in range(2):
        cmd = 'evaluation_benchmark.out ' + str(phase) + ' ' + str(player) + ' learned_data/' + str(phase) + '_' + str(player) + '.txt'
        print(cmd)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        res = p.stdout.read().decode().replace('\r\n', '\n').replace('\n', '')
        ans.append(res)
        print(res)

for res in ans:
    print(res)