import subprocess
import sys

mode = 2
exe = 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe'

ans = [
    [],
    [0, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005288, 24571284, 212258800, 1939886636, 18429641748, 184042084512, 1891832540064, 20301186039128], # from https://aartbik.blogspot.com/2009/02/perft-for-reversi.html
    [0, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005320, 24571420, 212260880, 1939899208, 18429791868, 184043158384, 1891845643044]
]

print('perft mode', mode)
for depth in range(1, 16):
    cmd = exe + ' -perft ' + str(depth) + ' ' + str(mode)
    egaroucid = subprocess.run(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = egaroucid.stdout.decode().replace('\r', '').replace('\n', '')
    n_leaves = int(result.split()[5])
    if depth < len(ans[mode]):
        wrong_ans = False
        if ans[mode][depth] != n_leaves:
            wrong_ans = True
        if wrong_ans:
            print(result, 'WRONG ANSWER')
        else:
            print(result)
    else:
        print(result, '?')
