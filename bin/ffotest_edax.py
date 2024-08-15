# The FFO endgame test suite
# http://radagast.se/othello/ffotest.html
# https://github.com/abulmo/edax-reversi/tree/master/problem

import subprocess
import sys

start = 40
end = 59
n_threads = 32
hash_level = 25
exe = 'versions/edax_4_5_2/wEdax-x64-modern.exe'
try:
    if len(sys.argv) >= 2:
        start = int(sys.argv[1])
    if len(sys.argv) >= 3:
        end = int(sys.argv[2])
    if len(sys.argv) >= 4:
        n_threads = int(sys.argv[3])
    if len(sys.argv) >= 5:
        hash_level = int(sys.argv[4])
    if len(sys.argv) >= 6:
        exe = sys.argv[5]
except:
    print('usage: python ffotest.py [start=40] [end=59] [n_threads=32] [hash_level=25] [exe=versions/edax_4_5_2/wEdax-x64-modern.exe]')
    exit()

cmd = exe + ' -l 60 -h ' + str(hash_level) + ' -solve problem/ffo' + str(start) + '-' + str(end) + '.txt -n ' + str(n_threads)

print(cmd)

edax = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for i in range(5):
    line = edax.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line, flush=True)
for i in range(start, end + 1):
    line = edax.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line, flush=True)
for i in range(2):
    line = edax.stdout.readline().decode().replace('\n', '').replace('\r', '')
    print(line, flush=True)
edax.kill()
