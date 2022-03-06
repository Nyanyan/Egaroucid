import subprocess
import sys

if len(sys.argv) != 2:
    print('arg err')
    exit()

cmd = 'check_data_diversity_bin.out big_data_new_4.dat big_data_new_6.dat   big_data_new_7.dat   big_data_new_8.dat   big_data_new_9.dat   big_data_new_10.dat'

p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
res = p.stdout.read().decode().splitlines()

with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(res))