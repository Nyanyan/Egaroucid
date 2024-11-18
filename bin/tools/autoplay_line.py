import subprocess
from tqdm import trange
import os
import sys

line_dr = './../problem/etc/first12_all_shuffled'
out_dr = './../transcript/first12_all_shuffled'

exe = './../versions/Egaroucid_for_Console_beta/Egaroucid_for_Console.exe'


#IDX_START = int(sys.argv[1])
#IDX_END = int(sys.argv[2])

# IDX_START = 10
# IDX_END = 100

#print(IDX_START, IDX_END)


LEVEL = 11
N_GAMES_PER_FILE = 10000
N_THREAD = 30

def fill0(n, r):
    res = str(n)
    return '0' * (r - len(res)) + res

idx_lst = [
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
    162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 
    222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 
    252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 
    282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 
    312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 
    342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 
    372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 
    402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 
    432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 
    462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 
    492, 493, 494, 495, 496, 497, 498, 499

]
idx_lst2 = [
    4, 14, 24, 34, 44, 54, 64, 
    81, 101, 131, 133, 161, 191, 221, 251, 281, 311, 341, 371, 401, 431, 461, 491
]

IDX_START = 766
IDX_END = 999

for idx in range(IDX_START, IDX_END + 1):
    print(fill0(idx, 7))
    file = line_dr + '/' + fill0(idx, 7) + '.txt'
    cmd = exe + ' -nobook -l ' + str(LEVEL) + ' -thread ' + str(N_THREAD) + ' -selfplayline ' + file
    print(cmd)
    with open(out_dr + '/' + fill0(idx, 7) + '.txt', 'w') as f:
        egaroucid = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        for i in trange(N_GAMES_PER_FILE):
            line = egaroucid.stdout.readline().decode().replace('\r', '').replace('\n', '') + '\n'
            f.write(line)
        egaroucid.kill()
