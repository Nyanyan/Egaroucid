import subprocess

# Core i9-13900K
tasks = [ # start, end, n_threads, hash_level, exe, out_file
    [40, 59, 42, 25, 'Egaroucid_for_Console_7_3_0_x64_SIMD.exe', '0_ffo40_59_Core_i9-13900K_x64_SIMD.txt'],
    [40, 59, 42, 25, 'Egaroucid_for_Console_7_3_0_x64_Generic.exe', '1_ffo40_59_Core_i9-13900K_x64_Generic.txt'],
    [40, 59, 42, 25, 'Egaroucid_for_Console_7_3_0_x86_Generic.exe', '2_ffo40_59_Core_i9-13900K_x86_Generic.txt'],
]

for start, end, n_threads, hash_level, exe, out_file in tasks:
    cmd = 'python ffotest.py ' + str(start) + ' ' + str(end) + ' ' + str(n_threads) + ' ' + str(hash_level) + ' ' + str(exe)
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    n_lines = (end - start + 1) + 3
    res = ''
    for i in range(n_lines):
        line = p.stdout.readline().decode().replace('\r\n', '\n')
        print(line, end='')
        res += line
    with open('ffotest_result/' + out_file, 'w') as f:
        f.write(res)

