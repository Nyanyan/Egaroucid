import subprocess

#'''
# Core i9-13900K
tasks = [ # is_egaroucid, start, end, n_threads, hash_level, exe, CPU, revision, file
    #40-59
    [True,  40, 59, 42, 25, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_SIMD.exe',       'Core_i9-13900K', 'SIMD',       '000_ffo40_59_Core_i9-13900K_SIMD.txt'],
    [True,  40, 59, 42, 25, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_Generic.exe',    'Core_i9-13900K', 'Generic',    '001_ffo40_59_Core_i9-13900K_Generic.txt'],

    #[False, 40, 59, 32, 25, 'versions/edax_4_5_2/wEdax-x64-modern.exe',                                     'Core_i9-13900K', 'x64_modern',     '010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt'],
    #[False, 40, 59, 32, 25, 'versions/edax_4_5_2/wEdax-x64.exe',                                            'Core_i9-13900K', 'x64',            '011_ffo40_59_Core_i9-13900K_edax_x64.txt'],

    [False, 40, 59, 32, 25, 'versions/edax_4_5_3/bin/wEdax-x64-modern.exe',                                     'Core_i9-13900K', 'x64_modern',     '010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt'],
    [False, 40, 59, 32, 25, 'versions/edax_4_5_3/bin/wEdax-x64.exe',                                            'Core_i9-13900K', 'x64',            '011_ffo40_59_Core_i9-13900K_edax_x64.txt'],
    
    #[False, 40, 59, 32, 25, 'versions/edax_4_6/wEdax-x86-64-v3.exe',                                     'Core_i9-13900K', 'v3',        '010_ffo40_59_Core_i9-13900K_edax_v3.txt'],
    #[False, 40, 59, 32, 25, 'versions/edax_4_6/wEdax-x86-64-v2.exe',                                     'Core_i9-13900K', 'v2',        '011_ffo40_59_Core_i9-13900K_edax_v2.txt'],
    #[False, 40, 59, 32, 25, 'versions/edax_4_6/wEdax-x86-64.exe',                                        'Core_i9-13900K', '-',        '012_ffo40_59_Core_i9-13900K_edax.txt'],

    #60-79
    #[True,  60, 79, 42, 30, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_SIMD.exe',      'Core_i9-13900K', 'x64_SIMD',       '020_ffo60_79_Core_i9-13900K_x64_SIMD.txt'],

    #[False, 60, 79, 32, 30, 'versions/edax_4_5_2/wEdax-x64-modern.exe',                                    'Core_i9-13900K', 'x64_modern',     '030_ffo60_79_Core_i9-13900K_edax_x64_modern.txt'],
]
#'''

'''
# Core i9-11900K
tasks = [ # is_egaroucid, start, end, n_threads, hash_level, exe, out_file
    #40-59
    #[True,  40, 59, 26, 25, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_AVX512.exe',     'Core_i9-11900K', 'AVX512',         '100_ffo40_59_Core_i9-11900K_AVX512.txt'],
    [True,  40, 59, 26, 25, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_SIMD.exe',       'Core_i9-11900K', 'SIMD',           '101_ffo40_59_Core_i9-11900K_SIMD.txt'],
    #[True,  40, 59, 26, 25, 'versions/Egaroucid_for_Console_beta/Egaroucid_for_Console_Generic.exe',    'Core_i9-11900K', 'Generic',        '102_ffo40_59_Core_i9-11900K_Generic.txt'],

    #[False, 40, 59, 16, 25, 'versions/edax_4_5_2/wEdax-x64-avx512.exe',                                     'Core_i9-11900K', 'x64_avx512',         '110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt'],
    #[False, 40, 59, 16, 25, 'versions/edax_4_5_2/wEdax-x64-modern.exe',                                     'Core_i9-11900K', 'x64_modern',         '111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt'],
    #[False, 40, 59, 16, 25, 'versions/edax_4_5_2/wEdax-x64.exe',                                            'Core_i9-11900K', 'x64',                '112_ffo40_59_Core_i9-11900K_edax_x64.txt'],

    #[False, 40, 59, 16, 25, 'versions/edax_4_5_3/bin/wEdax-x64-avx512.exe',                                     'Core_i9-11900K', 'x64_avx512',         '110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt'],
    #[False, 40, 59, 16, 25, 'versions/edax_4_5_3/bin/wEdax-x64-modern.exe',                                     'Core_i9-11900K', 'x64_modern',         '111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt'],
    #[False, 40, 59, 16, 25, 'versions/edax_4_5_3/bin/wEdax-x64.exe',                                            'Core_i9-11900K', 'x64',                '112_ffo40_59_Core_i9-11900K_edax_x64.txt'],
]
#'''

import datetime
now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
summary_file = 'ffotest_result/summary_' + now + '.txt'

for is_egaroucid, start, end, n_threads, hash_level, exe, cpu, revision, out_file in tasks:
    if is_egaroucid:
        cmd = 'python ffotest.py ' + str(start) + ' ' + str(end) + ' ' + str(n_threads) + ' ' + str(hash_level) + ' ' + exe
        n_lines = (end - start + 1) + 3 + 6
    else:
        cmd = 'python ffotest_edax.py ' + str(start) + ' ' + str(end) + ' ' + str(n_threads) + ' ' + str(hash_level) + ' ' + exe
        n_lines = (end - start + 1) + 8
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    res = ''
    for i in range(n_lines):
        line = p.stdout.readline().decode().replace('\r', '').replace('\n\n', '\n')
        print(line, end='')
        res += line
    summary = ''
    if is_egaroucid:
        info_line = res.splitlines()[-1].split()
        tim = info_line[4][:-1]
        n_nodes = info_line[1]
        nps = info_line[6]
        summary = 'Egaroucid,' + cpu + ',' + revision + ',' + tim + ',' + n_nodes + ',' + nps + ',' + out_file + '\n'
    else:
        info_line = res.splitlines()[-4].replace('(', '').split()
        tim_str = info_line[4]
        tim_min = float(tim_str.split(':')[0])
        tim = str(float(tim_str.split(':')[1]) + tim_min * 60)
        n_nodes = info_line[1]
        nps = info_line[5]
        summary = 'Edax,' + cpu + ',' + revision + ',' + tim + ',' + n_nodes + ',' + nps + ',' + out_file + '\n'
    with open('ffotest_result/' + out_file, 'w') as f:
        f.write(res)
    with open(summary_file, 'a') as f:
        f.write(summary)

