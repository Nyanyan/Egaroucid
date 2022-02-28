import subprocess

'''
cmd = 'copy /b '
strt = 4
end = 8
for i in range(strt, end + 1):
    cmd += 'big_data_' + str(i) + '.txt'
    if i == end:
        cmd += ' big_data.txt'
    else:
        cmd += ' + '
print(cmd)
'''

cmd = 'copy /b big_data_4.txt + big_data_6.txt + big_data_7.txt + big_data_8.txt + big_data_9.txt + big_data_10.txt big_data.txt'

subprocess.run(cmd, shell=True, check=True)