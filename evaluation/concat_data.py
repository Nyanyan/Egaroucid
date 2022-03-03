import subprocess

#cmd = 'copy /b big_data_4.txt + big_data_6.txt + big_data_7.txt + big_data_8.txt + big_data_9.txt + big_data_10.txt big_data.txt'
cmd = 'copy /b big_data_new_4.txt + big_data_new_6.txt + big_data_new_7.txt + big_data_new_8.txt + big_data_new_9.txt + big_data_new_10.txt big_data.txt'

subprocess.run(cmd, shell=True, check=True)