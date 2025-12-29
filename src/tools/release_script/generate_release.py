# What this code can do: 
# Egaroucid for Console : do everything
# Egaroucid Installer   : generate GUID, run inno setup
# Egaroucid Portable    : do everything

import os
import sys
import glob
import shutil

# common
if len(sys.argv) != 2:
    print('[ERROR] please execute `python generate_release.py X.Y.Z`')
    exit(1)
VERSION_DOT = sys.argv[1]
VERSION_UNDERBAR = VERSION_DOT.replace('.', '_')
FORMAT_FILES_DIR = 'format_files'
COMMON_FILES_IN_DIR = FORMAT_FILES_DIR + '/0_common_files'
DST_ROOT = './../../../release'

# special tasks
with open(COMMON_FILES_IN_DIR + '/tasks.txt', 'r') as f:
    common_files_tasks = [list(elem.split()) for elem in f.read().splitlines()] # [[type, file, dst], [type, file, dst], ...]

# main directory
os.mkdir(DST_ROOT + '/' + VERSION_UNDERBAR)


def copy_common_files(correct_task_type, dst_dir_root):
    for task_type, filename, dst_dir in common_files_tasks: 
        if task_type != correct_task_type:
            continue
        common_file_in_dir = COMMON_FILES_IN_DIR + '/' + filename
        common_file_dst_dir = dst_dir_root + '/' + dst_dir
        if os.path.isfile(common_file_in_dir):
            os.makedirs(common_file_dst_dir, exist_ok=True) # ensure parent directory exists
            shutil.copy2(common_file_in_dir, common_file_dst_dir) # copy common special files
        else:
            shutil.copytree(common_file_in_dir, common_file_dst_dir, dirs_exist_ok=True) # copy common special files
    

#'''
# Egaroucid for Console
print('\n')
print('<<<<<<<< Egaroucid for Console >>>>>>>>')
CONSOLE_DST_DIR = DST_ROOT + '/' + VERSION_UNDERBAR + '/console'
CONSOLE_IN_EXES_DIR = FORMAT_FILES_DIR + '/1_console_exes'
CONSOLE_IN_FILES_DIR = FORMAT_FILES_DIR + '/console_files'
os.mkdir(CONSOLE_DST_DIR)
console_exes = glob.glob(CONSOLE_IN_EXES_DIR + '/*.exe')
console_dir_names = ['.'.join(elem.replace('\\', '/').split('/')[-1].split('.')[:-1]) for elem in console_exes]
for i in range(len(console_dir_names)):
    separated = console_dir_names[i].split(VERSION_UNDERBAR)
    console_dir_names[i] = separated[0] + VERSION_UNDERBAR + '_Windows' + separated[1] # add 'Windows'
for console_dir_name, console_exe in zip(console_dir_names, console_exes):
    print(console_dir_name)
    console_dir = CONSOLE_DST_DIR + '/' + console_dir_name
    os.mkdir(console_dir)
    shutil.copy2(console_exe, console_dir) # copy main executable
    shutil.copytree(CONSOLE_IN_FILES_DIR, console_dir, dirs_exist_ok=True) # copy other resources
    copy_common_files('console', console_dir) # copy common special files
    shutil.make_archive(console_dir, format='zip', root_dir = CONSOLE_DST_DIR, base_dir = console_dir_name) # zip archive with folder inside
#'''



#'''
# Egaroucid Installer
print('\n')
print('<<<<<<<< Egaroucid Installer >>>>>>>>')
INSTALLER_DST_DIR = DST_ROOT + '/' + VERSION_UNDERBAR + '/GUI_Installer'
INSTALLER_DST_FILES_DIR = INSTALLER_DST_DIR + '/files'
INSTALLER_DST_INSTALLER_DIR = INSTALLER_DST_DIR + '/installer'
INSTALLER_IN_EXES_DIR = FORMAT_FILES_DIR + '/2_GUI_Installer_exes'
INSTALLER_IN_FILES_DIR = FORMAT_FILES_DIR + '/GUI_Installer_files'
INSTALLER_IN_SETUP_FILE = FORMAT_FILES_DIR + '/GUI_Installer_setup/egaroucid_setup.iss'
os.mkdir(INSTALLER_DST_DIR)
installer_exes = glob.glob(INSTALLER_IN_EXES_DIR + '/*.exe')
os.mkdir(INSTALLER_DST_INSTALLER_DIR)
for installer_exe in installer_exes:
    print(installer_exe)
    shutil.copy2(installer_exe, INSTALLER_DST_INSTALLER_DIR) # copy main executable
shutil.copy2(INSTALLER_IN_SETUP_FILE, INSTALLER_DST_INSTALLER_DIR) # copy setup file
with open(INSTALLER_DST_INSTALLER_DIR + '/egaroucid_setup.iss', 'r', encoding='utf-8') as f:
    installer_setup = f.read()
installer_setup = installer_setup.replace('REPLACE_VERSION_DOT', VERSION_DOT).replace('REPLACE_VERSION_UNDERBAR', VERSION_UNDERBAR) # replace version information
with open(INSTALLER_DST_INSTALLER_DIR + '/egaroucid_setup.iss', 'w', encoding='utf-8') as f:
    f.write(installer_setup) # rewrite setup file
shutil.copytree(INSTALLER_IN_FILES_DIR, INSTALLER_DST_FILES_DIR, dirs_exist_ok=True) # copy files
copy_common_files('installer', INSTALLER_DST_FILES_DIR) # copy common special files
#'''




#'''
# Egaroucid Portable
print('\n')
print('<<<<<<<< Egaroucid Portable >>>>>>>>')
PORTABLE_DST_DIR = DST_ROOT + '/' + VERSION_UNDERBAR + '/GUI_Portable'
PORTABLE_IN_EXES_DIR = FORMAT_FILES_DIR + '/3_GUI_Portable_exes'
PORTABLE_IN_FILES_DIR = FORMAT_FILES_DIR + '/GUI_Portable_files'
os.mkdir(PORTABLE_DST_DIR)
portable_exes = glob.glob(PORTABLE_IN_EXES_DIR + '/*.exe')
portable_dir_names = ['.'.join(elem.replace('\\', '/').split('/')[-1].split('.')[:-1]) for elem in portable_exes]
for i in range(len(portable_dir_names)):
    separated = portable_dir_names[i].split(VERSION_UNDERBAR)
    portable_dir_names[i] = separated[0] + VERSION_UNDERBAR + '_Windows' + separated[1] # add 'Windows'
for portable_dir_name, portable_exe in zip(portable_dir_names, portable_exes):
    print(portable_dir_name)
    portable_dir = PORTABLE_DST_DIR + '/' + portable_dir_name
    os.mkdir(portable_dir)
    shutil.copy2(portable_exe, portable_dir) # copy main executable
    shutil.copytree(PORTABLE_IN_FILES_DIR, portable_dir, dirs_exist_ok=True) # copy other resources
    copy_common_files('portable', portable_dir) # copy common special files
    shutil.make_archive(portable_dir, format='zip', root_dir = PORTABLE_DST_DIR, base_dir = portable_dir_name) # zip archive with folder inside
#'''