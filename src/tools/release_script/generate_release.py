# What this code can do: 
# Egaroucid for Console : do everything
# Egaroucid Installer   : generate GUID, run inno setup
# Egaroucid Portable    : do everything

import os
import sys
import glob
import shutil
import argparse

# common
parser = argparse.ArgumentParser(description='Generate Egaroucid release files.')
parser.add_argument('version', metavar='X.Y.Z')
parser.add_argument('--console', action='store_true', help='generate console release')
parser.add_argument('--gui', action='store_true', help='generate GUI installer and portable releases')
args = parser.parse_args()
if not args.console and not args.gui:
    args.console = True
    args.gui = True

VERSION_DOT = args.version
VERSION_UNDERBAR = VERSION_DOT.replace('.', '_')
FORMAT_FILES_DIR = 'format_files'
COMMON_FILES_IN_DIR = FORMAT_FILES_DIR + '/0_common_files'
DST_ROOT = './../../../release'

# special tasks
with open(COMMON_FILES_IN_DIR + '/tasks.txt', 'r') as f:
    common_files_tasks = [list(elem.split()) for elem in f.read().splitlines()] # [[type, file, dst], [type, file, dst], ...]

for elem in common_files_tasks:
    print(elem)

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
            print(common_file_in_dir, common_file_dst_dir)
    

def versioned_exes(exes_dir):
    all_exes = sorted(glob.glob(exes_dir + '/*.exe'))
    filtered_exes = sorted(glob.glob(exes_dir + '/*_' + VERSION_UNDERBAR + '_*.exe'))
    skipped_exes = [elem for elem in all_exes if elem not in filtered_exes]
    for skipped_exe in skipped_exes:
        print('[SKIP] version mismatch: ' + skipped_exe)
    if len(filtered_exes) == 0:
        print('[ERROR] no executables for version ' + VERSION_UNDERBAR + ' in ' + exes_dir)
        sys.exit(1)
    return filtered_exes


def windows_dir_name(exe_path):
    exe_name = os.path.splitext(os.path.basename(exe_path))[0]
    separated = exe_name.split(VERSION_UNDERBAR, 1)
    if len(separated) != 2:
        print('[ERROR] invalid executable name for version ' + VERSION_UNDERBAR + ': ' + exe_path)
        sys.exit(1)
    return separated[0] + VERSION_UNDERBAR + '_Windows' + separated[1] # add 'Windows'


def generate_console_release():
    # Egaroucid for Console
    print('\n')
    print('<<<<<<<< Egaroucid for Console >>>>>>>>')
    CONSOLE_DST_DIR = DST_ROOT + '/' + VERSION_UNDERBAR + '/console'
    CONSOLE_IN_EXES_DIR = FORMAT_FILES_DIR + '/1_console_exes'
    CONSOLE_IN_FILES_DIR = FORMAT_FILES_DIR + '/console_files'
    os.mkdir(CONSOLE_DST_DIR)
    console_exes = versioned_exes(CONSOLE_IN_EXES_DIR)
    for console_exe in console_exes:
        console_dir_name = windows_dir_name(console_exe)
        print(console_dir_name)
        console_dir = CONSOLE_DST_DIR + '/' + console_dir_name
        os.mkdir(console_dir)
        shutil.copy2(console_exe, console_dir) # copy main executable
        shutil.copytree(CONSOLE_IN_FILES_DIR, console_dir, dirs_exist_ok=True) # copy other resources
        copy_common_files('console', console_dir) # copy common special files
        shutil.make_archive(console_dir, format='zip', root_dir = CONSOLE_DST_DIR, base_dir = console_dir_name) # zip archive with folder inside


def generate_gui_installer_release():
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
    installer_exes = versioned_exes(INSTALLER_IN_EXES_DIR)
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


def generate_gui_portable_release():
    # Egaroucid Portable
    print('\n')
    print('<<<<<<<< Egaroucid Portable >>>>>>>>')
    PORTABLE_DST_DIR = DST_ROOT + '/' + VERSION_UNDERBAR + '/GUI_Portable'
    PORTABLE_IN_EXES_DIR = FORMAT_FILES_DIR + '/3_GUI_Portable_exes'
    PORTABLE_IN_FILES_DIR = FORMAT_FILES_DIR + '/GUI_Portable_files'
    os.mkdir(PORTABLE_DST_DIR)
    portable_exes = versioned_exes(PORTABLE_IN_EXES_DIR)
    for portable_exe in portable_exes:
        portable_dir_name = windows_dir_name(portable_exe)
        print(portable_dir_name)
        portable_dir = PORTABLE_DST_DIR + '/' + portable_dir_name
        os.mkdir(portable_dir)
        shutil.copy2(portable_exe, portable_dir) # copy main executable
        shutil.copytree(PORTABLE_IN_FILES_DIR, portable_dir, dirs_exist_ok=True) # copy other resources
        copy_common_files('portable', portable_dir) # copy common special files
        shutil.make_archive(portable_dir, format='zip', root_dir = PORTABLE_DST_DIR, base_dir = portable_dir_name) # zip archive with folder inside


if args.console:
    generate_console_release()

if args.gui:
    generate_gui_installer_release()
    generate_gui_portable_release()
