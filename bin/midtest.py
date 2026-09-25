import subprocess
import sys
import os

def fill0(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def parse_args(args, names):
    # positional (in the order of names) or name=value
    n_positional = 0
    for arg in sys.argv[1:]:
        name, sep, value = arg.partition('=')
        if sep and name.isidentifier():
            if not (name in names):
                raise ValueError('unknown option ' + name)
        else:
            if n_positional >= len(names):
                raise ValueError('too many arguments')
            name = names[n_positional]
            value = arg
            n_positional += 1
        args[name] = type(args[name])(value)

args = {
    'level': 23,
    'n_threads': 32,
    'hash_level': 25,
    'exe': 'Egaroucid_for_Console.exe',
    'eval_file': '',
}
try:
    parse_args(args, ['level', 'n_threads', 'hash_level', 'exe', 'eval_file'])
except Exception as e:
    print(e)
    print('usage: python midtest.py [level=23] [n_threads=32] [hash_level=25] [exe=Egaroucid_for_Console.exe] [eval_file=]')
    print('arguments can be given in this order or as name=value (e.g. exe=Egaroucid_for_Console_clang.exe)')
    exit()
level = args['level']
n_threads = args['n_threads']
hash_level = args['hash_level']
exe = args['exe']
eval_file = args['eval_file']


script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(exe):
    exe = os.path.join(script_dir, exe)

cmd_version = exe + ' -v'
# print(cmd_version)
version = subprocess.run((cmd_version).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE).stdout.decode()

def strip_newlines(s):
    while s.endswith('\n') or s.endswith('\r'):
        s = s[:-1]
    return s

version = strip_newlines(version)
print(version)

cmd = exe + ' -l ' + str(level) + ' -nobook -thread ' + str(n_threads) + ' -hash ' + str(hash_level)
if eval_file != '':
    cmd += ' -eval ' + eval_file
cmd += ' -solve ' + os.path.join(script_dir, 'problem/midgame_test.txt')

print(cmd.replace(script_dir, 'script_dir'))
egaroucid = subprocess.Popen((cmd).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

res = ''
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print('#   ' + line)
for i in range(32):
    line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
    policy = line.split()[3][:-1]
    line = '#' + fill0(i, 2) + ' ' + line
    print(line)
    res += line + '\n'
line = egaroucid.stdout.readline().decode().replace('\n', '').replace('\r', '')
print(line)
egaroucid.kill()
