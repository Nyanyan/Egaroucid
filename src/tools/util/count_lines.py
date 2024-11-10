from glob import glob
import subprocess

drs = [
    './../../engine/*.hpp',
    './../../console/*.hpp',
    './../../gui/*.hpp',
    './../../*.cpp',
]

files = []
for dr in drs:
    files.extend(glob(dr))

print(len(files), 'files')

n_lines = 0
for file in files:
    cmd = 'find /v /c "" ' + file
    p = subprocess.run(cmd, stdout=subprocess.PIPE)
    out = int(p.stdout.decode().split()[-1])
    #print(file, out)
    n_lines += out
print(n_lines, 'lines')