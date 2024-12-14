import sys
import datetime

if len(sys.argv) < 2:
    print('please input [version] [date]')
version = sys.argv[1]

if len(sys.argv) == 3:
    date = sys.argv[2]
else:
    date = datetime.date.today()

citation_file = './../../../CITATION.cff'

with open(citation_file, 'r') as f:
    data = f.read().splitlines()

data_proc = []

for datum in data:
    if datum[:9] == 'version: ':
        datum = 'version: ' + version
    elif datum[:15] == 'date-released: ':
        datum = 'date-released: ' + date
    data_proc.append(datum)

with open(citation_file, 'w') as f:
    for datum in data:
        f.write(datum + '\n')