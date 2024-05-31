import pyperclip
import sys
import glob


if sys.argv[1] == 'ja':
    head = '''<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
'''
else:
    head = '''<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
'''


files = glob.glob('./ffotest_result/*.txt')

# CPU, version, time, nodes, NPS
results = []
for file in files:
    with open(file, 'r') as f:
        last_line_split = f.read().splitlines()[-1].split()
    filename = file.replace('\\', '/').split('/')[-1]
    cpu = ' '.join(filename.split('_')[1:-1])
    edition = filename.split('_')[-1].split('.')[0]
    nodes = last_line_split[1]
    time = last_line_split[4][:-1]
    nps = last_line_split[6]
    print(cpu, edition, time, nodes, nps)
    results.append([cpu, edition, time, nodes, nps])

res = head
for result, file in zip(results, files):
    res += '<tr>\n'
    for elem in result:
        res += '<td>' + elem + '</td>'
    filename = file.replace('\\', '/').split('/')[-1]
    res += '<td><a href="' + './files/' + filename + '">' + filename + '</a></td>'
    res += '\n'
    res += '</tr>\n'

res += '''</table>
</div>'''

#print(res)

pyperclip.copy(res)