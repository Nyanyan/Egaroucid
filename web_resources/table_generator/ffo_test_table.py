import pyperclip
import sys
import glob


if sys.argv[1] == 'ja':
    head = '''<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
'''
else:
    head = '''<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
'''

# CPU, version, time, nodes, NPS, file
results = []

#7.3.0~
#summary_file = input('summary file: ')
summary_file = sys.argv[2]
with open(summary_file, 'r') as f:
    summary = f.read().splitlines()
for line in summary:
    line_elem = line.split(',')
    engine = line_elem[0]
    #cpu = line_elem[1].replace('_', ' ')
    edition = line_elem[2]
    time = line_elem[3]
    nodes = line_elem[4]
    nps = line_elem[5]
    file = line_elem[6]
    print(engine, edition, time, nodes, nps, file)
    results.append([engine, edition, time, nodes, nps, file])

res = head
for result in results:
    res += '<tr>\n'
    for elem in result[:-1]:
        res += '<td>' + elem + '</td>'
    filename = result[-1].replace('\\', '/').split('/')[-1]
    res += '<td><a href="' + './files/' + filename + '">' + filename + '</a></td>'
    res += '\n'
    res += '</tr>\n'

res += '''</table>
</div>'''

#print(res)

pyperclip.copy(res)