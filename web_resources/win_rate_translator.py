import pyperclip
import sys

if sys.argv[1] == 'ja':
    head = '''<div class="table_wrapper"><table>
<tr><th>名称</th><th>勝率</th></tr>
'''
else:
    head = '''<div class="table_wrapper"><table>
<tr><th>Name</th><th>Winning Rate</th></tr>
'''

res = head
while True:
    try:
        data = input().split()
        name = data[0]
        winning_rate = data[-1]
        res += '<tr>'
        res += '<td>' + name + '</td>'
        res += '<td>' + winning_rate + '</td>'
        res += '</tr>\n'
    except:
        break

res += '</table></div>'

pyperclip.copy(res)