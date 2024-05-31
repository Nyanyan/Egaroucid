import pyperclip
import sys

if sys.argv[1] == 'ja':
    head = '''<div class="table_wrapper"><table>
<tr>
<th>名称</th>
<th>勝率</th>
'''
else:
    head = '''<div class="table_wrapper"><table>
<tr>
<th>Name</th>
<th>Winning Rate</th>
'''

res = head
while True:
    try:
        data = input().split()
        name = data[0]
        winning_rate = data[-1]
        res += '<tr>\n'
        res += '<td>' + name + '</td>\n'
        res += '<td>' + winning_rate + '</td>\n'
        res += '</tr>\n'
    except:
        break

res += '</table></div>'

pyperclip.copy(res)