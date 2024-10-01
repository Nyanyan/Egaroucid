import pyperclip
import sys



if sys.argv[1] == 'ja':
    NAME = '名称'
    WIN_RATE = '勝率'
else:
    NAME = 'Name'
    WIN_RATE = 'Winning Rate'

table = [[], []]
while True:
    try:
        data = input().split()
        name = data[0]
        winning_rate = data[-1]
        table[0].append(name)
        table[1].append(winning_rate)
    except:
        break

res = '<div class="table_wrapper"><table>\n'

res += '<tr><th>' + NAME + '</th>'
for elem in table[0]:
    res += '<td>' + elem + '</td>'

res += '<tr><th>' + WIN_RATE + '</th>'
for elem in table[1]:
    res += '<td>' + elem + '</td>'

res += '\n</table></div>'

pyperclip.copy(res)