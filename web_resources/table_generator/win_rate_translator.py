import pyperclip
import sys



if sys.argv[1] == 'ja':
    NAME = '名称'
    WIN_RATE = '勝率'
    DISCS_EARNED = '平均獲得石数'
else:
    NAME = 'Name'
    WIN_RATE = 'Winning Rate'
    DISCS_EARNED = 'Avg. Discs Earned'

table = [[], [], []]
while True:
    try:
        data = input().split()
        name = data[0]
        winning_rate = data[-1]
        table[0].append(name)
        table[1].append(winning_rate)
    except:
        break

while True:
    try:
        data = input().split()
        #name = data[0]
        avg_discs = data[-1]
        #table[0].append(name)
        table[2].append(avg_discs)
    except:
        break


res = '<div class="table_wrapper"><table>\n'

res += '<tr><th>' + NAME + '</th>'
for elem in table[0]:
    res += '<td>' + elem + '</td>'

res += '<tr><th>' + WIN_RATE + '</th>'
for elem in table[1]:
    res += '<td>' + elem + '</td>'

res += '<tr><th>' + DISCS_EARNED + '</th>'
for elem in table[2]:
    res += '<td>' + elem + '</td>'

res += '\n</table></div>'

pyperclip.copy(res)