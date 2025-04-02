import pyperclip
import sys

levels = ['1', '5']

if sys.argv[1] == 'ja':
    NAME = '名称'
    WIN_RATE = '勝率'
    DISCS_EARNED = '平均獲得石数'
    LEVEL = 'レベル'
else:
    NAME = 'Name'
    WIN_RATE = 'Winning Rate'
    DISCS_EARNED = 'Avg. Discs Earned'
    LEVEL = 'Level '

table = [[[], [], []], [[], [], []]]

for idx, level in enumerate(levels):
    print('level', level, 'win rate')
    while True:
        try:
            data = input().split()
            name = data[0]
            winning_rate = data[-1]
            table[idx][0].append(name)
            table[idx][1].append(winning_rate)
        except:
            break
    print('level', level, 'discs')
    while True:
        try:
            data = input().split()
            #name = data[0]
            avg_discs = data[-1]
            #table[0].append(name)
            table[idx][2].append(avg_discs)
        except:
            break


res = '<div class="table_wrapper"><table>\n'

# name
res += '<tr><th>' + NAME + '</th>'
for elem in table[0][0]:
    res += '<td>' + elem + '</td>'
res += '</tr>'

for idx, level in enumerate(levels):
    res += '<tr><th>' + LEVEL + level + ' ' + WIN_RATE + '</th>'
    for elem in table[idx][1]:
        res += '<td>' + elem + '</td>'
    res += '</tr>'

    res += '<tr><th>' + LEVEL + level + ' ' + DISCS_EARNED + '</th>'
    for elem in table[idx][2]:
        res += '<td>' + elem + '</td>'
    res += '</tr>'

res += '\n</table></div>'

pyperclip.copy(res)