import pyperclip
import sys

if sys.argv[1] == 'ja':
    head = '''<table>
<tr>
<td>レベル</td>
<td>Egaroucid勝ち</td>
<td>引分</td>
<td>Edax勝ち</td>
<td>Egaroucid勝率</td>
</tr>
'''
else:
    head = '''<table>
<tr>
<td>Level</td>
<td>Egaroucid win</td>
<td>Draw</td>
<td>Edax Win</td>
<td>Egaroucid Win Ratio</td>
</tr>
'''

'''
# 5.7.0まで
idxes = [0, 8]
#idxes = [0, 13]
'''
'''
# 5.8.0
idxes = [1, 9]
idxes = [1, 14]
'''
'''
# 5.9.0-6.0.0
#idxes = [1, 9]
idxes = [1, 15]
'''
# 6.1.0
#idxes = [1, 6]
idxes = [1, 12]

res = head
while True:
    data = input().split()
    try:
        use_data = []
        for idx in idxes:
            use_data.append(data[idx])
        res += '<tr>\n'
        res += '<td>' + use_data[0] + '</td>\n'
        wdl = use_data[1].split('-')
        for i in range(3):
            res += '<td>' + wdl[i] + '</td>\n'
        win_rate = int(wdl[0]) / (int(wdl[0]) + int(wdl[2]))
        res += '<td>' + str(round(win_rate, 2)) + '</td>\n'
        res += '</tr>\n'
    except:
        break

res += '</table>'

pyperclip.copy(res)