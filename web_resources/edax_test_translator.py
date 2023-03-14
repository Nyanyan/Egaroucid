import pyperclip
import sys

if sys.argv[1] == 'ja':
    head = '''<table>
<tr>
<th>レベル</th>
<th>Egaroucid勝ち</th>
<th>引分</th>
<th>Edax勝ち</th>
<th>Egaroucid勝率</th>
</tr>
'''
else:
    head = '''<table>
<tr>
<th>Level</th>
<th>Egaroucid win</th>
<th>Draw</th>
<th>Edax Win</th>
<th>Egaroucid Win Ratio</th>
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
idxes = [1, 6]
#idxes = [1, 12]

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