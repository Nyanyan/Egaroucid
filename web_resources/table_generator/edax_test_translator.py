import pyperclip
import sys

if sys.argv[1] == 'ja':
    head = '''<div class="table_wrapper"><table>
<tr><th>レベル</th><th>平均獲得石数</th><th>勝率</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th></tr>
'''
else:
    head = '''<div class="table_wrapper"><table>
<tr><th>Level</th><th>Avg. Discs Earned</th><th>Winning Rate</th><th>Egaroucid Win</th><th>Draw</th><th>Edax Win</th></tr>
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
'''
# 6.1.0から
idxes = [1, 6, 12] # level, black, white
'''
'''
# 7.5.0から
idxes = [1, 6, 12, 22] # level, black, white, discs
'''

# 7.6.0から
idxes = [1, 4, 13] # level, result, discs


res = head
while True:
    data = input().split()
    try:
        use_data = []
        for idx in idxes:
            use_data.append(data[idx])
        res += '<tr>'
        res += '<td>' + use_data[0] + '</td>' # level
        discs = str(round(float(use_data[2]), 2))
        sgn = '+' if float(discs) >= 0 else ''
        res += '<td>' + sgn + discs + '</td>' # discs
        wdl = [int(elem) for elem in use_data[1].split('-')] # Egaroucid result
        win_rate = (wdl[0] + wdl[1] * 0.5) / sum(wdl)
        res += '<td>' + str(round(win_rate, 3)) + '</td>' # win rate
        for i in range(3):
            res += '<td>' + str(wdl[i]) + '</td>' # wdl
        res += '</tr>\n'
    except:
        break

res += '</table></div>'

pyperclip.copy(res)