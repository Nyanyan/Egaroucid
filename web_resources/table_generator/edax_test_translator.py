import pyperclip
import sys

if sys.argv[1] == 'ja':
    BLACK = '黒'
    WHITE = '白'
    head = '''<div class="table_wrapper"><table>
<tr><th>レベル</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th><th>Egaroucid勝率</th></tr>
'''
else:
    BLACK = 'Black'
    WHITE = 'White'
    head = '''<div class="table_wrapper"><table>
<tr><th>Level</th><th>Egaroucid win</th><th>Draw</th><th>Edax Win</th><th>Egaroucid Win Ratio</th></tr>
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
# 6.1.0から
idxes = [1, 6, 12] # level, black, white

res = head
while True:
    data = input().split()
    try:
        use_data = []
        for idx in idxes:
            use_data.append(data[idx])
        res += '<tr>'
        res += '<td>' + use_data[0] + '</td>' # level
        wdl_black = [int(elem) for elem in use_data[1].split('-')] # Egaroucid black
        wdl_white = [int(elem) for elem in use_data[2].split('-')] # Egaroucid white
        for i in range(3):
            res += '<td>' + str(wdl_black[i] + wdl_white[i]) + '(' + BLACK + ': ' + str(wdl_black[i]) + ' ' + WHITE + ': ' + str(wdl_white[i]) + ')' + '</td>'
        win_rate = (wdl_black[0] + wdl_white[0] + wdl_black[1] * 0.5 + wdl_white[1] * 0.5) / (sum(wdl_black) + sum(wdl_white))
        res += '<td>' + str(round(win_rate, 3)) + '</td>'
        res += '</tr>\n'
    except:
        break

res += '</table></div>'

pyperclip.copy(res)