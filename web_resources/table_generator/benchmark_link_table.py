import pyperclip
import sys


data = [ # version, dir, release date
    ['7.4.0', '7_4_0', 'TBD'],
    ['7.3.0', '7_3_0', '2024/08/16'],
    ['7.2.0', '7_2_0', '2024/06/25'],
    ['7.1.0', '7_1_0', '2024/06/06'],
    ['7.0.0', '7_0_0', '2024/04/17'],
    ['6.5.0', '6_5_0', '2023/10/25'],
    ['6.4.0', '6_4_0', '2023/09/01'],
    ['6.3.0', '6_3_0', '2023/07/09'],
    ['6.2.0', '6_2_0', '2023/03/15'],
    ['6.1.0', '6_1_0', '2022/12/23'],
    ['6.0.0', '6_0_0', '2022/10/10'],
    ['5.10.0', '5_10_0', '2022/06/08'],
    ['5.9.0', '5_9_0', '2022/06/07'],
    ['5.8.0', '5_8_0', '2022/05/13'],
    ['5.7.0', '5_7_0', '2022/03/26'],
    ['5.5.0/5.6.0', '5_5_0', '2022/03/16'],
    ['5.4.1', '5_4_1', '2022/03/02']
]


if sys.argv[1] == 'ja':
    VERSION = 'バージョン'
    RELEASE_DATE = 'リリース日'
else:
    VERSION = 'Version'
    RELEASE_DATE = 'Date'

table = [[], []]
for version, dr, date in data:
        table[0].append('<a href="./benchmarks/' + dr + '/">' + version + '</a>')
        table[1].append(date)

res = '<div class="table_wrapper"><table>\n'

res += '<tr><th>' + VERSION + '</th>'
for elem in table[0]:
    res += '<td>' + elem + '</td>'

res += '<tr><th>' + RELEASE_DATE + '</th>'
for elem in table[1]:
    res += '<td>' + elem + '</td>'

res += '\n</table></div>'

pyperclip.copy(res)