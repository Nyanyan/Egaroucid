import pyperclip
import sys
import re

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

def parse_input():
    """
    一括入力を受け取り、パースする
    Enter 3回で入力終了
    """
    print('Input all data (press Enter 3 times to finish):')
    lines = []
    empty_count = 0
    
    while True:
        line = input()
        if line.strip() == '':
            empty_count += 1
            if empty_count >= 3:
                break
        else:
            empty_count = 0
            lines.append(line)
    
    return lines

def extract_level_data(lines):
    """
    入力から各レベルのデータを抽出
    """
    levels = []
    current_level = None
    current_section = None  # 'win_rate' or 'disc'
    data = {}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # レベル情報の行を検出
        level_match = re.search(r'at level (\d+)', line, re.IGNORECASE)
        if level_match:
            current_level = level_match.group(1)
            if current_level not in data:
                data[current_level] = {'names': [], 'win_rates': [], 'discs': []}
                levels.append(current_level)
            i += 1
            continue
        
        # Win Rate セクション
        if 'Win Rate' in line:
            current_section = 'win_rate'
            i += 1
            # 次の行はヘッダー（vs > ...）なのでスキップ
            if i < len(lines):
                i += 1
            continue
        
        # Average Disc Difference セクション
        if 'Average Disc' in line or 'Disc Difference' in line:
            current_section = 'disc'
            i += 1
            # 次の行はヘッダー（vs > ...）なのでスキップ
            if i < len(lines):
                i += 1
            continue
        
        # データ行を処理
        if current_level and current_section:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                value = parts[-1]  # 最後の列（all列の値）
                
                if current_section == 'win_rate':
                    # 名前が初めて出現したら追加
                    if name not in data[current_level]['names']:
                        data[current_level]['names'].append(name)
                    data[current_level]['win_rates'].append(value)
                elif current_section == 'disc':
                    data[current_level]['discs'].append(value)
        
        i += 1
    
    return levels, data

# 入力を受け取る
input_lines = parse_input()

# データを解析
levels, parsed_data = extract_level_data(input_lines)

# テーブル形式に変換
table = []
for level in levels:
    table.append([
        parsed_data[level]['names'],
        parsed_data[level]['win_rates'],
        parsed_data[level]['discs']
    ])

res = '<div class="table_wrapper"><table>\n'

# name
if table:
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

print('\nGenerated HTML copied to clipboard!')
pyperclip.copy(res)