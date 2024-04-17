import datetime

#version_dot = input('version (X.Y.Z): ')
version_dot = '7.0.0'
date_str = '2024/04/17'


version_underbar = version_dot.replace('.', '_')

release_table_arr = [
'''
<div class="table_wrapper">
<table>
<tr>
    <th>OS</th>
    <th>CPU</th>
    <th>追加要件</th>
    <th>リリース日</th>
    <th>インストール版</th>
    <th>Zip版</th>
</tr>
<tr>
    <td>Windows</td>
    <td>x64</td>
    <td>AVX2(標準)</td>
    <td>DATE</td>
    <td>[Egaroucid VERSION_DOT SIMD インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_SIMD_installer.exe)</td>
    <td>[Egaroucid VERSION_DOT SIMD Zip](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Windows_x64_SIMD_Portable.zip)</td>
</tr>
<tr>
    <td>Windows</td>
    <td>x64</td>
    <td>-</td>
    <td>DATE</td>
    <td>[Egaroucid VERSION_DOT Generic インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Generic_installer.exe)</td>
    <td>[Egaroucid VERSION_DOT Generic Zip](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Windows_x64_Generic_Portable.zip)</td>
</tr>
</table>
</div>
''',
'''
<div class="table_wrapper"><table>
<tr>
    <th>OS</th>
    <th>CPU</th>
    <th>Requirements</th>
    <th>Date</th>
    <th>Installer</th>
    <th>Zip</th>
</tr>
<tr>
    <td>Windows</td>
    <td>x64</td>
    <td>AVX2 (Standard)</td>
    <td>DATE</td>
    <td>[Egaroucid VERSION_DOT SIMD Installer](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_SIMD_installer.exe)</td>
    <td>[Egaroucid VERSION_DOT SIMD Zip](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Windows_x64_SIMD_Portable.zip)</td>
</tr>
<tr>
    <td>Windows</td>
    <td>x64</td>
    <td>-</td>
    <td>DATE</td>
    <td>[Egaroucid VERSION_DOT Generic Installer](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Generic_installer.exe)</td>
    <td>[Egaroucid VERSION_DOT Generic Zip](https://github.com/Nyanyan/Egaroucid/releases/download/vVERSION_DOT/Egaroucid_VERSION_UNDERBAR_Windows_x64_Generic_Portable.zip)</td>
</tr>
</table>
</div>
'''
]

#dt_now = datetime.datetime.now()
#date_str = dt_now.strftime('%Y/%m/%d')
print(date_str, version_dot, version_underbar)

lang_arr = ['ja', 'en']
in_file_name = 'download/index_writing.md'
out_file_name = 'download/index.md'
identifier = 'DOWNLOAD_TABLE_HERE'

for lang, release_table in zip(lang_arr, release_table_arr):
    in_file = lang + '/' + in_file_name
    out_file = lang + '/' + out_file_name
    release_table = release_table.replace('DATE', date_str).replace('VERSION_DOT', version_dot).replace('VERSION_UNDERBAR', version_underbar)
    s = ''
    with open(in_file, 'r', encoding='utf-8') as f:
        while True:
            ss = f.readline()
            if not ss:
                break
            if identifier in ss:
                print(lang, 'replace release table')
                ss = ss.replace(identifier, release_table)
            s += ss
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(s)





