# Egaroucid ダウンロード

インストール版とZip版があります。



## ダウンロード

Egaroucidはx64のCPU(Intel製かAMD製)に対応しています。ARMのCPUを使っている場合は動きません。

EgaroucidはSIMDバージョン(AVX2が必要)に最適化して作っていますが、こちらは概ね2013年以降のCPUでないと動作しません。その場合にはGenericバージョンを使用してください。

以下から自分の環境に合ったものをダウンロードしてください。




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
    <td>2024/02/13</td>
    <td>[Egaroucid 6.5.2 SIMD インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/v6.5.2/Egaroucid_6_5_2_SIMD_installer.exe)</td>
    <td>[Egaroucid 6.5.2 SIMD Zip](https://github.com/Nyanyan/Egaroucid/releases/download/v6.5.2/Egaroucid_6_5_2_Windows_x64_SIMD_Portable.zip)</td>
</tr>
<tr>
    <td>Windows</td>
    <td>x64</td>
    <td>-</td>
    <td>2024/02/13</td>
    <td>[Egaroucid 6.5.2 Generic インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/v6.5.2/Egaroucid_6_5_2_Generic_installer.exe)</td>
    <td>[Egaroucid 6.5.2 Generic Zip](https://github.com/Nyanyan/Egaroucid/releases/download/v6.5.2/Egaroucid_6_5_2_Windows_x64_Generic_Portable.zip)</td>
</tr>
</table>
</div>




過去のバージョンや各バージョンのリリースノートは[GitHubのリリース](https://github.com/Nyanyan/Egaroucid/releases)からご覧ください。



## インストール

インストール版の場合、ダウンロードしたインストーラを実行してください。管理者権限が必要です。



「WindowsによってPCが保護されました」と出た場合は、「詳細情報」をクリックすると実行することができます。ただし、この操作は自己責任で行ってください。

<div class="centering_box">
    <img class="pic2" src="img/cant_run1.png" alt="「WindowsによってPCが保護されました」という画面">
    <img class="pic2" src="img/cant_run2.png" alt="「WindowsによってPCが保護されました」という画面において「詳細情報」を押して実行する">
</div>




## 実行

インストールまたは解凍した<code>Egaroucid_[バージョン情報].exe</code>を実行するとEgaroucidが起動します。

<div class="centering_box">
    <img class="pic2" src="img/egaroucid.png" alt="Egaroucid">
</div>


## book形式の変更について

### egbk3形式への変更

Egaroucidはバージョン6.5.1からbook形式を変更し、拡張子が```.egbk3```のものを使うようになりました。以前のバージョンをお使いで新しく6.5.1をインストールした場合、初回起動時に古い```.egbk2```形式および```.egbk```形式のbookを自動で変換します。また、古い形式もbookの読み込みなど各種機能が使えます。ただし、保存形式は新しい```.egbk3```形式になります。

### egbk2形式への変更

Egaroucidはバージョン6.3.0からbook形式を変更しました。新しいbookの拡張子は```.egbk2```です。6.2.0以前のバージョンをお使いで新しく6.3.0をインストールした場合、初回起動時に古い```.egbk```形式のbookを自動で変換します。また、古い形式もbookの参照および統合機能が使えます。ただし、保存形式は新しい```.egbk2```形式になります。
