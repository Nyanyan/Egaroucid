# Egaroucid ダウンロード

インストール版とZip版があります。



## ダウンロード

以下から自分の環境に合ったものをダウンロードしてください。



EgaroucidはSIMDバージョン(AVX2が必要)に最適化して作っていますが、こちらは概ね2013年以降のCPUでないと動作しません。その場合にはGenericバージョンを使用してください。



インストール版はインストールが必要です。



Zip版はZipファイルを解凍し、中の```Egaroucid_[バージョン情報].exe```を実行してください。



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
        <td>2023/07/09</td>
        <td>[Egaroucid 6.3.0 SIMD インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/v6.3.0/Egaroucid_6_3_0_SIMD_installer.exe)</td>
        <td>[Egaroucid 6.3.0 SIMD Zip](https://github.com/Nyanyan/Egaroucid/releases/download/v6.3.0/Egaroucid_6_3_0_Windows_x64_SIMD_Portable.zip)</td>
    </tr>
    <tr>
        <td>Windows</td>
        <td>x64</td>
        <td>-</td>
        <td>2023/07/09</td>
        <td>[Egaroucid 6.3.0 Generic インストーラ](https://github.com/Nyanyan/Egaroucid/releases/download/v6.3.0/Egaroucid_6_3_0_Generic_installer.exe)</td>
        <td>[Egaroucid 6.3.0 Generic Zip](https://github.com/Nyanyan/Egaroucid/releases/download/v6.3.0/Egaroucid_6_3_0_Windows_x64_Generic_Portable.zip)</td>
    </tr>
</table>



過去のバージョンや各バージョンのリリースノートは[GitHubのリリース](https://github.com/Nyanyan/Egaroucid/releases)からご覧ください。



## インストール

インストール版の場合、ダウンロードしたインストーラを実行してください。管理者権限が必要です。



「WindowsによってPCが保護されました」と出た場合は、「詳細情報」をクリックすると実行することができます。ただし、この操作は自己責任で行ってください。

<div class="centering_box">
    <img class="pic2" src="img/cant_run1.png">
    <img class="pic2" src="img/cant_run2.png">
</div>



## 実行

インストールまたは解凍した<code>Egaroucid_[バージョン情報].exe</code>を実行するとEgaroucidが起動します。

<div class="centering_box">
    <img class="pic2" src="img/egaroucid.png">
</div>


## book形式の変更について

Egaroucidはバージョン6.3.0からbook形式を変更しました。新しいbookの拡張子は```.egbk2```です。6.2.0以前のバージョンをお使いで新しく6.3.0をインストールした場合、初回起動時に古い```.egbk```形式のbookを自動で変換します。また、古い形式もbookの参照および統合機能が使えます。ただし、保存形式は新しい```.egbk2```形式になります。
