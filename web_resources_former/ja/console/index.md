# Egaroucid for Console

ダウンロードまたは手元でのコンパイルが必要です。



## ダウンロード

以下から自分の環境に合ったものをダウンロードしてください。

<table>
    <tr>
        <td>OS</td>
        <td>環境</td>
        <td>リリース日</td>
        <td>ダウンロード</td>
    </tr>
    <tr>
        <td>Windows</td>
        <td>AVX2(2013年以降のCPU)</td>
        <td>2022/10/10</td>
        <td>[Egaroucid 6.0.0](https://github.com/Nyanyan/Egaroucid/releases/download/v6.0.0/Egaroucid_6_0_0_setup_Windows.exe)</td>
    </tr>
</table>


## 手元でのコンパイル

**MacOSでの動作確認はまだできていません。**

<ul>
    <li><code>g++</code>コマンドが必要
        <ul>
            <li>Windowsではバージョン12.2.0で動作確認済</li>
            <li>Ubuntuではバージョン11.3.0で動作確認済</li>
        </ul>
    </li>
    <li>C++17の機能が必要</li>
</ul>

コードを入手します。



<code>$ git clone git@github.com:Nyanyan/Egaroucid.git</code>



ディレクトリを移動します。



<code>$ cd Egaroucid/src</code>



<code>g++</code>コマンドにてコンパイルします。出力ファイルは任意の名前で構いません。



<code>$ g++ -O2 Egaroucid_console.cpp -o Egaroucid_for_console.exe -mtune=native -march=native -mfpmath=both -pthread -std=c++17 -Wall -Wextra
</code>



実行します。



<code>$ Egaroucid_for_console.exe</code>



## 使い方

<code>$ Egaroucid_for_console.exe -help</code>



を実行すると使えるオプションやコマンドが確認できます。





また、Egaroucid for ConsoleはGo Text Protocol (GTP)に対応しています。GTPコマンドを使う場合には



<code>$ Egaroucid_for_console.exe -gtp</code>



を実行してください。



WindowsにてGoGUIを用いた動作確認を、UbuntuにてQuarryを用いた動作確認を行いました。



## フォルダ構成

Egaroucid for Consoleはいくつかの外部ファイルを必要とします。上記の方法でダウンロードやコンパイルした場合には特にエラーなく動きますが、もし動かなくなった場合にはフォルダ構成を確認してください。

<ul>
    <li>Egaroucid_for_console.exe</li>
    <li>resources
        <ul>
            <li>hash (ハッシュファイル なくても動きます)
                <ul>
                    <li>hash23.eghs</li>
                    <li>hash24.eghs</li>
                    <li>hash25.eghs</li>
                    <li>hash26.eghs</li>
                    <li>hash27.eghs</li>
                </ul>
            </li>
            <li>book.egbk (bookファイル)</li>
            <li>eval.egev (評価ファイル)</li>
        </ul>
    </li>
</ul>

