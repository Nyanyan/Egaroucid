# Egaroucid 自己対戦の棋譜



## ダウンロード

[Egaroucid_Transcript.zip](https://github.com/Nyanyan/Egaroucid/releases/download/transcript/Egaroucid_Transcript.zip)をダウンロードし、解凍してください。



## 概要

オセロAI Egaroucidによる自己対戦の棋譜です。

大量のデータ(200万局)を収録しましたので、ご自身のオセロAIの制作などにご利用ください。

Webサイト: [https://www.egaroucid.nyanyan.dev/ja/](https://www.egaroucid.nyanyan.dev/ja/)

GitHubレポジトリ: [https://github.com/Nyanyan/Egaroucid](https://github.com/Nyanyan/Egaroucid)

作者: 山名琢翔 ( [https://nyanyan.dev/ja/](https://nyanyan.dev/ja/) )



## 利用規約

<ul>
    <li>この対局データはオセロAIの評価関数生成など、自身の活動に自由に活用してください。
        <ul>
            <li>強制ではありませんが、もしこのデータが役に立った場合にはEgaroucid作者の山名にご連絡いただけるか、「Egaroucidのサイトで公開しているデータを使った」と明記して公開していただけると嬉しいです。</li>
        </ul>
    </li>
    <li>この対局データを使ったことによるいかなる損害にも責任を負いません。自己責任でご利用ください。</li>
    <li>この対局データの再配布を禁止します。
        <ul>
            <li>宣伝してくださる場合は、EgaroucidのWebサイトまたはGitHubをぜひ宣伝してください。</li>
        </ul>
    </li>
</ul>



## 詳細

各フォルダ内に```XXXXXXX.txt```というテキストファイルが入っています。これを開くと```f5d6```形式でオセロの棋譜が収録されています。テキストファイルには1万局ずつ収録してあります。

評価関数生成を念頭において生成したデータですので、対局結果にばらつきが生まれるよう、対局開始から適当な手数 $N$ をランダム打ちしています。この $N$ は以下の方法で決めます。

1. 定数 $N_{min},N_{max}$ を定めておく
2. 各対局において、 $N_{min}\leq N \leq N_{max}$ を満たすように $N$ をランダムに決める
3. 対局開始から $N$ 手をランダムに打ち、その後をAIに打たせる

フォルダごとの棋譜の詳細は以下にまとめました。

<div class="table_wrapper"><table>
<tr>
	<th>フォルダ名</th>
	<th>0000_egaroucid_6_3_0_lv11</th>
</tr>
<tr>
	<td>AI名称</td>
	<td>Egaroucid for Console 6.3.0</td>
</tr>
<tr>
	<td>レベル</td>
	<td>11</td>
</tr>
<tr>
	<td>収録対局数</td>
	<td>2,000,000</td>
</tr>
<tr>
	<td> $N_{min}$ </td>
	<td>10</td>
</tr>
<tr>
	<td> $N_{max}$ </td>
	<td>19</td>
</tr>
    </table></div>




## 変更履歴

<div class="table_wrapper"><table>
<tr>
	<th>日付</th>
	<th>内容</th>
</tr>
<tr>
	<td>2023/07/17</td>
	<td>公開</td>
</tr>
    </table></div>
