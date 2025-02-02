# Egaroucid 学習データ



## 概要

オセロAI Egaroucidによって生成した、オセロAIの学習用データです。

大量のデータを収録しましたので、ご自身のオセロAIの制作などにご利用ください。

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


## Egaroucid 7.4.0 lv.17 & 7.5.1 lv.17 によるデータ

[Egaroucid_Train_Data.zip](https://github.com/Nyanyan/Egaroucid/releases/download/training_data/Egaroucid_Train_Data.zip)をダウンロードし、解凍してください。

各フォルダ内に```XXXXXXX.txt```というテキストファイルが入っています。これを開くと各行について、オセロの盤面を表す文字列とその盤面の(手番側の)スコアがスペース区切りで記録されています。1つのテキストファイルには100万局面ずつ収録してあります。

テキストファイルは各100万行ずつあり、各行に以下のようなデータが入っています。

<code>-XO-OOXOOXX-OXOO-XXOXXOOX-OXOOXOOXOOOXXXO-XOOOXXO-O-OO---OOOX-O- 4</code>

前半64文字で盤面を表しています。各文字はa1、b1、c1、…、a2、b2、c2、…、h8の順番で並んでいます。<code>X</code>がその盤面での手番側(これから着手する方)の石、<code>O</code>ｆが相手の石、<code>-</code>が空きマスを表します。

盤面を表す文字列から1つスペースを空けて、数字が記録されています。これはその盤面での手番側の評価値(予想最終石差)を表します。

盤上の合計の石数と収録されている局面の数の対応は以下の通りです。

<div class="table_wrapper"><table>
<tr>
	<th>盤上の石数</th>
	<th>収録局面数</th>
</tr>
<tr>
	<td>4</td>
	<td>1</td>
</tr>
<tr>
	<td>5</td>
	<td>1</td>
</tr>
<tr>
	<td>6</td>
	<td>3</td>
</tr>
<tr>
	<td>7</td>
	<td>14</td>
</tr>
<tr>
	<td>8</td>
	<td>60</td>
</tr>
<tr>
	<td>9</td>
	<td>322</td>
</tr>
<tr>
	<td>10</td>
	<td>1773</td>
</tr>
<tr>
	<td>11</td>
	<td>10649</td>
</tr>
<tr>
	<td>12</td>
	<td>67245</td>
</tr>
<tr>
	<td>13</td>
	<td>434029</td>
</tr>
<tr>
	<td>14から63</td>
	<td>各500000</td>
</tr>
<tr>
	<td>合計</td>
	<td>63249078</td>
</tr>
    </table></div>


序盤11手まで(盤上の合計の石数が15枚以下)のデータはEgaroucid for Console 7.4.0 レベル17において生成しました。11手までの進行をすべて列挙して、その進行すべてについてEgaroucidを使って評価値を計算し、その結果をnegamaxすることで生成しました。

序盤12手以降(盤上の合計の石数が16枚以上)のデータはEgaroucid for Console 7.5.1 レベル17での自己対戦によって生成しました。各局面に紐づけられたスコアは、自己対戦の終局時のスコアです。自己対戦時、序盤の$7 \leq N \leq 59$手をランダム打ちさせることで、対戦結果をばらつかせました。$N$手より前の局面(ランダム打ちによって悪手の応酬になっており、最終スコアと局面のスコアがかけ離れている)は収録していません。公開している局面は、これらの棋譜から、序盤ランダム打ちした直後の局面を優先して収録しました。

2025/02/02 公開



## Egaroucid 6.3.0 レベル11による自己対戦の棋譜

[Egaroucid_Transcript.zip](https://github.com/Nyanyan/Egaroucid/releases/download/transcript/Egaroucid_Transcript.zip)をダウンロードし、解凍してください。

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
2023/07/17 公開
