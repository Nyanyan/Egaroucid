# Egaroucid Technology



## Benchmarks

I used 2 benchmarks for evaluating Egaroucid. The first one is [The FFO endgame test suite](http://www.radagast.se/othello/ffotest.html). This test is for the speed of endgame complete search. The second one is the matches against old versions of Egaroucid and [Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4). To test the strength of its evaluation function, I used no book, and used [XOT](https://berg.earthlingz.de/xot/aboutxot.php?lang=en) for the starting positions.

### The FFO endgame test suite

終盤探索は、以下3つの指標で評価しています。

<ul>
    <li>計算時間</li>
    <li>訪問ノード数</li>
    <li>NPS (1秒あたりのノード訪問回数)</li>
</ul>


ユーザにとって一番重要なのは計算時間です。決まったテストケースを処理するのにかかる時間を秒数で表します。ここでは[The FFO endgame test suite](http://www.radagast.se/othello/ffotest.html)の40から59番のテストケース(20から34手完全読み)にかかる時間を使いました。これは減ると嬉しい値です。

計算時間を短くするには、まず(厳密に)無駄な探索を減らせば良いです。無駄な探索が多いと訪問ノード数(探索した盤面の数)が増えます。これも減ると嬉しい値です。

計算時間を短くするためのもう一つの観点は、1秒あたりのノード訪問回数を上げることです。これはNodes Per Secondの頭文字を取ってNPSと言われます。これは上がると嬉しい値です。

There are some graphs of results of The FFO endgame test suite on Core i9 13900K.

<div class="centering_box">
	<img class="pic2" src="img/ffo_time.png">
    <img class="pic2" src="img/ffo_node.png">
    <img class="pic2" src="img/ffo_nps.png">
</div>



### XOTによる対戦

オセロAIの強さを評価するためには、対戦するのが一番でしょう。ここでは、各バージョンに[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)を加え、総当たり戦をした結果を掲載します。

対戦はレベル1(中盤1手読み、終盤2手完全読み)で行いました。

対戦にはそれぞれXOTの進行を初期盤面として使い、各進行では先手後手それぞれ1回ずつ対戦させています。

<table>
<tr>
<th>Name</th>
<th>Winning Rate</th>
<tr>
<td>Edax</td>
<td>0.3969</td>
</tr>
<tr>
<td>6.1.0</td>
<td>0.5713</td>
</tr>
<tr>
<td>6.2.0</td>
<td>0.4750</td>
</tr>
<tr>
<td>6.3.0</td>
<td>0.4750</td>
</tr>
<tr>
<td>6.4.0</td>
<td>0.5819</td>
</tr>
</table>

対戦の詳細は[こちら](./battle.txt)

### バージョンごとの詳細

各バージョンのベンチマークを公開します。上で載せなかった古いバージョンのベンチマークもあります。

こちらの詳細はバージョンごとに少し条件が違うものもありますので、詳細はそれぞれのページをご覧ください。

<table>
	<tr>
		<th>Version</th>
		<th>Date</th>
	</tr>
    <tr>
		<td><a href="./benchmarks/6_4_0/">6.4.0</a></td>
		<td>2023/09/01</td>
	</tr>
    <tr>
		<td><a href="./benchmarks/6_3_0/">6.3.0</a></td>
		<td>2023/07/09</td>
	</tr>
    <tr>
		<td><a href="./benchmarks/6_2_0/">6.2.0</a></td>
		<td>2023/03/15</td>
	</tr>
    <tr>
		<td><a href="./benchmarks/6_1_0/">6.1.0</a></td>
		<td>2022/12/23</td>
	</tr>
	<tr>
		<td><a href="./benchmarks/6_0_0/">6.0.0</a></td>
		<td>2022/10/10</td>
	</tr>
    	<tr>
		<td><a href="./benchmarks/5_10_0/">5.10.0</a></td>
		<td>2022/06/08</td>
	</tr>
    	<tr>
		<td><a href="./benchmarks/5_9_0/">5.9.0</a></td>
		<td>2022/06/07</td>
	</tr>
    	<tr>
		<td><a href="./benchmarks/5_8_0/">5.8.0</a></td>
		<td>2022/05/13</td>
	</tr>
    	<tr>
		<td><a href="./benchmarks/5_7_0/">5.7.0</a></td>
		<td>2022/03/26</td>
	</tr>
    	<tr>
		<td><a href="./benchmarks/5_5_0/">5.5.0/5.6.0</a></td>
		<td>2022/03/16</td>
	</tr>
    <tr>
		<td><a href="./benchmarks/5_4_1/">5.4.1</a></td>
		<td>2022/03/02</td>
	</tr>
</table>



## Technology Explanation

I wrote [Technology Explanation](https://www.egaroucid.nyanyan.dev/ja/technology/explanation/) only in Japanese. Please translate by yourself.



## Download Transcript

Huge dataset of games played by Egaroucid is available. Please see [Download Transcript](./transcript) page.
