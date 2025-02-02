# Egaroucid 技術資料



## 技術解説

日本語のみでEgaroucidの[技術解説](./explanation)を書きました。のんびりと追記します。



## ベンチマーク

Egaroucidの性能の確認として2種類のベンチマークを使用しています。1つ目は[The FFO endgame test suite](http://www.radagast.se/othello/ffotest.html)です。これは、終盤の完全読みにかかる時間に関するベンチマークです。2つ目は対戦です。Egaroucidの過去バージョンの他、他の強豪オセロAIとの対戦として、[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)とも対戦しました。単純に評価関数の強さを計測するため、bookを使わず、[XOT](https://berg.earthlingz.de/xot/aboutxot.php?lang=en)という初期局面集を用いて対戦させました。

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

以下はThe FFO endgame test suiteの40から59番をCore i9 13900Kで実行した結果を、バージョンを横軸に取ってグラフにしたものです。SIMD版を使用しました。

<div class="centering_box">
	<img class="pic2" src="img/ffo_time.png">
    <img class="pic2" src="img/ffo_node.png">
    <img class="pic2" src="img/ffo_nps.png">
</div>
バージョン7.4.0は7.3.0よりも計算時間が多くかかっていますが、これはIntelマイクロコード0x129による性能低下に起因するものです。7.3.0のベンチマーク測定時には0x129適用前でしたが、7.4.0測定時には0x129適用済みでした。

### XOTによる対戦

オセロAIの強さを評価するためには、対戦するのが一番でしょう。ここでは、各バージョンに[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)を加え、総当たり戦をした結果を掲載します。

対戦はレベル1(中盤1手読み、終盤2手完全読み)で行いました。

対戦にはそれぞれ[XOT](https://berg.earthlingz.de/xot/aboutxot.php?lang=en)の進行を初期盤面として使い、各進行では先手後手それぞれ1回ずつ対戦させています。

<div class="table_wrapper"><table>
<tr><th>名称</th><td>7.5.0</td><td>7.4.0</td><td>7.3.0</td><td>7.2.0</td><td>7.1.0</td><td>7.0.0</td><td>6.5.X</td><td>6.4.X</td><td>6.3.X</td><td>6.1.X</td><td>6.0.X</td><td>Edax</td></tr><tr><th>勝率</th><td>0.5790</td><td>0.5716</td><td>0.5405</td><td>0.5359</td><td>0.5385</td><td>0.5339</td><td>0.5002</td><td>0.4724</td><td>0.4403</td><td>0.4781</td><td>0.4135</td><td>0.3963</td></tr><tr><th>平均獲得石数</th><td>+4.92</td><td>+3.98</td><td>+2.75</td><td>+2.62</td><td>+2.69</td><td>+2.29</td><td>+0.49</td><td>-1.72</td><td>-4.03</td><td>-1.44</td><td>-5.71</td><td>-6.84</td></tr>
</table></div>
対戦の詳細は[こちら](./battle.txt)をご覧ください。

Egaroucid 6.2.0はEgaroucid 6.3.0と同一の評価関数のため、省いています。



### バージョンごとの詳細

各バージョンのベンチマークを公開します。上で載せなかった古いバージョンのベンチマークもあります。

こちらの詳細はバージョンごとに少し条件が違うものもありますので、詳細はそれぞれのページをご覧ください。

<div class="table_wrapper"><table>
<tr><th>バージョン</th><td><a href="./benchmarks/7_5_0/">7.5.0</a></td><td><a href="./benchmarks/7_4_0/">7.4.0</a></td><td><a href="./benchmarks/7_3_0/">7.3.0</a></td><td><a href="./benchmarks/7_2_0/">7.2.0</a></td><td><a href="./benchmarks/7_1_0/">7.1.0</a></td><td><a href="./benchmarks/7_0_0/">7.0.0</a></td><td><a href="./benchmarks/6_5_0/">6.5.0</a></td><td><a href="./benchmarks/6_4_0/">6.4.0</a></td><td><a href="./benchmarks/6_3_0/">6.3.0</a></td><td><a href="./benchmarks/6_2_0/">6.2.0</a></td><td><a href="./benchmarks/6_1_0/">6.1.0</a></td><td><a href="./benchmarks/6_0_0/">6.0.0</a></td><td><a href="./benchmarks/5_10_0/">5.10.0</a></td><td><a href="./benchmarks/5_9_0/">5.9.0</a></td><td><a href="./benchmarks/5_8_0/">5.8.0</a></td><td><a href="./benchmarks/5_7_0/">5.7.0</a></td><td><a href="./benchmarks/5_5_0/">5.5.0/5.6.0</a></td><td><a href="./benchmarks/5_4_1/">5.4.1</a></td></tr><tr><th>リリース日</th><td>2024/12/24</td><td>2024/10/03</td><td>2024/08/16</td><td>2024/06/25</td><td>2024/06/06</td><td>2024/04/17</td><td>2023/10/25</td><td>2023/09/01</td><td>2023/07/09</td><td>2023/03/15</td><td>2022/12/23</td><td>2022/10/10</td><td>2022/06/08</td><td>2022/06/07</td><td>2022/05/13</td><td>2022/03/26</td><td>2022/03/16</td><td>2022/03/02</td></tr>
</table></div>


## 学習データ公開

Egaroucidで生成した、オセロAI向け学習データを大量に公開しています。詳しくは[学習データ](./train-data)をご覧ください。

