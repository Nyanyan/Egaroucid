# Egaroucid 7.6.0 ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。また、比較としてオープンソースで最速クラスのオセロAI [Edax 4.5.3](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.3)の結果も掲載します (Edaxはバージョン4.6が最新ですが、手元で実行したところ4.6よりも4.5.3の方が速かったため、4.5.3を採用しました)。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>20.547</td><td>14035262270</td><td>683080852</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>30.346</td><td>14290308977</td><td>470912442</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>26.093</td><td>28087572364</td><td>1076440898</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>30.299</td><td>27886392112</td><td>920373349</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
</table>
</div>



### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidおよび[Edax 4.5.3](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.3)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>AVX512</td><td>34.272</td><td>13061469404</td><td>381111969</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>35.089</td><td>13275458500</td><td>378336757</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>62.678</td><td>13403862190</td><td>213852742</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_avx512</td><td>39.046</td><td>27252069199</td><td>697947785</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>40.718</td><td>26747187985</td><td>656888550</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>48.263</td><td>26602214755</td><td>551192731</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
</table>
</div>



## Edax 4.6との対戦

現状世界最強とも言われるオセロAI、[Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにし、2戦で獲得した石数が多い方が勝ちとしました。

勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。全ての条件でEgaroucidが勝ち越しています。

また、平均獲得石数は平均してEgaroucidがEdaxよりも何枚多く石を獲得できたかを表します。この値が大きいほど、Edaxに対して大勝しているということになります。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。bookは双方未使用です。

<div class="table_wrapper"><table>
<tr><th>レベル</th><th>平均獲得石数</th><th>勝率</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th></tr>
<tr><td>1</td><td>+11.27</td><td>0.742</td><td>734</td><td>15</td><td>251</td></tr>
<tr><td>5</td><td>+8.37</td><td>0.793</td><td>781</td><td>25</td><td>194</td></tr>
<tr><td>10</td><td>+1.72</td><td>0.633</td><td>590</td><td>87</td><td>323</td></tr>
<tr><td>15</td><td>+0.89</td><td>0.614</td><td>138</td><td>31</td><td>81</td></tr>
<tr><td>21</td><td>+0.45</td><td>0.565</td><td>50</td><td>13</td><td>37</td></tr>
</table></div>


