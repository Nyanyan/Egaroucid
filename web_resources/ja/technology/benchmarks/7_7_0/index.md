# Egaroucid 7.7.0 ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。また、比較としてオープンソースで最速クラスのオセロAI [Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)の結果も掲載します (Edaxはバージョン4.6が最新ですが、手元で実行したところ4.6よりも4.5.5の方が速かったため、4.5.5を採用しました)。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>20.746</td><td>14789131082</td><td>712866628</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>31.906</td><td>15735727776</td><td>493190239</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>22.798</td><td>27703082424</td><td>1215154067</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt">010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>27.468</td><td>27905804830</td><td>1015938723</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
</table>
</div>


### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidおよび[Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>AVX512</td><td>37.615</td><td>14224643683</td><td>378164128</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>39.356</td><td>14201687484</td><td>360851902</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>70.868</td><td>14512377695</td><td>204780404</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v4</td><td>35.406</td><td>26154694442</td><td>738707972</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt">110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>38.344</td><td>25950762077</td><td>676788078</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt">111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>48.314</td><td>25908072899</td><td>536243592</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
</table>
</div>



## Edax 4.6との対戦

世界最強クラスのオープンソースオセロAI、[Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにし、2戦で獲得した石数が多い方が勝ちとしました。

勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。全ての条件でEgaroucidが勝ち越しています。

また、平均獲得石数は平均してEgaroucidがEdaxよりも何枚多く石を獲得できたかを表します。この値が大きいほど、Edaxに対して大勝しているということになります。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。bookは双方未使用です。

<div class="table_wrapper"><table>
<tr><th>レベル</th><th>平均獲得石数</th><th>勝率</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th></tr>
<tr><td>1</td><td>+10.92</td><td>0.726</td><td>720</td><td>12</td><td>268</td></tr>
<tr><td>5</td><td>+8.98</td><td>0.794</td><td>784</td><td>21</td><td>195</td></tr>
<tr><td>10</td><td>+2.16</td><td>0.671</td><td>637</td><td>69</td><td>294</td></tr>
<tr><td>15</td><td>+0.89</td><td>0.616</td><td>138</td><td>32</td><td>80</td></tr>
<tr><td>21</td><td>+0.24</td><td>0.565</td><td>46</td><td>21</td><td>33</td></tr>
</table></div>



