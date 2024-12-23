# Egaroucid 7.5.0 ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。また、比較としてオープンソースで最速クラスのオセロAI [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果も掲載します。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>21.782</td><td>15046058762</td><td>690756531</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>31.655</td><td>14998990657</td><td>473826904</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>25.266</td><td>28427839936</td><td>1125142086</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>28.875</td><td>27614464872</td><td>956345104</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
</table>
</div>


### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidおよび[Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Egaroucid</td><td>AVX512</td><td>40.883</td><td>14240516785</td><td>348323674</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>41.286</td><td>13641914629</td><td>330424711</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>79.045</td><td>14938279042</td><td>188984490</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_avx512</td><td>42.235</td><td>26811007622</td><td>634805437</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>44.107</td><td>27673287721</td><td>627412604</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>55.455</td><td>26631546154</td><td>480237060</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
</table>
</div>



## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

また、平均獲得石数は平均してEgaroucidがEdaxよりも何枚多く石を獲得できたかを表します。この値が大きいほど、Edaxに対して大勝しているということになります。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。bookは双方未使用です。

<div class="table_wrapper"><table>
<tr><th>レベル</th><th>平均獲得石数</th><th>勝率</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th></tr>
<tr><td>1</td><td>10.5</td><td>0.674</td><td>1330(黒: 644 白: 686)</td><td>35(黒: 18 白: 17)</td><td>635(黒: 338 白: 297)</td></tr>
<tr><td>5</td><td>9.042</td><td>0.727</td><td>1399(黒: 719 白: 680)</td><td>108(黒: 52 白: 56)</td><td>493(黒: 229 白: 264)</td></tr>
<tr><td>10</td><td>1.773</td><td>0.605</td><td>1109(黒: 616 白: 493)</td><td>203(黒: 88 白: 115)</td><td>688(黒: 296 白: 392)</td></tr>
<tr><td>15</td><td>1.072</td><td>0.575</td><td>249(黒: 129 白: 120)</td><td>77(黒: 34 白: 43)</td><td>174(黒: 87 白: 87)</td></tr>
<tr><td>21</td><td>0.43</td><td>0.55</td><td>82(黒: 48 白: 34)</td><td>56(黒: 26 白: 30)</td><td>62(黒: 26 白: 36)</td></tr>
</table></div>



