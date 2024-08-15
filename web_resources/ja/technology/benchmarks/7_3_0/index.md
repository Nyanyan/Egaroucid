# Egaroucid 7.3.0 ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_SIMD</td><td>22.892</td><td>19272934563</td><td>841906978</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_x64_SIMD.txt">000_ffo40_59_Core_i9-13900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_Generic</td><td>39.477</td><td>21898593531</td><td>554717773</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_x64_Generic.txt">001_ffo40_59_Core_i9-13900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86_Generic</td><td>92.359</td><td>19548877069</td><td>211661852</td><td><a href="./files/002_ffo40_59_Core_i9-13900K_x86_Generic.txt">002_ffo40_59_Core_i9-13900K_x86_Generic.txt</a></td>
</tr>
</table>
</div>


比較として、オープンソースで最速クラスのオセロAI [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果も掲載します。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_modern</td><td>23.328</td><td>27849601649</td><td>1193827231</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64</td><td>27.767</td><td>27892181236</td><td>1004508274</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86</td><td>42.098</td><td>27862333601</td><td>661844591</td><td><a href="./files/012_ffo40_59_Core_i9-13900K_edax_x86.txt">012_ffo40_59_Core_i9-13900K_edax_x86.txt</a></td>
</tr>
</table>
</div>


### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidの結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_AVX512</td><td>47.282</td><td>17422173608</td><td>368473702</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_x64_AVX512.txt">100_ffo40_59_Core_i9-11900K_x64_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_SIMD</td><td>48.25</td><td>17524528057</td><td>363202654</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_x64_SIMD.txt">101_ffo40_59_Core_i9-11900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_Generic</td><td>92.032</td><td>18384451854</td><td>199761516</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_x64_Generic.txt">102_ffo40_59_Core_i9-11900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86_Generic</td><td>252.075</td><td>18597008331</td><td>73775695</td><td><a href="./files/103_ffo40_59_Core_i9-11900K_x86_Generic.txt">103_ffo40_59_Core_i9-11900K_x86_Generic.txt</a></td>
</tr>
</table>
</div>


[Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_avx512</td><td>43.19</td><td>26820387942</td><td>620986060</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_modern</td><td>44.11</td><td>26929672444</td><td>610511731</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64</td><td>56.327</td><td>26971739786</td><td>478842115</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86</td><td>89.701</td><td>26583076444</td><td>296352063</td><td><a href="./files/113_ffo40_59_Core_i9-11900K_edax_x86.txt">113_ffo40_59_Core_i9-11900K_edax_x86.txt</a></td>
</tr>
</table>
</div>





## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

バージョン6.3.0までは引き分けを省いて(勝ち)/(勝ち+負け)で勝率を計算していましたが、一般的ではなかったので、バージョン6.4.0からは引き分けを0.5勝として(勝ち+0.5*引き分け)/(勝ち+引き分け+負け)で計算しました。

<div class="table_wrapper"><table>
<tr><th>レベル</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th><th>Egaroucid勝率</th></tr>
<tr><td>1</td><td>1232(黒: 597 白: 635)</td><td>58(黒: 33 白: 25)</td><td>710(黒: 370 白: 340)</td><td>0.63</td></tr>
<tr><td>5</td><td>1313(黒: 673 白: 640)</td><td>109(黒: 52 白: 57)</td><td>578(黒: 275 白: 303)</td><td>0.684</td></tr>
<tr><td>10</td><td>1013(黒: 568 白: 445)</td><td>222(黒: 113 白: 109)</td><td>765(黒: 319 白: 446)</td><td>0.562</td></tr>
<tr><td>15</td><td>228(黒: 125 白: 103)</td><td>103(黒: 52 白: 51)</td><td>169(黒: 73 白: 96)</td><td>0.559</td></tr>
<tr><td>21</td><td>78(黒: 47 白: 31)</td><td>54(黒: 25 白: 29)</td><td>68(黒: 28 白: 40)</td><td>0.525</td></tr>
</table></div>



