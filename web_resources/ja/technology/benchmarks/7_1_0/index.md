# Egaroucid 7.1.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>SIMD</td><td>27.75</td><td>25746866156</td><td>927814996</td><td><a href="./files/0_Core_i9-13900K_SIMD.txt">0_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>Generic</td><td>48.578</td><td>27178268841</td><td>559476899</td><td><a href="./files/1_Core_i9-13900K_Generic.txt">1_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>AVX512</td><td>54.842</td><td>23263286724</td><td>424187424</td><td><a href="./files/2_Core_i9-11900K_AVX512.txt">2_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>SIMD</td><td>55.464</td><td>23143759716</td><td>417275344</td><td><a href="./files/3_Core_i9-11900K_SIMD.txt">3_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>Generic</td><td>112.36</td><td>23298639098</td><td>207357058</td><td><a href="./files/4_Core_i9-11900K_Generic.txt">4_Core_i9-11900K_Generic.txt</a></td>
</tr>
</table>
</div>






## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a>との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには<a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a>に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

バージョン6.3.0までは引き分けを省いて(勝ち)/(勝ち+負け)で勝率を計算していましたが、一般的ではなかったので、バージョン6.4.0からは引き分けを0.5勝として(勝ち+0.5*引き分け)/(勝ち+引き分け+負け)で計算しました。

<div class="table_wrapper"><table>
<tr><th>レベル</th><th>Egaroucid勝ち</th><th>引分</th><th>Edax勝ち</th><th>Egaroucid勝率</th></tr>
<tr><td>1</td><td>1232(黒: 594 白: 638)</td><td>53(黒: 30 白: 23)</td><td>715(黒: 376 白: 339)</td><td>0.629</td></tr>
<tr><td>5</td><td>1255(黒: 642 白: 613)</td><td>102(黒: 53 白: 49)</td><td>643(黒: 305 白: 338)</td><td>0.653</td></tr>
<tr><td>10</td><td>1016(黒: 565 白: 451)</td><td>234(黒: 109 白: 125)</td><td>750(黒: 326 白: 424)</td><td>0.567</td></tr>
<tr><td>15</td><td>215(黒: 113 白: 102)</td><td>111(黒: 51 白: 60)</td><td>174(黒: 86 白: 88)</td><td>0.541</td></tr>
<tr><td>21</td><td>85(黒: 57 白: 28)</td><td>50(黒: 19 白: 31)</td><td>65(黒: 24 白: 41)</td><td>0.55</td></tr>
</table></div>


