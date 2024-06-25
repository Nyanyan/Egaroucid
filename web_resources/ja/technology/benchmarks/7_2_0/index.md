# Egaroucid 7.2.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>版</th><th>時間(秒)</th><th>ノード数</th><th>NPS</th><th>ファイル</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>SIMD</td><td>27.364</td><td>26421069847</td><td>965541216</td><td><a href="./files/0_Core_i9-13900K_SIMD.txt">0_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>Generic</td><td>47.844</td><td>28178107090</td><td>588958011</td><td><a href="./files/1_Core_i9-13900K_Generic.txt">1_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>AVX512</td><td>54.558</td><td>22702428048</td><td>416115474</td><td><a href="./files/2_Core_i9-11900K_AVX512.txt">2_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>SIMD</td><td>54.676</td><td>22894546165</td><td>418731183</td><td><a href="./files/3_Core_i9-11900K_SIMD.txt">3_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>Generic</td><td>115.479</td><td>23069092108</td><td>199768720</td><td><a href="./files/4_Core_i9-11900K_Generic.txt">4_Core_i9-11900K_Generic.txt</a></td>
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
<tr><td>1</td><td>1236(黒: 586 白: 650)</td><td>51(黒: 33 白: 18)</td><td>713(黒: 381 白: 332)</td><td>0.631</td></tr>
<tr><td>5</td><td>1348(黒: 679 白: 669)</td><td>88(黒: 49 白: 39)</td><td>564(黒: 272 白: 292)</td><td>0.696</td></tr>
<tr><td>10</td><td>1028(黒: 563 白: 465)</td><td>233(黒: 117 白: 116)</td><td>739(黒: 320 白: 419)</td><td>0.572</td></tr>
<tr><td>15</td><td>238(黒: 134 白: 104)</td><td>102(黒: 43 白: 59)</td><td>160(黒: 73 白: 87)</td><td>0.578</td></tr>
<tr><td>21</td><td>79(黒: 46 白: 33)</td><td>60(黒: 34 白: 26)</td><td>61(黒: 20 白: 41)</td><td>0.545</td></tr>
</table></div>



