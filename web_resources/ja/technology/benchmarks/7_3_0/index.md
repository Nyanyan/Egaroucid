# Egaroucid 7.3.0 ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

比較として、オープンソースで最速クラスのオセロAI [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果も掲載します。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidの結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

[Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>




## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、[Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

バージョン6.3.0までは引き分けを省いて(勝ち)/(勝ち+負け)で勝率を計算していましたが、一般的ではなかったので、バージョン6.4.0からは引き分けを0.5勝として(勝ち+0.5*引き分け)/(勝ち+引き分け+負け)で計算しました。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>


