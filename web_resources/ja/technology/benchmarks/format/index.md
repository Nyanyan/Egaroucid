# Egaroucid !!VERSION!! ベンチマーク

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html)はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。ここでは、The FFO endgame test suiteのうち40番から59番を実行した結果を掲載します。

### Core i9-13900K

Core i9-13900KではAVX512版が動かないため、SIMD版、Generic版、x86版の結果を掲載します。

Egaroucidの結果は以下の通りです。また、比較としてオープンソースで最速クラスのオセロAI [Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)の結果も掲載します (Edaxはバージョン4.6が最新ですが、手元で実行したところ4.6よりも4.5.5の方が速かったため、4.5.5を採用しました)。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

### Core i9-11900K

Core i9-11900KではAVX512版が動きます。

Egaroucidおよび[Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)の結果は以下の通りです。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>


## Edax 4.6との対戦

世界最強クラスのオープンソースオセロAI、[Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにし、2戦で獲得した石数が多い方が勝ちとしました。

勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。全ての条件でEgaroucidが勝ち越しています。

また、平均獲得石数は平均してEgaroucidがEdaxよりも何枚多く石を獲得できたかを表します。この値が大きいほど、Edaxに対して大勝しているということになります。

テストには[XOT](https://berg.earthlingz.de/xot/index.php)に収録されている局面を使用しました。bookは双方未使用です。

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>


