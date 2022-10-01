# [オセロ研究支援AIアプリ Egaroucid](https://www.egaroucid-app.nyanyan.dev/) 各種ベンチマーク

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="最強レベルAI搭載オセロ研究支援ソフト" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../en/>English</a>

これはバージョン6.0.0の結果です。

## FFO endgame test

FFO endgame testはオセロAIの終盤探索力の指標として広く使われるベンチマークです。

depthで示される深さを完全読みして、訪問ノード数と探索時間を計測しました。各項目は以下の情報を表します。

* depth: 読み深さ
* time: 探索時間
* policy: 最善手
* nodes: 訪問ノード数
* nps: 1秒あたりのノード訪問数

### Core i9-11900K

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 25582789 time 179 nps 142920608
#41 depth 22 value 0 policy h4 nodes 34534657 time 203 nps 170121463
#42 depth 22 value 6 policy g2 nodes 45448022 time 221 nps 205647158
#43 depth 23 value -12 policy g3 nodes 115894371 time 640 nps 181084954
#44 depth 23 value -14 policy d2 nodes 30071323 time 276 nps 108954068
#45 depth 24 value 6 policy b2 nodes 500460645 time 1961 nps 255206856
#46 depth 24 value -8 policy b3 nodes 171538835 time 859 nps 199695966
#47 depth 25 value 4 policy g2 nodes 28275799 time 251 nps 112652585
#48 depth 25 value 28 policy f6 nodes 198596283 time 1210 nps 164129159
#49 depth 26 value 16 policy e1 nodes 346430224 time 1682 nps 205963272
#50 depth 26 value 10 policy d8 nodes 1316812303 time 5687 nps 231547793
#51 depth 27 value 6 policy e2 nodes 694688172 time 3248 nps 213881826
#52 depth 27 value 0 policy a3 nodes 648565467 time 2756 nps 235328543
#53 depth 28 value -2 policy d8 nodes 4562379057 time 16301 nps 279883384
#54 depth 28 value -2 policy c7 nodes 7620802496 time 24178 nps 315195735
#55 depth 29 value 0 policy g6 nodes 17205705816 time 80931 nps 212597222
#56 depth 29 value 2 policy h5 nodes 1380209141 time 7747 nps 178160467
#57 depth 30 value -10 policy a6 nodes 3588039500 time 17106 nps 209753273
#58 depth 30 value 4 policy g1 nodes 2668367658 time 11814 nps 225864877
#59 depth 34 value 64 policy g8 nodes 5597 time 7 nps 799571
23 threads
177.257 sec
41182408155 nodes
232331632.3473826 nps</pre></div>






## Edax4.4との対戦

現状世界最強とも言われるオセロAI、[Edax4.4](https://github.com/abulmo/edax-reversi)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、ある程度手を進めた局面から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

なお、テストに使った局面はEgaroucidの評価関数の学習に使ったものとは別のデータで、[XOT](https://berg.earthlingz.de/xot/index.php)の局面です。

bookは双方未使用です。

Egaroucid先手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 493           | 18   | 489           | 0.50          |
| 5      | 590           | 57   | 353           | 0.63          |
| 10     | 598           | 115  | 287           | 0.68          |
| 15     | 50            | 18   | 32            | 0.61          |

Egaroucid後手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 526           | 25   | 449           | 0.54          |
| 5      | 539           | 52   | 409           | 0.57          |
| 10     | 473           | 115  | 412           | 0.53          |
| 15     | 36            | 21   | 43            | 0.46          |




## 過去バージョン

* [5.4.1](./../5_4_1)
* [5.5.0 と 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.9.0](./../5_10_0)

