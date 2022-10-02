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

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 26708298 time 154 nps 173430506
#41 depth 22 value 0 policy h4 nodes 42940109 time 231 nps 185887917
#42 depth 22 value 6 policy g2 nodes 44336316 time 192 nps 230918312
#43 depth 23 value -12 policy g3 nodes 121614421 time 660 nps 184264274
#44 depth 23 value -14 policy d2 nodes 30599629 time 245 nps 124896444
#45 depth 24 value 6 policy b2 nodes 517943847 time 2030 nps 255144752
#46 depth 24 value -8 policy b3 nodes 166601078 time 755 nps 220663679
#47 depth 25 value 4 policy g2 nodes 31087376 time 235 nps 132286706
#48 depth 25 value 28 policy f6 nodes 186075560 time 1186 nps 156893389
#49 depth 26 value 16 policy e1 nodes 370947489 time 1793 nps 206886496
#50 depth 26 value 10 policy d8 nodes 1204718003 time 5165 nps 233246467
#51 depth 27 value 6 policy e2 nodes 781121888 time 3410 nps 229068002
#52 depth 27 value 0 policy a3 nodes 613123214 time 2604 nps 235454383
#53 depth 28 value -2 policy d8 nodes 4877119722 time 17510 nps 278533393
#54 depth 28 value -2 policy c7 nodes 8182705438 time 24710 nps 331149552
#55 depth 29 value 0 policy g6 nodes 18024313466 time 79955 nps 225430723
#56 depth 29 value 2 policy h5 nodes 1554001293 time 8738 nps 177844048
#57 depth 30 value -10 policy a6 nodes 2839876130 time 14835 nps 191430814
#58 depth 30 value 4 policy g1 nodes 2863663099 time 12629 nps 226752957
#59 depth 34 value 64 policy g8 nodes 6261 time 7 nps 894428
177.044 sec
178.06661581993103 sec total
42479502637 nodes
239937544.54824787 nps</pre></div>







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
| 11     | 115           | 21   | 64            | 0.64          |
| 15     | 98            | 28   | 74            | 0.57          |

Egaroucid後手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 526           | 25   | 449           | 0.54          |
| 5      | 539           | 52   | 409           | 0.57          |
| 10     | 473           | 115  | 412           | 0.53          |
| 11     | 94            | 26   | 80            | 0.59          |
| 15     | 97            | 35   | 68            | 0.59          |




## 過去バージョン

* [5.4.1](./../5_4_1)
* [5.5.0 と 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.9.0](./../5_10_0)

