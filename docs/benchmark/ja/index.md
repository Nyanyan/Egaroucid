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

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 27060281 time 102 nps 265296872
#41 depth 22 value 0 policy h4 nodes 33056487 time 152 nps 217476888
#42 depth 22 value 6 policy g2 nodes 60158183 time 249 nps 241599128
#43 depth 23 value -12 policy c7 nodes 123105384 time 509 nps 241857335
#44 depth 23 value -14 policy d2 nodes 19512855 time 182 nps 107213489
#45 depth 24 value 6 policy b2 nodes 483588331 time 1555 nps 310989280
#46 depth 24 value -8 policy b3 nodes 148832967 time 614 nps 242398969
#47 depth 25 value 4 policy g2 nodes 35991843 time 234 nps 153811294
#48 depth 25 value 28 policy f6 nodes 170617090 time 962 nps 177356642
#49 depth 26 value 16 policy e1 nodes 365884607 time 1673 nps 218699705
#50 depth 26 value 10 policy d8 nodes 1147103368 time 4738 nps 242107084
#51 depth 27 value 6 policy e2 nodes 590452638 time 2682 nps 220153854
#52 depth 27 value 0 policy a3 nodes 666160199 time 2660 nps 250436165
#53 depth 28 value -2 policy d8 nodes 5055604486 time 17028 nps 296899488
#54 depth 28 value -2 policy c7 nodes 7053196821 time 21663 nps 325587260
#55 depth 29 value 0 policy g6 nodes 16591406135 time 72651 nps 228371338
#56 depth 29 value 2 policy h5 nodes 1418277387 time 7423 nps 191065254
#57 depth 30 value -10 policy a6 nodes 2812414286 time 13307 nps 211348484
#58 depth 30 value 4 policy g1 nodes 2650627564 time 11088 nps 239053712
#59 depth 34 value 64 policy g8 nodes 7415 time 3 nps 2471666
23 threads
159.475 sec
39453058327 nodes
247393374.0523593 nps</pre></div>









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
| 15     | 58            | 12   | 30            | 0.66          |

Egaroucid後手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 526           | 25   | 449           | 0.54          |
| 5      | 539           | 52   | 409           | 0.57          |
| 10     | 473           | 115  | 412           | 0.53          |
| 15     | 49            | 18   | 33            | 0.60          |




## 過去バージョン

* [5.4.1](./../5_4_1)
* [5.5.0 と 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.9.0](./../5_10_0)

