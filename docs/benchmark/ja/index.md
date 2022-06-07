# [オセロ研究支援AIアプリ Egaroucid](https://www.egaroucid-app.nyanyan.dev/) 各種ベンチマーク

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="最強レベルAI搭載オセロ研究支援ソフト" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../en/>English</a>

これはバージョン5.9.0の結果です。

## FFO endgame test

FFO endgame testはオセロAIの終盤探索力の指標として広く使われるベンチマークです。

depthで示される深さを完全読みして、訪問ノード数と探索時間を計測しました。

重要なのはdepth(深さ)、whole time(探索+前処理の時間ミリ秒)、policy(最善手)、value(評価値)です。

### Core i9-11900K、16スレッド

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 18499320 time 146 nps 126707671
#41 depth 22 value 0 policy h4 nodes 26526311 time 256 nps 103618402
#42 depth 22 value 6 policy g2 nodes 40548462 time 350 nps 115852748
#43 depth 23 value -12 policy c7 nodes 82911207 time 574 nps 144444611
#44 depth 23 value -14 policy d2 nodes 15270420 time 301 nps 50732292
#45 depth 24 value 6 policy b2 nodes 498339937 time 2533 nps 196739019
#46 depth 24 value -8 policy b3 nodes 76840991 time 837 nps 91805246
#47 depth 25 value 4 policy g2 nodes 37334630 time 435 nps 85826735
#48 depth 25 value 28 policy f6 nodes 204129724 time 1838 nps 111060785
#49 depth 26 value 16 policy e1 nodes 256316623 time 2496 nps 102690954
#50 depth 26 value 10 policy d8 nodes 1179213508 time 7966 nps 148030819
#51 depth 27 value 6 policy a3 nodes 249226421 time 2557 nps 97468291
#52 depth 27 value 0 policy a3 nodes 560569333 time 5363 nps 104525327
#53 depth 28 value -2 policy d8 nodes 4883501669 time 30118 nps 162145616
#54 depth 28 value -2 policy c7 nodes 6314578504 time 28803 nps 219233361
#55 depth 29 value 0 policy g6 nodes 23384774474 time 138860 nps 168405404
#56 depth 29 value 2 policy h5 nodes 1140351709 time 11095 nps 102780685
#57 depth 30 value -10 policy a6 nodes 1668140404 time 13494 nps 123620898
#58 depth 30 value 4 policy g1 nodes 1921933011 time 15759 nps 121957802
#59 depth 34 value 64 policy g8 nodes 28904 time 23 nps 1256695
263.804 sec
264.7655096054077 sec total
42559035562 nodes
161328242.03575382 nps</pre></div>




## Edax4.4との対戦

現状世界最強とも言われるオセロAI、[Edax4.4](https://github.com/abulmo/edax-reversi)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、ある程度手を進めた局面から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

なお、テストに使った局面はEgaroucidの評価関数の学習に使ったものとは別のデータで、[XOT](https://berg.earthlingz.de/xot/index.php)の局面です。

bookは双方未使用です。

レベル設定はEdaxのレベルに合わせました。

Egaroucid先手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 497           | 24   | 479           | 0.51          |
| 5      | 573           | 52   | 375           | 0.60          |
| 10     | 536           | 131  | 333           | 0.62          |

Egaroucid後手の結果

| レベル | Egaroucid勝ち | 引分 | Egaroucid負け | Egaroucid勝率 |
| ------ | ------------- | ---- | ------------- | ------------- |
| 1      | 532           | 22   | 446           | 0.54          |
| 5      | 535           | 54   | 411           | 0.57          |
| 10     | 468           | 109  | 423           | 0.53          |


## 評価関数の精度

評価関数の精度をmse(誤差の2乗の平均値)とmae(誤差の絶対値の平均値)で示します。単位はmseなら石^2、maeなら石です。

評価関数は2手ごと30フェーズに分けて別のものを使っているので、フェーズごとに結果が出ます。フェーズをxとして、フェーズxの評価関数は2x+1から2x+2手目までに使う評価関数です。

注意していただきたいのは、この数字が学習データによるものであって、専用のテストデータによるものではないことです。

<div style="font-size:60%"><pre>phase 0 mse 184.415 mae 9.79577
phase 1 mse 181.731 mae 9.74168
phase 2 mse 176.078 mae 9.61612
phase 3 mse 166.574 mae 9.39029
phase 4 mse 153.356 mae 9.05595
phase 5 mse 273.548 mae 12.0107
phase 6 mse 94.0067 mae 7.18964
phase 7 mse 87.0103 mae 6.94262
phase 8 mse 80.5171 mae 6.70287
phase 9 mse 73.5077 mae 6.42004
phase 10 mse 67.8453 mae 6.13735
phase 11 mse 62.4096 mae 5.91408
phase 12 mse 57.1586 mae 5.69015
phase 13 mse 51.493 mae 5.42289
phase 14 mse 45.5248 mae 5.11642
phase 15 mse 39.6584 mae 4.78097
phase 16 mse 36.1119 mae 4.56925
phase 17 mse 32.6206 mae 4.34778
phase 18 mse 29.1944 mae 4.12316
phase 19 mse 28.7966 mae 4.10928
phase 20 mse 29.1392 mae 4.15409
phase 21 mse 29.9168 mae 4.22495
phase 22 mse 31.1575 mae 4.32002
phase 23 mse 33.4589 mae 4.478
phase 24 mse 33.8493 mae 4.50079
phase 25 mse 32.3792 mae 4.39883
phase 26 mse 29.3648 mae 4.17235
phase 27 mse 24.5691 mae 3.80085
phase 28 mse 16.3138 mae 3.05533
phase 29 mse 4.51746 mae 1.3841</pre></div>



## 過去バージョン

* [5.4.1](./../5_4_1)
* [5.5.0 と 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)

