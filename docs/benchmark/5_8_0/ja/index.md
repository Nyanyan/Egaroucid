# [オセロ研究支援AIアプリ Egaroucid](https://www.egaroucid-app.nyanyan.dev/) 各種ベンチマーク

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="最強レベルAI搭載オセロ研究支援ソフト" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../en/>English</a>

これはバージョン5.5.0および5.6.0の結果です。

## FFO endgame test

FFO endgame testはオセロAIの終盤探索力の指標として広く使われるベンチマークです。

depthで示される深さを完全読みして、訪問ノード数と探索時間を計測しました。

重要なのはdepth(深さ)、whole time(探索+前処理の時間ミリ秒)、policy(最善手)、value(評価値)です。

### Core i9-11900K、16スレッド

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 30240013 time 208 nps 145384677
#41 depth 22 value 0 policy h4 nodes 30934637 time 282 nps 109697294
#42 depth 22 value 6 policy g2 nodes 38954689 time 323 nps 120602752
#43 depth 23 value -12 policy c7 nodes 113328845 time 764 nps 148336184
#44 depth 23 value -14 policy d2 nodes 24596008 time 400 nps 61490020
#45 depth 24 value 6 policy b2 nodes 597485227 time 2957 nps 202057905
#46 depth 24 value -8 policy b3 nodes 101741641 time 975 nps 104350401
#47 depth 25 value 4 policy g2 nodes 55996875 time 568 nps 98586047
#48 depth 25 value 28 policy f6 nodes 605660062 time 3955 nps 153137815
#49 depth 26 value 16 policy e1 nodes 798738457 time 4756 nps 167943325
#50 depth 26 value 10 policy d8 nodes 2343428530 time 13330 nps 175801090
#51 depth 27 value 6 policy e2 nodes 532335784 time 5417 nps 98271328
#52 depth 27 value 0 policy a3 nodes 490500268 time 4371 nps 112216945
#53 depth 28 value -2 policy d8 nodes 4995871831 time 45686 nps 109352358
#54 depth 28 value -2 policy c7 nodes 8094332723 time 57814 nps 140006446
#55 depth 29 value 0 policy g6 nodes 19667033928 time 153868 nps 127817570
#56 depth 29 value 2 policy h5 nodes 1737263912 time 17839 nps 97385722
#57 depth 30 value -10 policy a6 nodes 5202620937 time 40738 nps 127709287
#58 depth 30 value 4 policy g1 nodes 2560167511 time 20250 nps 126428025
#59 depth 34 value 64 policy g8 nodes 2056 time 26 nps 79076
374.527 sec
48021233934 nodes
128218349.90267725 nps</pre></div>



## Edax4.4との対戦

現状世界最強とも言われるオセロAI、[Edax4.4](https://github.com/abulmo/edax-reversi)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、ある程度手を進めた局面から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

なお、テストに使った局面はEgaroucidの評価関数の学習に使ったものとは別のデータで、[XOT](https://berg.earthlingz.de/xot/index.php)の局面です。

bookは双方未使用です。

以下に結果を載せますが、大事なのはstart depth(何手目まで打った棋譜で実験したか)、WDL(win-draw-lose)の数字と、Egaroucid win rate(Egaroucidの勝率(分母は引き分けを除外した対戦数))です。Egaroucid win rateが0.5を上回っていればEgaroucidがEdaxに勝ち越しています。

レベル設定はEdaxのレベルに合わせました．

<div style="font-size:60%"><pre>level: 1 start depth: 8 Egaroucid plays black WDL: 489-24-487 Egaroucid plays white WDL: 534-27-439 Egaroucid win rate: 0.5248845561826577
level: 5 start depth: 8 Egaroucid plays black WDL: 582-59-359 Egaroucid plays white WDL: 549-44-407 Egaroucid win rate: 0.5962045334739061
level: 10 start depth: 8 Egaroucid plays black WDL: 585-119-296 Egaroucid plays white WDL: 443-124-433 Egaroucid win rate: 0.585088218554354
level: 11 start depth: 8 Egaroucid plays black WDL: 552-132-316 Egaroucid plays white WDL: 501-115-384 Egaroucid win rate: 0.6006845407872219</pre></div>



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

