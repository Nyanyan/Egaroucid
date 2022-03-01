# [オセロ研究支援AIアプリ Egaroucid](https://www.egaroucid-app.nyanyan.dev/) 各種ベンチマーク

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="最強レベルAI搭載オセロ研究支援ソフト" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../en/>English</a>

## FFO endgame test

FFO endgame testはオセロAIの終盤探索力の指標として広く使われるベンチマークです。

depthで示される深さを完全読みして、訪問ノード数と探索時間を計測しました。

重要なのはdepth(深さ)、overall time(探索+前処理の時間ミリ秒)、policy(最善手、a1が63でh8が0)、value(評価値)です。

### Core i7-1165G7、8スレッド

<div style="font-size:60%"><pre>#40 endsearch depth 20 overall time 454 search time 437 mpct 10000 policy 55 value 38 nodes 21883562 nps 50076800
#41 endsearch depth 22 overall time 826 search time 809 mpct 10000 policy 32 value 0 nodes 20718255 nps 25609709
#42 endsearch depth 22 overall time 1558 search time 1543 mpct 10000 policy 49 value 6 nodes 46661916 nps 30241034
#43 endsearch depth 23 overall time 3096 search time 3082 mpct 10000 policy 13 value -12 nodes 102837142 nps 33367015
#44 endsearch depth 23 overall time 1138 search time 1118 mpct 10000 policy 52 value -14 nodes 20038331 nps 17923372
#45 endsearch depth 24 overall time 12883 search time 12868 mpct 10000 policy 54 value 6 nodes 620671228 nps 48233698
#46 endsearch depth 24 overall time 3637 search time 3622 mpct 10000 policy 46 value -8 nodes 104409194 nps 28826392
#47 endsearch depth 25 overall time 2158 search time 2140 mpct 10000 policy 49 value 4 nodes 59663965 nps 27880357
#48 endsearch depth 25 overall time 13829 search time 13810 mpct 10000 policy 18 value 28 nodes 521653685 nps 37773619
#49 endsearch depth 26 overall time 17074 search time 17056 mpct 10000 policy 59 value 16 nodes 759255959 nps 44515476
#50 endsearch depth 26 overall time 83807 search time 83789 mpct 10000 policy 4 value 10 nodes 3464616181 nps 41349296
#51 endsearch depth 27 overall time 18537 search time 18520 mpct 10000 policy 51 value 6 nodes 590955123 nps 31909023
#52 endsearch depth 27 overall time 23248 search time 23233 mpct 10000 policy 47 value 0 nodes 713734705 nps 30720729
#53 endsearch depth 28 overall time 251202 search time 251188 mpct 10000 policy 4 value -2 nodes 8562663481 nps 34088664
#54 endsearch depth 28 overall time 342245 search time 342232 mpct 10000 policy 13 value -2 nodes 10929039954 nps 31934593
#55 endsearch depth 29 overall time 1484852 search time 1484835 mpct 10000 policy 17 value 0 nodes 41169384044 nps 27726571
#56 endsearch depth 29 overall time 74321 search time 74283 mpct 10000 policy 24 value 2 nodes 1911766004 nps 25736251
#57 endsearch depth 30 overall time 390517 search time 390501 mpct 10000 policy 23 value -10 nodes 11861268107 nps 30374488
#58 endsearch depth 30 overall time 185752 search time 185737 mpct 10000 policy 57 value 4 nodes 6519574712 nps 35101109
#59 endsearch depth 34 overall time 4047 search time 4031 mpct 10000 policy 1 value 64 nodes 58477674 nps 14506989</pre></div>
### Core i9-11900K、16スレッド

<div style="font-size:60%"><pre>#40 endsearch depth 20 overall time 329 search time 305 mpct 10000 policy 55 value 38 nodes 21899271 nps 71800888
#41 endsearch depth 22 overall time 571 search time 547 mpct 10000 policy 32 value 0 nodes 29195205 nps 53373318
#42 endsearch depth 22 overall time 675 search time 651 mpct 10000 policy 49 value 6 nodes 38560430 nps 59232611
#43 endsearch depth 23 overall time 1359 search time 1335 mpct 10000 policy 41 value -12 nodes 103808885 nps 77759464
#44 endsearch depth 23 overall time 708 search time 684 mpct 10000 policy 52 value -14 nodes 29233404 nps 42738894
#45 endsearch depth 24 overall time 7027 search time 7000 mpct 10000 policy 54 value 6 nodes 740350828 nps 105764404
#46 endsearch depth 24 overall time 2009 search time 1981 mpct 10000 policy 46 value -8 nodes 125044941 nps 63122130
#47 endsearch depth 25 overall time 770 search time 744 mpct 10000 policy 49 value 4 nodes 45921608 nps 61722591
#48 endsearch depth 25 overall time 10347 search time 10317 mpct 10000 policy 18 value 28 nodes 1144183176 nps 110902701
#49 endsearch depth 26 overall time 7885 search time 7856 mpct 10000 policy 59 value 16 nodes 887698342 nps 112996224
#50 endsearch depth 26 overall time 30077 search time 30048 mpct 10000 policy 4 value 10 nodes 2678797208 nps 89150599
#51 endsearch depth 27 overall time 12585 search time 12558 mpct 10000 policy 51 value 6 nodes 960039193 nps 76448414
#52 endsearch depth 27 overall time 13495 search time 13466 mpct 10000 policy 47 value 0 nodes 1000193465 nps 74275468
#53 endsearch depth 28 overall time 119188 search time 119159 mpct 10000 policy 4 value -2 nodes 12209490563 nps 102463855
#54 endsearch depth 28 overall time 135965 search time 135940 mpct 10000 policy 13 value -2 nodes 13338877012 nps 98123267
#55 endsearch depth 29 overall time 778209 search time 778178 mpct 10000 policy 17 value 0 nodes 76506263895 nps 98314606
#56 endsearch depth 29 overall time 35400 search time 35372 mpct 10000 policy 24 value 2 nodes 2258792760 nps 63858214
#57 endsearch depth 30 overall time 142756 search time 142731 mpct 10000 policy 23 value -10 nodes 12676954794 nps 88817109
#58 endsearch depth 30 overall time 50923 search time 50893 mpct 10000 policy 57 value 4 nodes 4423383484 nps 86915361
#59 endsearch depth 34 overall time 1614 search time 1583 mpct 10000 policy 3 value 64 nodes 94965014 nps 59990533</pre></div>




## Edax4.4との対戦

現状世界最強とも言われるオセロAI、[Edax4.4](https://github.com/abulmo/edax-reversi)との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、ある程度手を進めた局面から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

なお、テストに使った局面はEgaroucidの評価関数の学習に使ったものとは別のデータで、最終的に引き分けまたは2石差になったものを使いました。

bookは双方未使用です。

以下に結果を載せますが、大事なのはstart depth(何手目まで打った棋譜で実験したか)、WDL(win-draw-lose)の数字と、Egaroucid win rate(Egaroucidの勝率(分母は引き分けを除外した対戦数))です。Egaroucid win rateが0.5を上回っていればEgaroucidがEdaxに勝ち越しています。

### レベル1同士

両者とも中盤1手読み、終盤2マス空き完全読みです。

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 190-8-202 Egaroucid plays white WDL: 209-13-178 Egaroucid win rate: 0.5121951219512195 
start depth: 20 Egaroucid plays black WDL: 177-9-214 Egaroucid plays white WDL: 210-7-183 Egaroucid win rate: 0.49362244897959184 
start depth: 30 Egaroucid plays black WDL: 175-9-216 Egaroucid plays white WDL: 191-11-198 Egaroucid win rate: 0.46923076923076923
start depth: 40 Egaroucid plays black WDL: 184-16-200 Egaroucid plays white WDL: 225-10-165 Egaroucid win rate: 0.5284237726098191
start depth: 50 Egaroucid plays black WDL: 171-29-200 Egaroucid plays white WDL: 211-30-159 Egaroucid win rate: 0.5155195681511471</pre></div>

### レベル5同士

両者とも中盤5手読み、終盤10マス空き完全読みです。

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 179-14-107 Egaroucid plays white WDL: 161-17-122 Egaroucid win rate: 0.5975395430579965
start depth: 20 Egaroucid plays black WDL: 152-12-136 Egaroucid plays white WDL: 159-18-123 Egaroucid win rate: 0.5456140350877193
start depth: 30 Egaroucid plays black WDL: 163-23-114 Egaroucid plays white WDL: 143-21-136 Egaroucid win rate: 0.5503597122302158
start depth: 40 Egaroucid plays black WDL: 155-14-131 Egaroucid plays white WDL: 144-26-130 Egaroucid win rate: 0.5339285714285714</pre></div>
### レベル15同士

両者中盤15手読みです。実験条件一致のため、前向きな枝刈りに使う確証(Selectivity)を両者73%に揃えました。本来Egaroucidは88%です。

EgaroucidはEdaxより終盤読み切りタイミングが早いため、評価関数の強さを計測するという目的のもと、Edaxが終盤読み切りを始める直前まで両者に打たせてその局面から必勝読みして勝敗を決定しました。

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 63-11-26 Egaroucid plays white WDL: 52-11-37 Egaroucid win rate: 0.6460674157303371
start depth: 20 Egaroucid plays black WDL: 52-9-39 Egaroucid plays white WDL: 49-7-44 Egaroucid win rate: 0.5489130434782609</pre></div>


## 評価関数の精度

評価関数の精度をmse(誤差の2乗の平均値)とmae(誤差の絶対値の平均値)で示します。単位はmseなら石^2、maeなら石です。

評価関数は4手ごと15フェーズに分けて別のものを使っているので、フェーズごとに結果が出ます。フェーズをxとして、フェーズxの評価関数は4x+1から4x+4手目までに使う評価関数です。

注意していただきたいのは、この数字が学習データによるものであって、専用のテストデータによるものではないことです。

<div style="font-size:60%"><pre>phase: 0 player: 0 mse: 204.541 mae: 10.3472
phase: 0 player: 1 mse: 203.126 mae: 10.3203
phase: 1 player: 0 mse: 193.086 mae: 10.0982
phase: 1 player: 1 mse: 189.377 mae: 10.0326
phase: 2 player: 0 mse: 167.657 mae: 9.49416
phase: 2 player: 1 mse: 171.334 mae: 9.59947
phase: 3 player: 0 mse: 153.715 mae: 9.10538
phase: 3 player: 1 mse: 147.108 mae: 8.92787
phase: 4 player: 0 mse: 125.843 mae: 8.25102
phase: 4 player: 1 mse: 118.686 mae: 8.01218
phase: 5 player: 0 mse: 96.8807 mae: 7.21713
phase: 5 player: 1 mse: 89.9291 mae: 6.95694
phase: 6 player: 0 mse: 70.7947 mae: 6.22337
phase: 6 player: 1 mse: 64.1276 mae: 5.93864
phase: 7 player: 0 mse: 44.4569 mae: 4.97553
phase: 7 player: 1 mse: 38.9941 mae: 4.6714
phase: 8 player: 0 mse: 27.5961 mae: 3.98422
phase: 8 player: 1 mse: 26.0039 mae: 3.86099
phase: 9 player: 0 mse: 25.1337 mae: 3.83054
phase: 9 player: 1 mse: 25.4945 mae: 3.86686
phase: 10 player: 0 mse: 27.1559 mae: 4.01309
phase: 10 player: 1 mse: 27.7309 mae: 4.05462
phase: 11 player: 0 mse: 29.705 mae: 4.1883
phase: 11 player: 1 mse: 30.1991 mae: 4.22131
phase: 12 player: 0 mse: 30.7841 mae: 4.25091
phase: 12 player: 1 mse: 30.218 mae: 4.21713
phase: 13 player: 0 mse: 26.6728 mae: 3.95094
phase: 13 player: 1 mse: 24.7366 mae: 3.79713
phase: 14 player: 0 mse: 12.1516 mae: 2.52663
phase: 14 player: 1 mse: 9.08629 mae: 2.101</pre></div>
