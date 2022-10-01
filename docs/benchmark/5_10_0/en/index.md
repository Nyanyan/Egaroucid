# [Othello research support AI app Egaroucid](https://www.egaroucid-app.nyanyan.dev/) Benchmarks

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="Othello research support AI app Egaroucid" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../ja/>日本語</a>

Version 5.10.0

## FFO endgame test

FFO endgame test is a famous test to evaluate endgame solvers.

I did complete searches for this test and got the time to solve it and number of nodes searched.

* depth: search depth
* whole time: search time including some other calculation in milliseconds
* policy: the best move
* nodes: number of nodes visited
* nps: node per second

### Core i9-11900K, 16 threads

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 19437182 time 152 nps 127876197
#41 depth 22 value 0 policy h4 nodes 28051331 time 254 nps 110438311
#42 depth 22 value 6 policy g2 nodes 41090111 time 348 nps 118075031
#43 depth 23 value -12 policy c7 nodes 86703422 time 545 nps 159088847
#44 depth 23 value -14 policy d2 nodes 15931449 time 293 nps 54373546
#45 depth 24 value 6 policy b2 nodes 534802372 time 2520 nps 212223163
#46 depth 24 value -8 policy b3 nodes 103987485 time 924 nps 112540568
#47 depth 25 value 4 policy g2 nodes 39916702 time 443 nps 90105422
#48 depth 25 value 28 policy f6 nodes 167648119 time 1615 nps 103806884
#49 depth 26 value 16 policy e1 nodes 273867075 time 2482 nps 110341287
#50 depth 26 value 10 policy d8 nodes 1177832829 time 7292 nps 161523975
#51 depth 27 value 6 policy a3 nodes 263723106 time 2488 nps 105998032
#52 depth 27 value 0 policy a3 nodes 431703973 time 3830 nps 112716442
#53 depth 28 value -2 policy d8 nodes 5006612436 time 28493 nps 175713769
#54 depth 28 value -2 policy c7 nodes 6461535397 time 27352 nps 236236304
#55 depth 29 value 0 policy g6 nodes 22750219689 time 123158 nps 184723848
#56 depth 29 value 2 policy h5 nodes 1221714952 time 10716 nps 114008487
#57 depth 30 value -10 policy a6 nodes 1997564547 time 13915 nps 143554764
#58 depth 30 value 4 policy g1 nodes 1964184672 time 13976 nps 140539830
#59 depth 34 value 64 policy g8 nodes 13079 time 38 nps 344184
240.834 sec
242.29370999336243 sec total
42586539928 nodes
176829434.08322746 nps</pre></div>








## Play against Edax4.4

Edax is one of the best othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different lines. These lines, boards in [XOT](https://berg.earthlingz.de/xot/index.php), are not in learning data of evaluation function.

No book used.

if the win rate is over 0.5, Egaroucid win more than Edax do.

Levels are exactly same as Edax 4.4

Egaroucid plays first

| Level | Egaroucid Win | Draw | Egaroucid Loss | Egaroucid Win Rate |
| ----- | ------------- | ---- | -------------- | ------------------ |
| 1     | 497           | 24   | 479            | 0.51               |
| 5     | 573           | 52   | 375            | 0.60               |
| 10    | 536           | 131  | 333            | 0.62               |

Edax plays first

| Level | Egaroucid Win | Draw | Egaroucid Loss | Egaroucid Win Rate |
| ----- | ------------- | ---- | -------------- | ------------------ |
| 1     | 532           | 22   | 446            | 0.54               |
| 5     | 535           | 54   | 411            | 0.57               |
| 10    | 468           | 109  | 423            | 0.53               |



## Accuracy of evaluation function

The mse (mean squared error) and mae (mean absolute error) of my evaluation function.

30 evaluation functions are used in Egaroucid, for each 2 moves.

Evaluation function of phase X is used 2X + 1 moves to 2X + 2 moves.

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



## Other versions

* [5.4.1](./../5_4_1)
* [5.5.0 and 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
