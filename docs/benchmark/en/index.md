# [Othello research support AI app Egaroucid](https://www.egaroucid-app.nyanyan.dev/) Benchmarks

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="Othello research support AI app Egaroucid" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../ja/>日本語</a>

Version 6.0.0

## FFO endgame test

FFO endgame test is a famous test to evaluate endgame solvers.

I did complete searches for this test and got the time to solve it and number of nodes searched.

* depth: search depth
* time: search time
* policy: the best move
* nodes: number of nodes visited
* nps: node per second

### Core i9-11900K

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 25762481 time 149 nps 172902557
#41 depth 22 value 0 policy h4 nodes 43082407 time 263 nps 163811433
#42 depth 22 value 6 policy g2 nodes 41356640 time 222 nps 186291171
#43 depth 23 value -12 policy c7 nodes 110484289 time 655 nps 168678303
#44 depth 23 value -14 policy d2 nodes 18462790 time 228 nps 80977149
#45 depth 24 value 6 policy b2 nodes 456685904 time 2255 nps 202521465
#46 depth 24 value -8 policy b3 nodes 149245136 time 719 nps 207573207
#47 depth 25 value 4 policy g2 nodes 31094957 time 216 nps 143958134
#48 depth 25 value 28 policy f6 nodes 179961365 time 1181 nps 152380495
#49 depth 26 value 16 policy e1 nodes 405509512 time 2002 nps 202552203
#50 depth 26 value 10 policy d8 nodes 1231298288 time 5367 nps 229420213
#51 depth 27 value 6 policy e2 nodes 617973874 time 2943 nps 209980928
#52 depth 27 value 0 policy a3 nodes 583499133 time 2516 nps 231915394
#53 depth 28 value -2 policy d8 nodes 4633947259 time 18462 nps 250999201
#54 depth 28 value -2 policy c7 nodes 7127192303 time 23963 nps 297424875
#55 depth 29 value 0 policy g6 nodes 17225651509 time 80626 nps 213648841
#56 depth 29 value 2 policy h5 nodes 1348674675 time 7733 nps 174405104
#57 depth 30 value -10 policy a6 nodes 3229265024 time 15977 nps 202119610
#58 depth 30 value 4 policy g1 nodes 2816182635 time 12609 nps 223347024
#59 depth 34 value 64 policy g8 nodes 6452 time 8 nps 806500
23 threads
178.094 sec
40275336633 nodes
226146510.4551529 nps</pre></div>











## Play against Edax4.4

[Edax4.4](https://github.com/abulmo/edax-reversi) is one of the best othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different lines. These lines, boards in [XOT](https://berg.earthlingz.de/xot/index.php), are not in learning data of evaluation function.

No book used.

if the win rate is over 0.5, Egaroucid win more than Edax do.

Egaroucid plays first

| Level | Egaroucid win | Draw | Edax win | Egaroucid win ratio |
| ----- | ------------- | ---- | -------- | ------------------- |
| 1     | 493           | 18   | 489      | 0.50                |
| 5     | 590           | 57   | 353      | 0.63                |
| 10    | 598           | 115  | 287      | 0.68                |
| 15    | 45            | 20   | 35       | 0.56                |

Edax plays first

| Level | Egaroucid win | Draw | Edax win | Egaroucid win ratio |
| ----- | ------------- | ---- | -------- | ------------------- |
| 1     | 526           | 25   | 449      | 0.54                |
| 5     | 539           | 52   | 409      | 0.57                |
| 10    | 473           | 115  | 412      | 0.53                |
| 15    | 41            | 21   | 38       | 0.52                |




## Older versions

* [5.4.1](./../5_4_1)
* [5.5.0 and 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.10.0](./../5_10_0)
