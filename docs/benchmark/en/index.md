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
| 15    | 58            | 12   | 30       | 0.66                |

Edax plays first

| Level | Egaroucid win | Draw | Edax win | Egaroucid win ratio |
| ----- | ------------- | ---- | -------- | ------------------- |
| 1     | 526           | 25   | 449      | 0.54                |
| 5     | 539           | 52   | 409      | 0.57                |
| 10    | 473           | 115  | 412      | 0.53                |
| 15    | 49            | 18   | 33       | 0.60                |




## Older versions

* [5.4.1](./../5_4_1)
* [5.5.0 and 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.10.0](./../5_10_0)
