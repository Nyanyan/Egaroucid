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
| 11    | 115           | 21   | 64       | 0.64                |
| 15    | 98            | 28   | 74       | 0.57                |

Edax plays first

| Level | Egaroucid win | Draw | Edax win | Egaroucid win ratio |
| ----- | ------------- | ---- | -------- | ------------------- |
| 1     | 526           | 25   | 449      | 0.54                |
| 5     | 539           | 52   | 409      | 0.57                |
| 10    | 473           | 115  | 412      | 0.53                |
| 11    | 94            | 26   | 80       | 0.59                |
| 15    | 97            | 35   | 68       | 0.59                |




## Older versions

* [5.4.1](./../5_4_1)
* [5.5.0 and 5.6.0](./../5_5_0)
* [5.7.0](./../5_7_0)
* [5.8.0](./../5_8_0)
* [5.9.0](./../5_9_0)
* [5.10.0](./../5_10_0)
