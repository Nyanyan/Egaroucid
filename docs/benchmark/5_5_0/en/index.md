# [Othello research support AI app Egaroucid](https://www.egaroucid-app.nyanyan.dev/) Benchmarks

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="Othello research support AI app Egaroucid" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../ja/>日本語</a>

Version 5.5.0 and 5.6.0

## FFO endgame test

FFO endgame test is a famous test to evaluate endgame solvers.

I did complete searches for this test and got the time to solve it and number of nodes searched.

* depth: search depth
* whole time: search time including some other calculation in milliseconds
* policy: the best move
* nodes: number of nodes visited
* nps: node per second

### Core i9-11900K, 16 threads

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 27871488 whole time 135 search time 128 nps 217746000
#41 depth 22 value 0 policy h4 nodes 28862801 whole time 199 search time 191 nps 151114141
#42 depth 22 value 6 policy g2 nodes 48660062 whole time 213 search time 205 nps 237366156
#43 depth 23 value -12 policy c7 nodes 166809244 whole time 776 search time 769 nps 216917092
#44 depth 23 value -14 policy d2 nodes 42549145 whole time 323 search time 316 nps 134649193
#45 depth 24 value 6 policy b2 nodes 912441891 whole time 3241 search time 3226 nps 282840015
#46 depth 24 value -8 policy b3 nodes 227129647 whole time 1089 search time 1073 nps 211677210
#47 depth 25 value 4 policy g2 nodes 151105116 whole time 852 search time 837 nps 180531799
#48 depth 25 value 28 policy f6 nodes 724631805 whole time 3025 search time 3011 nps 240661509
#49 depth 26 value 16 policy e1 nodes 1079040603 whole time 4537 search time 4514 nps 239043110
#50 depth 26 value 10 policy d8 nodes 3688114040 whole time 13943 search time 13922 nps 264912659
#51 depth 27 value 6 policy a3 nodes 2782257813 whole time 14732 search time 14709 nps 189153430
#52 depth 27 value 0 policy a3 nodes 604311247 whole time 3785 search time 3762 nps 160635631
#53 depth 28 value -2 policy d8 nodes 11344109584 whole time 69739 search time 69586 nps 163022872
#54 depth 28 value -2 policy c7 nodes 12621020703 whole time 63972 search time 63874 nps 197592458
#55 depth 29 value 0 policy b7 nodes 78910958123 whole time 458573 search time 458442 nps 172128553
#56 depth 29 value 2 policy h5 nodes 3533955047 whole time 25576 search time 25526 nps 138445312
#57 depth 30 value -10 policy a6 nodes 9948249883 whole time 59654 search time 59586 nps 166956162
#58 depth 30 value 4 policy g1 nodes 3949220953 whole time 23824 search time 23781 nps 166066227
#59 depth 34 value 64 policy g8 nodes 7096 whole time 43 search time 11 nps 645090
748.231 sec
747.469 sec search
802.5422124862671 sec total
130791306291 nodes
174978903.86223373 nps</pre></div>




## Play against Edax4.4

Edax is one of the best othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different lines. These lines are not in learning data of evaluation function.

No book used.

* start depth: the depth of the line
* WDL: win-draw-lose
* Egaroucid win rate: Egaroucid's win rate excluding draws

if the win rate is over 0.5, Egaroucid win more than Edax do.

### Level 1

Lookahead depth is 1, complete search depth is 2

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 148-7-145 Egaroucid plays white WDL: 181-10-109 Egaroucid win rate: 0.5643224699828473
start depth: 20 Egaroucid plays black WDL: 154-8-138 Egaroucid plays white WDL: 165-4-131 Egaroucid win rate: 0.5425170068027211
</pre></div>


### Level 5

Lookahead depth is 5, complete search depth is 10

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 107-8-85 Egaroucid plays white WDL: 114-12-74 Egaroucid win rate: 0.5815789473684211
start depth: 20 Egaroucid plays black WDL: 105-12-83 Egaroucid plays white WDL: 97-7-96 Egaroucid win rate: 0.5301837270341208</pre></div>

### Level 15

Lookahead depth is 15. Selectivity is 73% (to align the conditions, Egaroucid uses the same selectivity as Edax). The depth of complete search is also the same.

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 50-11-39 Egaroucid plays white WDL: 52-9-39 Egaroucid win rate: 0.5666666666666667
start depth: 20 Egaroucid plays black WDL: 61-10-29 Egaroucid plays white WDL: 48-6-46 Egaroucid win rate: 0.592391304347826</pre></div>


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

