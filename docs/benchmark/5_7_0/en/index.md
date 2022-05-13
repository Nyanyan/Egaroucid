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

<div style="font-size:60%"><pre>#40 depth 20 value 38 policy a2 nodes 32617519 time 187 nps 174425235
#41 depth 22 value 0 policy h4 nodes 26560625 time 242 nps 109754648
#42 depth 22 value 6 policy g2 nodes 51191219 time 360 nps 142197830
#43 depth 23 value -12 policy c7 nodes 168771687 time 929 nps 181670276
#44 depth 23 value -14 policy d2 nodes 43705622 time 626 nps 69817287
#45 depth 24 value 6 policy b2 nodes 973544606 time 4256 nps 228746382
#46 depth 24 value -8 policy b3 nodes 169047118 time 2315 nps 73022513
#47 depth 25 value 4 policy g2 nodes 115350803 time 816 nps 141361278
#48 depth 25 value 28 policy f6 nodes 822747204 time 4137 nps 198875321
#49 depth 26 value 16 policy e1 nodes 1021389690 time 6387 nps 159916970
#50 depth 26 value 10 policy d8 nodes 3470416861 time 15945 nps 217649223
#51 depth 27 value 6 policy a3 nodes 2093326335 time 19308 nps 108417564
#52 depth 27 value 0 policy a3 nodes 666596389 time 7950 nps 83848602
#53 depth 28 value -2 policy d8 nodes 12227417145 time 67989 nps 179844050
#54 depth 28 value -2 policy c7 nodes 13290990536 time 71954 nps 184715103
#55 depth 29 value 0 policy b7 nodes 41506305139 time 276500 nps 150113219
#56 depth 29 value 2 policy h5 nodes 3409497345 time 28172 nps 121024327
#57 depth 30 value -10 policy a6 nodes 13317806725 time 88863 nps 149868974
#58 depth 30 value 4 policy g1 nodes 4861384409 time 31941 nps 152198879
#59 depth 34 value 64 policy g8 nodes 2975 time 1624 nps 1831
630.501 sec
631.2902810573578 sec total
98268669952 nodes
155858071.52090162 nps</pre></div>





## Play against Edax4.4

Edax is one of the best othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different lines. These lines, boards in [XOT](https://berg.earthlingz.de/xot/index.php), are not in learning data of evaluation function.

No book used.

* start depth: the depth of the line
* WDL: win-draw-lose
* Egaroucid win rate: Egaroucid's win rate excluding draws

if the win rate is over 0.5, Egaroucid win more than Edax do.

### Level 1

Lookahead depth is 1, complete search depth is 2

<div style="font-size:60%"><pre>start depth: 8 Egaroucid plays black WDL: 61-2-67 Egaroucid plays white WDL: 76-3-51 Egaroucid win rate: 0.5372549019607843
</pre></div>

### Level 5

Lookahead depth is 5, complete search depth is 10

<div style="font-size:60%"><pre>level 5
start depth: 8 Egaroucid plays black WDL: 68-6-56 Egaroucid plays white WDL: 67-11-52 Egaroucid win rate: 0.5555555555555556</pre></div>


### Level 11

Lookahead depth is 11. Selectivity is 73% (to align the conditions, Egaroucid uses the same selectivity as Edax). The depth of complete search is also the same.

<div style="font-size:60%"><pre>level 11
start depth: 8 Egaroucid plays black WDL: 69-18-43 Egaroucid plays white WDL: 61-15-54 Egaroucid win rate: 0.5726872246696035</pre></div>



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

