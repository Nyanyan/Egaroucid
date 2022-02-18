# [Othello research support AI app Egaroucid](https://www.egaroucid-app.nyanyan.dev/) Benchmarks

<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-text="Othello research support AI app Egaroucid" data-url="https://www.egaroucid-app.nyanyan.dev/" data-hashtags="egaroucid" data-related="takuto_yamana,Nyanyan_Cube" data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> <a href=./../ja/>日本語</a>

## FFO endgame test

FFO endgame test is a famous test to evaluate endgame solvers.

I did complete searches for this test and got the time to solve it and number of nodes searched.

* depth: search depth
* overall time: search time including some other calculation in milliseconds
* policy: the best move (a0 is 63, h8 is 0)
* nodes: number of nodes visited
* nps: node per second

### Core i7-1165G7, 8 threads

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

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 190-8-202 Egaroucid plays white WDL: 209-13-178 Egaroucid win rate: 0.5121951219512195 
start depth: 20 Egaroucid plays black WDL: 177-9-214 Egaroucid plays white WDL: 210-7-183 Egaroucid win rate: 0.49362244897959184 
start depth: 30 Egaroucid plays black WDL: 175-9-216 Egaroucid plays white WDL: 191-11-198 Egaroucid win rate: 0.46923076923076923
start depth: 40 Egaroucid plays black WDL: 184-16-200 Egaroucid plays white WDL: 225-10-165 Egaroucid win rate: 0.5284237726098191
start depth: 50 Egaroucid plays black WDL: 171-29-200 Egaroucid plays white WDL: 211-30-159 Egaroucid win rate: 0.5155195681511471</pre></div>

### Level 5

Lookahead depth is 5, complete search depth is 10

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 179-14-107 Egaroucid plays white WDL: 161-17-122 Egaroucid win rate: 0.5975395430579965
start depth: 20 Egaroucid plays black WDL: 152-12-136 Egaroucid plays white WDL: 159-18-123 Egaroucid win rate: 0.5456140350877193
start depth: 30 Egaroucid plays black WDL: 163-23-114 Egaroucid plays white WDL: 143-21-136 Egaroucid win rate: 0.5503597122302158
start depth: 40 Egaroucid plays black WDL: 155-14-131 Egaroucid plays white WDL: 144-26-130 Egaroucid win rate: 0.5339285714285714</pre></div>
### Level 15

Lookahead depth is 15. Selectivity is 73% (Egaroucid's original selectivity is 88%, but this time, to align the conditions, Egaroucid uses 73% selectivity)

Edax do complete search earlier than Egaroucid and the purpose of this experiment is the strength of the evaluation function. So Edax and Egaroucid plays until 32 moves, which is the last move of Edax's midgame search, then do WDL search to see which will win.

<div style="font-size:60%"><pre>start depth: 10 Egaroucid plays black WDL: 63-11-26 Egaroucid plays white WDL: 52-11-37 Egaroucid win rate: 0.6460674157303371
start depth: 20 Egaroucid plays black WDL: 52-9-39 Egaroucid plays white WDL: 49-7-44 Egaroucid win rate: 0.5489130434782609</pre></div>

## Accuracy of evaluation function

The mse (mean squared error) and mae (mean absolute error) of my evaluation function.

30 evaluation functions are used in Egaroucid, for each 4 moves and each player.

Evaluation function of phase X is used 4X + 1 moves to 4X + 4 moves.

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
