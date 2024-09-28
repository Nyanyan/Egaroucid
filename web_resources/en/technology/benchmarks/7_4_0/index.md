# Egaroucid 7.4.0 Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are:

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

As a comparison, I also show [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result:

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>
### Core i9-11900K

With Core i9-11900K, you can run AVX512 edition.

Egaroucid's result are:

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>

[Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result are:

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>







## Play against Edax 4.4

[Edax](https://github.com/abulmo/edax-reversi/releases/tag/v4.4) 4.4 is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

I used [XOT](https://berg.earthlingz.de/xot/index.php) for its testcases.

No opening books used.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

<div class="table_wrapper"><table>
<tr><th>Level</th><th>Egaroucid win</th><th>Draw</th><th>Edax Win</th><th>Egaroucid Win Ratio</th></tr>
<tr><td>1</td><td>1273(Black: 629 White: 644)</td><td>47(Black: 29 White: 18)</td><td>680(Black: 342 White: 338)</td><td>0.648</td></tr>
<tr><td>5</td><td>1335(Black: 672 White: 663)</td><td>100(Black: 55 White: 45)</td><td>565(Black: 273 White: 292)</td><td>0.693</td></tr>
<tr><td>10</td><td>1064(Black: 610 White: 454)</td><td>226(Black: 108 White: 118)</td><td>710(Black: 282 White: 428)</td><td>0.589</td></tr>
<tr><td>15</td><td>245(Black: 130 White: 115)</td><td>104(Black: 50 White: 54)</td><td>151(Black: 70 White: 81)</td><td>0.594</td></tr>
<tr><td>21</td><td>84(Black: 59 White: 25)</td><td>43(Black: 15 White: 28)</td><td>73(Black: 26 White: 47)</td><td>0.527</td></tr>
</table></div>



