# Egaroucid !!VERSION!! Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are here. As a comparison, I also show [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result.

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>


### Core i9-11900K

With Core i9-11900K, you can run AVX512 edition.

Egaroucid's result and [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result are:

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>









## Play against Edax 4.4

[Edax](https://github.com/abulmo/edax-reversi/releases/tag/v4.4) 4.4 is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

The average number of discs earned indicates how many more discs Egaroucid was able to acquire than Edax on average. The higher this value, the bigger the victory over Edax.

I used [XOT](https://berg.earthlingz.de/xot/index.php) for its testcases. No opening books used.

<div class="table_wrapper">
<table>
<tr><td>TABLE</td></tr>
</table>
</div>


