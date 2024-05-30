# Egaroucid 7.1.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

### Egaroucid for Console 7.1.0 Windows x64 SIMD

This is the standard version. This version can be run on almost all CPUs. It requires AVX2, which for example Intel Core i series gen 3 or later have.

#### Core i9 13900K @ 32 threads





### Egaroucid for Console 7.1.0 Windows x64 AVX512

Optimized with AVX-512. If you have CPUs that have AVX-512 (such as Core i series gen 7-11), this may be faster than SIMD version.


#### Core i9 11900K @ 16 threads





### Egaroucid for Console 7.1.0 Windows x64 Generic

Without speedup by SIMD.

#### Core i9 13900K @ 32 threads






## Play against Edax 4.4

<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a> is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

I used <a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a> for its testcases.

No opening books used.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

<table>
<tr>
<th>Level</th>
<th>Egaroucid win</th>
<th>Draw</th>
<th>Edax Win</th>
<th>Egaroucid Win Ratio</th>
</tr>
<tr>
<td>1</td>
<td>1232(Black: 594 White: 638)</td>
<td>53(Black: 30 White: 23)</td>
<td>715(Black: 376 White: 339)</td>
<td>0.629</td>
</tr>
<tr>
<td>5</td>
<td>1255(Black: 642 White: 613)</td>
<td>102(Black: 53 White: 49)</td>
<td>643(Black: 305 White: 338)</td>
<td>0.653</td>
</tr>
<tr>
<td>10</td>
<td>1016(Black: 565 White: 451)</td>
<td>234(Black: 109 White: 125)</td>
<td>750(Black: 326 White: 424)</td>
<td>0.567</td>
</tr>
<tr>
<td>15</td>
<td>215(Black: 113 White: 102)</td>
<td>111(Black: 51 White: 60)</td>
<td>174(Black: 86 White: 88)</td>
<td>0.541</td>
</tr>
<tr>
<td>21</td>
<td>85(Black: 57 White: 28)</td>
<td>50(Black: 19 White: 31)</td>
<td>65(Black: 24 White: 41)</td>
<td>0.55</td>
</tr>
</table>



