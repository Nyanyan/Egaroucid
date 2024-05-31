# Egaroucid 7.1.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>SIMD</td><td>27.75</td><td>25746866156</td><td>927814996</td><td><a href="./files/0_Core_i9-13900K_SIMD.txt">0_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>Generic</td><td>50.639</td><td>28517711998</td><td>563157092</td><td><a href="./files/1_Core_i9-13900K_Generic.txt">1_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>AVX512</td><td>54.842</td><td>23263286724</td><td>424187424</td><td><a href="./files/2_Core_i9-11900K_AVX512.txt">2_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>SIMD</td><td>55.464</td><td>23143759716</td><td>417275344</td><td><a href="./files/3_Core_i9-11900K_SIMD.txt">3_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>Generic</td><td>112.36</td><td>23298639098</td><td>207357058</td><td><a href="./files/4_Core_i9-11900K_Generic.txt">4_Core_i9-11900K_Generic.txt</a></td>
</tr>
</table>
</div>





## Play against Edax 4.4

<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a> is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

I used <a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a> for its testcases.

No opening books used.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

<div class="table_wrapper"><table>
<tr><th>Level</th><th>Egaroucid win</th><th>Draw</th><th>Edax Win</th><th>Egaroucid Win Ratio</th></tr>
<tr><td>1</td><td>1232(Black: 594 White: 638)</td><td>53(Black: 30 White: 23)</td><td>715(Black: 376 White: 339)</td><td>0.629</td></tr>
<tr><td>5</td><td>1255(Black: 642 White: 613)</td><td>102(Black: 53 White: 49)</td><td>643(Black: 305 White: 338)</td><td>0.653</td></tr>
<tr><td>10</td><td>1016(Black: 565 White: 451)</td><td>234(Black: 109 White: 125)</td><td>750(Black: 326 White: 424)</td><td>0.567</td></tr>
<tr><td>15</td><td>215(Black: 113 White: 102)</td><td>111(Black: 51 White: 60)</td><td>174(Black: 86 White: 88)</td><td>0.541</td></tr>
<tr><td>21</td><td>85(Black: 57 White: 28)</td><td>50(Black: 19 White: 31)</td><td>65(Black: 24 White: 41)</td><td>0.55</td></tr>
</table></div>



