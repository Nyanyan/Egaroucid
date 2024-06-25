# Egaroucid 7.2.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>SIMD</td><td>27.364</td><td>26421069847</td><td>965541216</td><td><a href="./files/0_Core_i9-13900K_SIMD.txt">0_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>Generic</td><td>47.844</td><td>28178107090</td><td>588958011</td><td><a href="./files/1_Core_i9-13900K_Generic.txt">1_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>AVX512</td><td>54.558</td><td>22702428048</td><td>416115474</td><td><a href="./files/2_Core_i9-11900K_AVX512.txt">2_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>SIMD</td><td>54.676</td><td>22894546165</td><td>418731183</td><td><a href="./files/3_Core_i9-11900K_SIMD.txt">3_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>Generic</td><td>115.479</td><td>23069092108</td><td>199768720</td><td><a href="./files/4_Core_i9-11900K_Generic.txt">4_Core_i9-11900K_Generic.txt</a></td>
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
<tr><td>1</td><td>1236(Black: 586 White: 650)</td><td>51(Black: 33 White: 18)</td><td>713(Black: 381 White: 332)</td><td>0.631</td></tr>
<tr><td>5</td><td>1348(Black: 679 White: 669)</td><td>88(Black: 49 White: 39)</td><td>564(Black: 272 White: 292)</td><td>0.696</td></tr>
<tr><td>10</td><td>1028(Black: 563 White: 465)</td><td>233(Black: 117 White: 116)</td><td>739(Black: 320 White: 419)</td><td>0.572</td></tr>
<tr><td>15</td><td>238(Black: 134 White: 104)</td><td>102(Black: 43 White: 59)</td><td>160(Black: 73 White: 87)</td><td>0.578</td></tr>
<tr><td>21</td><td>79(Black: 46 White: 33)</td><td>60(Black: 34 White: 26)</td><td>61(Black: 20 White: 41)</td><td>0.545</td></tr>
</table></div>



