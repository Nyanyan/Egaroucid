# Egaroucid 7.5.0 Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are here. As a comparison, I also show [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result.

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>21.782</td><td>15046058762</td><td>690756531</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>31.655</td><td>14998990657</td><td>473826904</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>25.266</td><td>28427839936</td><td>1125142086</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>28.875</td><td>27614464872</td><td>956345104</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
</table>
</div>




### Core i9-11900K

With Core i9-11900K, you can run AVX512 edition.

Egaroucid's result and [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result are:

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Egaroucid</td><td>AVX512</td><td>40.883</td><td>14240516785</td><td>348323674</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>41.286</td><td>13641914629</td><td>330424711</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>79.045</td><td>14938279042</td><td>188984490</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_avx512</td><td>42.235</td><td>26811007622</td><td>634805437</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>44.107</td><td>27673287721</td><td>627412604</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>55.455</td><td>26631546154</td><td>480237060</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
</table>
</div>







## Play against Edax 4.4

[Edax](https://github.com/abulmo/edax-reversi/releases/tag/v4.4) 4.4 is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

The average number of discs earned indicates how many more discs Egaroucid was able to acquire than Edax on average. The higher this value, the bigger the victory over Edax.

I used [XOT](https://berg.earthlingz.de/xot/index.php) for its testcases. No opening books used.

<div class="table_wrapper"><table>
<tr><th>Level</th><th>Avg. Discs Earned</th><th>Winning Rate</th><th>Egaroucid Win</th><th>Draw</th><th>Edax Win</th></tr>
<tr><td>1</td><td>10.5</td><td>0.674</td><td>1330(Black: 644 White: 686)</td><td>35(Black: 18 White: 17)</td><td>635(Black: 338 White: 297)</td></tr>
<tr><td>5</td><td>9.042</td><td>0.727</td><td>1399(Black: 719 White: 680)</td><td>108(Black: 52 White: 56)</td><td>493(Black: 229 White: 264)</td></tr>
<tr><td>10</td><td>1.773</td><td>0.605</td><td>1109(Black: 616 White: 493)</td><td>203(Black: 88 White: 115)</td><td>688(Black: 296 White: 392)</td></tr>
<tr><td>15</td><td>1.072</td><td>0.575</td><td>249(Black: 129 White: 120)</td><td>77(Black: 34 White: 43)</td><td>174(Black: 87 White: 87)</td></tr>
<tr><td>21</td><td>0.43</td><td>0.55</td><td>82(Black: 48 White: 34)</td><td>56(Black: 26 White: 30)</td><td>62(Black: 26 White: 36)</td></tr>
</table></div>



