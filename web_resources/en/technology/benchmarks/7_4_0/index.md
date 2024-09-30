# Egaroucid 7.4.0 Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are:

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_SIMD</td><td>23.965</td><td>19280200483</td><td>804514937</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_x64_SIMD.txt">000_ffo40_59_Core_i9-13900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_Generic</td><td>36.944</td><td>18881840820</td><td>511093569</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_x64_Generic.txt">001_ffo40_59_Core_i9-13900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86_Generic</td><td>92.242</td><td>19458620997</td><td>210951854</td><td><a href="./files/002_ffo40_59_Core_i9-13900K_x86_Generic.txt">002_ffo40_59_Core_i9-13900K_x86_Generic.txt</a></td>
</tr>
</table>
</div>


As a comparison, I also show [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result:

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_modern</td><td>24.908</td><td>27698822259</td><td>1112045217</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64</td><td>29.469</td><td>27561483343</td><td>935270397</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86</td><td>45.156</td><td>28511338646</td><td>631396462</td><td><a href="./files/012_ffo40_59_Core_i9-13900K_edax_x86.txt">012_ffo40_59_Core_i9-13900K_edax_x86.txt</a></td>
</tr>
</table>
</div>

### Core i9-11900K

With Core i9-11900K, you can run AVX512 edition.

Egaroucid's result are:

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_AVX512</td><td>44.232</td><td>17098579900</td><td>386565832</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_x64_AVX512.txt">100_ffo40_59_Core_i9-11900K_x64_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_SIMD</td><td>48.614</td><td>18373521423</td><td>377947122</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_x64_SIMD.txt">101_ffo40_59_Core_i9-11900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_Generic</td><td>90.339</td><td>18611218041</td><td>206015320</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_x64_Generic.txt">102_ffo40_59_Core_i9-11900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86_Generic</td><td>242.571</td><td>18802482708</td><td>77513316</td><td><a href="./files/103_ffo40_59_Core_i9-11900K_x86_Generic.txt">103_ffo40_59_Core_i9-11900K_x86_Generic.txt</a></td>
</tr>
</table>
</div>


[Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result are:

<div class="table_wrapper">
<table>
<tr>
<th>CPU</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_avx512</td><td>46.642</td><td>27072480526</td><td>580431382</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_modern</td><td>46.561</td><td>27575952822</td><td>592254308</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64</td><td>56.86</td><td>26635810350</td><td>468445486</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86</td><td>91.764</td><td>26989485812</td><td>294118454</td><td><a href="./files/113_ffo40_59_Core_i9-11900K_edax_x86.txt">113_ffo40_59_Core_i9-11900K_edax_x86.txt</a></td>
</tr>
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


