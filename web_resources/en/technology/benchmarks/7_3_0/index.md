# Egaroucid 7.3.0 Benchmarks

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
<td>Core i9-13900K</td><td>x64_SIMD</td><td>22.892</td><td>19272934563</td><td>841906978</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_x64_SIMD.txt">000_ffo40_59_Core_i9-13900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64_Generic</td><td>39.477</td><td>21898593531</td><td>554717773</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_x64_Generic.txt">001_ffo40_59_Core_i9-13900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86_Generic</td><td>92.359</td><td>19548877069</td><td>211661852</td><td><a href="./files/002_ffo40_59_Core_i9-13900K_x86_Generic.txt">002_ffo40_59_Core_i9-13900K_x86_Generic.txt</a></td>
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
<td>Core i9-13900K</td><td>x64_modern</td><td>23.328</td><td>27849601649</td><td>1193827231</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x64</td><td>27.767</td><td>27892181236</td><td>1004508274</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-13900K</td><td>x86</td><td>42.098</td><td>27862333601</td><td>661844591</td><td><a href="./files/012_ffo40_59_Core_i9-13900K_edax_x86.txt">012_ffo40_59_Core_i9-13900K_edax_x86.txt</a></td>
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
<td>Core i9-11900K</td><td>x64_AVX512</td><td>47.282</td><td>17422173608</td><td>368473702</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_x64_AVX512.txt">100_ffo40_59_Core_i9-11900K_x64_AVX512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_SIMD</td><td>48.25</td><td>17524528057</td><td>363202654</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_x64_SIMD.txt">101_ffo40_59_Core_i9-11900K_x64_SIMD.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_Generic</td><td>92.032</td><td>18384451854</td><td>199761516</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_x64_Generic.txt">102_ffo40_59_Core_i9-11900K_x64_Generic.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86_Generic</td><td>252.075</td><td>18597008331</td><td>73775695</td><td><a href="./files/103_ffo40_59_Core_i9-11900K_x86_Generic.txt">103_ffo40_59_Core_i9-11900K_x86_Generic.txt</a></td>
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
<td>Core i9-11900K</td><td>x64_avx512</td><td>43.19</td><td>26820387942</td><td>620986060</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt">110_ffo40_59_Core_i9-11900K_edax_x64_avx512.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64_modern</td><td>44.11</td><td>26929672444</td><td>610511731</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt">111_ffo40_59_Core_i9-11900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x64</td><td>56.327</td><td>26971739786</td><td>478842115</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
<tr>
<td>Core i9-11900K</td><td>x86</td><td>89.701</td><td>26583076444</td><td>296352063</td><td><a href="./files/113_ffo40_59_Core_i9-11900K_edax_x86.txt">113_ffo40_59_Core_i9-11900K_edax_x86.txt</a></td>
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
<tr><td>1</td><td>1232(Black: 597 White: 635)</td><td>58(Black: 33 White: 25)</td><td>710(Black: 370 White: 340)</td><td>0.63</td></tr>
<tr><td>5</td><td>1313(Black: 673 White: 640)</td><td>109(Black: 52 White: 57)</td><td>578(Black: 275 White: 303)</td><td>0.684</td></tr>
<tr><td>10</td><td>1013(Black: 568 White: 445)</td><td>222(Black: 113 White: 109)</td><td>765(Black: 319 White: 446)</td><td>0.562</td></tr>
<tr><td>15</td><td>228(Black: 125 White: 103)</td><td>103(Black: 52 White: 51)</td><td>169(Black: 73 White: 96)</td><td>0.559</td></tr>
<tr><td>21</td><td>78(Black: 47 White: 31)</td><td>54(Black: 25 White: 29)</td><td>68(Black: 28 White: 40)</td><td>0.525</td></tr>
</table></div>



