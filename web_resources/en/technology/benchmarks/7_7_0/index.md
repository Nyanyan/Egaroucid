# Egaroucid 7.7.0 Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are here. As a comparison, I also show [Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)'s result. Although Edax 4.6 is the latest, but on my environment, I found that 4.5.5 is faster than 4.6.

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>20.746</td><td>14789131082</td><td>712866628</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>31.906</td><td>15735727776</td><td>493190239</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>22.798</td><td>27703082424</td><td>1215154067</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt">010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>27.468</td><td>27905804830</td><td>1015938723</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
</table>
</div>



### Core i9-11900K

With Core i9-11900K, you can run AVX512 edition.

Egaroucid's result and [Edax 4.5.5](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.5)'s result are:

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Egaroucid</td><td>AVX512</td><td>37.615</td><td>14224643683</td><td>378164128</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>39.356</td><td>14201687484</td><td>360851902</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>70.868</td><td>14512377695</td><td>204780404</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v4</td><td>35.406</td><td>26154694442</td><td>738707972</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt">110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>38.344</td><td>25950762077</td><td>676788078</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt">111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>48.314</td><td>25908072899</td><td>536243592</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
</tr>
</table>
</div>










## Play against Edax 4.6

[Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6) is one of the best open-source Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. In all conditions, Egaroucid is stronger than Edax.

The average number of discs earned indicates how many more discs Egaroucid was able to acquire than Edax on average. The higher this value, the bigger the victory over Edax.

I used [XOT](https://berg.earthlingz.de/xot/index.php) for its testcases. No opening books used.

<div class="table_wrapper"><table>
<tr><th>Level</th><th>Avg. Discs Earned</th><th>Winning Rate</th><th>Egaroucid Win</th><th>Draw</th><th>Edax Win</th></tr>
<tr><td>1</td><td>+10.92</td><td>0.726</td><td>720</td><td>12</td><td>268</td></tr>
<tr><td>5</td><td>+8.98</td><td>0.794</td><td>784</td><td>21</td><td>195</td></tr>
<tr><td>10</td><td>+2.16</td><td>0.671</td><td>637</td><td>69</td><td>294</td></tr>
<tr><td>15</td><td>+0.89</td><td>0.616</td><td>138</td><td>32</td><td>80</td></tr>
<tr><td>21</td><td>+0.24</td><td>0.565</td><td>46</td><td>21</td><td>33</td></tr>
</table></div>



