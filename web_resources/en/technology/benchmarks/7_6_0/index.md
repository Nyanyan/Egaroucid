# Egaroucid 7.6.0 Benchmarks

## The FFO endgame test suite

[The FFO endgame test suite](http://radagast.se/othello/ffotest.html) is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second). In this section, you can see the result of FFO#40-59.

### Core i9-13900K

AVX512 edition do not works with Core i9-13900K, so there are only SIMD, Generic, and x86 editions.

Egaroucid's result are here. As a comparison, I also show [Edax 4.5.2](https://github.com/okuhara/edax-reversi-AVX/releases/tag/v4.5.2)'s result. Although Edax 4.6 is the latest, but on my environment, I found that 4.5.2 is faster than 4.6.

<div class="table_wrapper">
<table>
<tr>
<th>AI</th><th>Edition</th><th>Time (sec)</th><th>Nodes</th><th>NPS</th><th>File</th>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>20.547</td><td>14035262270</td><td>683080852</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>30.346</td><td>14290308977</td><td>470912442</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64_modern</td><td>26.124</td><td>28067613584</td><td>1074399540</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt">010_ffo40_59_Core_i9-13900K_edax_x64_modern.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>x64</td><td>30.138</td><td>27759979840</td><td>921095621</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
</tr>
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









## Play against Edax 4.6

[Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6) is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. In all conditions, Egaroucid is stronger than Edax.

The average number of discs earned indicates how many more discs Egaroucid was able to acquire than Edax on average. The higher this value, the bigger the victory over Edax.

I used [XOT](https://berg.earthlingz.de/xot/index.php) for its testcases. No opening books used.

<div class="table_wrapper"><table>
<tr><th>Level</th><th>Avg. Discs Earned</th><th>Winning Rate</th><th>Egaroucid Win</th><th>Draw</th><th>Edax Win</th></tr>
<tr><td>1</td><td>+11.27</td><td>0.742</td><td>734</td><td>15</td><td>251</td></tr>
<tr><td>5</td><td>+8.37</td><td>0.793</td><td>781</td><td>25</td><td>194</td></tr>
<tr><td>10</td><td>+1.72</td><td>0.633</td><td>590</td><td>87</td><td>323</td></tr>
<tr><td>15</td><td>+0.89</td><td>0.614</td><td>138</td><td>31</td><td>81</td></tr>
<tr><td>21</td><td>+0.45</td><td>0.565</td><td>50</td><td>13</td><td>37</td></tr>
</table></div>



