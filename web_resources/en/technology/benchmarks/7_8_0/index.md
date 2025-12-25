# Egaroucid 7.8.0 Benchmarks

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
<td>Egaroucid</td><td>SIMD</td><td>20.186</td><td>14587235996</td><td>722641236</td><td><a href="./files/000_ffo40_59_Core_i9-13900K_SIMD.txt">000_ffo40_59_Core_i9-13900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>31.088</td><td>15058413062</td><td>484380245</td><td><a href="./files/001_ffo40_59_Core_i9-13900K_Generic.txt">001_ffo40_59_Core_i9-13900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>23.11</td><td>27528192590</td><td>1191180986</td><td><a href="./files/010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt">010_ffo40_59_Core_i9-13900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>28.42</td><td>27530414953</td><td>968698626</td><td><a href="./files/011_ffo40_59_Core_i9-13900K_edax_x64.txt">011_ffo40_59_Core_i9-13900K_edax_x64.txt</a></td>
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
<td>Egaroucid</td><td>AVX512</td><td>36.148</td><td>13608266119</td><td>376459724</td><td><a href="./files/100_ffo40_59_Core_i9-11900K_AVX512.txt">100_ffo40_59_Core_i9-11900K_AVX512.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>SIMD</td><td>37.717</td><td>13964135905</td><td>370234533</td><td><a href="./files/101_ffo40_59_Core_i9-11900K_SIMD.txt">101_ffo40_59_Core_i9-11900K_SIMD.txt</a></td>
</tr>
<tr>
<td>Egaroucid</td><td>Generic</td><td>67.869</td><td>14417592213</td><td>212432660</td><td><a href="./files/102_ffo40_59_Core_i9-11900K_Generic.txt">102_ffo40_59_Core_i9-11900K_Generic.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v4</td><td>34.425</td><td>26173069650</td><td>760292510</td><td><a href="./files/110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt">110_ffo40_59_Core_i9-11900K_edax_x64_v4.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>v3</td><td>36.061</td><td>25817656645</td><td>715944002</td><td><a href="./files/111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt">111_ffo40_59_Core_i9-11900K_edax_x64_v3.txt</a></td>
</tr>
<tr>
<td>Edax</td><td>-</td><td>47.94</td><td>26193986607</td><td>546391043</td><td><a href="./files/112_ffo40_59_Core_i9-11900K_edax_x64.txt">112_ffo40_59_Core_i9-11900K_edax_x64.txt</a></td>
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
<tr><td>1</td><td>+10.47</td><td>0.714</td><td>709</td><td>10</td><td>281</td></tr>
<tr><td>5</td><td>+9.89</td><td>0.834</td><td>822</td><td>24</td><td>154</td></tr>
<tr><td>10</td><td>+2.01</td><td>0.647</td><td>609</td><td>76</td><td>315</td></tr>
<tr><td>15</td><td>+1.12</td><td>0.64</td><td>146</td><td>28</td><td>76</td></tr>
<tr><td>21</td><td>+0.39</td><td>0.61</td><td>51</td><td>20</td><td>29</td></tr>
</table></div>



