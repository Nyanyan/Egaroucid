# Egaroucid Technology



## Technology Explanation

I wrote [Technology Explanation](https://www.egaroucid.nyanyan.dev/ja/technology/explanation/) only in Japanese. I'm afraid but please translate by yourself.



## Benchmarks

I used 2 benchmarks for evaluating Egaroucid. The first one is [The FFO endgame test suite](http://www.radagast.se/othello/ffotest.html). This test is for the speed of endgame complete search. The second one is the matches against old versions of Egaroucid and [Edax 4.4](https://github.com/abulmo/edax-reversi/releases/tag/v4.4). To test the strength of its evaluation function, I used no book, and used [XOT](https://berg.earthlingz.de/xot/aboutxot.php?lang=en) for the starting positions.

### The FFO endgame test suite

The endgame search is evaluated by 3 features:

<ul>
    <li>Search time</li>
    <li>Number of nodes visited</li>
    <li>NPS (Nodes Per Second)</li>
</ul>

The most important feature for users is the search time. This feature is shown as the actual time (second) to solve [The FFO endgame test suite](http://www.radagast.se/othello/ffotest.html) #40 to #59. This value is good if it decreases.

To shorten the search time, we can do two things: decrease the number of nodes and increase the number of nodes visited in a unit time.

There are some graphs of results of The FFO endgame test suite on Core i9 13900K, SIMD version.

<div class="centering_box">
	<img class="pic2" src="img/ffo_time.png">
    <img class="pic2" src="img/ffo_node.png">
    <img class="pic2" src="img/ffo_nps.png">
</div>
You can see that Egaroucid 7.4.0 takes longer time than 7.3.0. This is because of Intel microcode 0x129. In 7.3.0 measurement, 0x129 was not included, but in 7.4.0, 0x129 was included.

### Battles with XOT

It is the best way to evaluate the strength of Othello AI that we have battles with some engines. The result of battles by each version of Egaroucid and [Edax 4.6](https://github.com/abulmo/edax-reversi/releases/tag/v4.6) is below.

To avoid same lines, I used [XOT](https://berg.earthlingz.de/xot/aboutxot.php?lang=en) as the beginning board. Each battle is done in level 1 (lookahead depth is 1 for the midgame, 2 for the endgame).

<div class="table_wrapper"><table>
<tr><th>Name</th><td>7.6.0</td><td>7.5.0</td><td>7.4.0</td><td>7.3.0</td><td>7.2.0</td><td>7.1.0</td><td>7.0.0</td><td>Edax4.6</td></tr><tr><th>Level 1 Winning Rate</th><td>0.5711</td><td>0.5514</td><td>0.5354</td><td>0.5289</td><td>0.5049</td><td>0.4943</td><td>0.5366</td><td>0.2774</td></tr><tr><th>Level 1 Avg. Discs Earned</th><td>+3.36</td><td>+2.10</td><td>+1.59</td><td>+1.21</td><td>+0.44</td><td>+0.08</td><td>+1.41</td><td>-10.19</td></tr><tr><th>Level 5 Winning Rate</th><td>0.6151</td><td>0.5580</td><td>0.5291</td><td>0.5154</td><td>0.5211</td><td>0.5260</td><td>0.5011</td><td>0.2340</td></tr><tr><th>Level 5 Avg. Discs Earned</th><td>+2.51</td><td>+1.63</td><td>+1.15</td><td>+0.49</td><td>+0.52</td><td>+0.92</td><td>+0.11</td><td>-7.33</td></tr>
</table></div>

The further log is available [here](./battle.txt).



### Details

There are detailed benchmarks for each version including older versions.

<div class="table_wrapper"><table>
<tr><th>Version</th><td><a href="./benchmarks/7_6_0/">7.6.0</a></td><td><a href="./benchmarks/7_5_0/">7.5.0</a></td><td><a href="./benchmarks/7_4_0/">7.4.0</a></td><td><a href="./benchmarks/7_3_0/">7.3.0</a></td><td><a href="./benchmarks/7_2_0/">7.2.0</a></td><td><a href="./benchmarks/7_1_0/">7.1.0</a></td><td><a href="./benchmarks/7_0_0/">7.0.0</a></td><td><a href="./benchmarks/6_5_0/">6.5.0</a></td><td><a href="./benchmarks/6_4_0/">6.4.0</a></td><td><a href="./benchmarks/6_3_0/">6.3.0</a></td><td><a href="./benchmarks/6_2_0/">6.2.0</a></td><td><a href="./benchmarks/6_1_0/">6.1.0</a></td><td><a href="./benchmarks/6_0_0/">6.0.0</a></td><td><a href="./benchmarks/5_10_0/">5.10.0</a></td><td><a href="./benchmarks/5_9_0/">5.9.0</a></td><td><a href="./benchmarks/5_8_0/">5.8.0</a></td><td><a href="./benchmarks/5_7_0/">5.7.0</a></td><td><a href="./benchmarks/5_5_0/">5.5.0/5.6.0</a></td><td><a href="./benchmarks/5_4_1/">5.4.1</a></td></tr><tr><th>Date</th><td>2025/04/02</td><td>2024/12/24</td><td>2024/10/03</td><td>2024/08/16</td><td>2024/06/25</td><td>2024/06/06</td><td>2024/04/17</td><td>2023/10/25</td><td>2023/09/01</td><td>2023/07/09</td><td>2023/03/15</td><td>2022/12/23</td><td>2022/10/10</td><td>2022/06/08</td><td>2022/06/07</td><td>2022/05/13</td><td>2022/03/26</td><td>2022/03/16</td><td>2022/03/02</td></tr>
</table></div>




## Free Training Data

Huge dataset played by Egaroucid for Othello AI is available. Please see [Free Training Data](./train-data) page.

