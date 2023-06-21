# Egaroucid 6.3.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

I used Core i9-11900K for testing.

### Egaroucid for Console 6.3.0 Windows x64 SIMD

<table>
<tr>
<th>No.</th>
<th>Depth</th>
<th>Best Move</th>
<th>Score</th>
<th>Time (sec)</th>
<th>Nodes</th>
<th>NPS</th>
</tr>
<tr>
<td>#40</td>
<td>20@100%</td>
<td>a2</td>
<td>+38</td>
<td>0.048</td>
<td>16637389</td>
<td>346612270</td>
</tr>
<tr>
<td>#41</td>
<td>22@100%</td>
<td>h4</td>
<td>+0</td>
<td>0.079</td>
<td>21637011</td>
<td>273886215</td>
</tr>
<tr>
<td>#42</td>
<td>22@100%</td>
<td>g2</td>
<td>+6</td>
<td>0.151</td>
<td>49612800</td>
<td>328561589</td>
</tr>
<tr>
<td>#43</td>
<td>23@100%</td>
<td>g3</td>
<td>-12</td>
<td>0.299</td>
<td>87839072</td>
<td>293776160</td>
</tr>
<tr>
<td>#44</td>
<td>23@100%</td>
<td>b8</td>
<td>-14</td>
<td>0.105</td>
<td>24432291</td>
<td>232688485</td>
</tr>
<tr>
<td>#45</td>
<td>24@100%</td>
<td>b2</td>
<td>+6</td>
<td>0.919</td>
<td>372236448</td>
<td>405045101</td>
</tr>
<tr>
<td>#46</td>
<td>24@100%</td>
<td>b3</td>
<td>-8</td>
<td>0.245</td>
<td>74176130</td>
<td>302759714</td>
</tr>
<tr>
<td>#47</td>
<td>25@100%</td>
<td>g2</td>
<td>+4</td>
<td>0.105</td>
<td>19367069</td>
<td>184448276</td>
</tr>
<tr>
<td>#48</td>
<td>25@100%</td>
<td>f6</td>
<td>+28</td>
<td>0.724</td>
<td>163941559</td>
<td>226438617</td>
</tr>
<tr>
<td>#49</td>
<td>26@100%</td>
<td>e1</td>
<td>+16</td>
<td>0.689</td>
<td>200574488</td>
<td>291109561</td>
</tr>
<tr>
<td>#50</td>
<td>26@100%</td>
<td>d8</td>
<td>+10</td>
<td>4.215</td>
<td>1170054368</td>
<td>277592969</td>
</tr>
<tr>
<td>#51</td>
<td>27@100%</td>
<td>e2</td>
<td>+6</td>
<td>1.588</td>
<td>504147024</td>
<td>317472937</td>
</tr>
<tr>
<td>#52</td>
<td>27@100%</td>
<td>a3</td>
<td>+0</td>
<td>1.462</td>
<td>429207813</td>
<td>293575795</td>
</tr>
<tr>
<td>#53</td>
<td>28@100%</td>
<td>d8</td>
<td>-2</td>
<td>8.03</td>
<td>2920352985</td>
<td>363680321</td>
</tr>
<tr>
<td>#54</td>
<td>28@100%</td>
<td>c7</td>
<td>-2</td>
<td>11.666</td>
<td>4043145851</td>
<td>346575162</td>
</tr>
<tr>
<td>#55</td>
<td>29@100%</td>
<td>g6</td>
<td>+0</td>
<td>30.354</td>
<td>9199943295</td>
<td>303088334</td>
</tr>
<tr>
<td>#56</td>
<td>29@100%</td>
<td>h5</td>
<td>+2</td>
<td>3.985</td>
<td>785513108</td>
<td>197117467</td>
</tr>
<tr>
<td>#57</td>
<td>30@100%</td>
<td>a6</td>
<td>-10</td>
<td>6.331</td>
<td>1627571859</td>
<td>257079743</td>
</tr>
<tr>
<td>#58</td>
<td>30@100%</td>
<td>g1</td>
<td>+4</td>
<td>5.156</td>
<td>1199125027</td>
<td>232568857</td>
</tr>
<tr>
<td>#59</td>
<td>34@100%</td>
<td>e8</td>
<td>+64</td>
<td>0.305</td>
<td>7024089</td>
<td>23029800</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>76.456</td>
<td>22916539676</td>
<td>299735007</td>
</tr>
</table>

In this testcase, Egaroucid is faster than Edax 4.4







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
<td>1227(Black: 591 White: 636)</td>
<td>46(Black: 20 White: 26)</td>
<td>727(Black: 389 White: 338)</td>
<td>0.628</td>
</tr>
<tr>
<td>5</td>
<td>1154(Black: 593 White: 561)</td>
<td>87(Black: 45 White: 42)</td>
<td>759(Black: 362 White: 397)</td>
<td>0.603</td>
</tr>
<tr>
<td>10</td>
<td>1050(Black: 599 White: 451)</td>
<td>237(Black: 107 White: 130)</td>
<td>713(Black: 294 White: 419)</td>
<td>0.596</td>
</tr>
<tr>
<td>15</td>
<td>302(Black: 162 White: 140)</td>
<td>114(Black: 51 White: 63)</td>
<td>184(Black: 87 White: 97)</td>
<td>0.621</td>
</tr>
</table>



