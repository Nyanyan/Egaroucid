# Egaroucid 6.1.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

I used Core i9-11900K for testing.

### Egaroucid for Console 6.1.0 Windows x64 SIMD

<table>
<tr>
<td>No.</td>
<td>Depth</td>
<td>Best Move</td>
<td>Score</td>
<td>Time (sec)</td>
<td>Nodes</td>
<td>NPS</td>
</tr>
<tr>
<td>#40</td>
<td>20@100%</td>
<td>+38</td>
<td>a2</td>
<td>0.13</td>
<td>34046159</td>
<td>261893530</td>
</tr>
<tr>
<td>#41</td>
<td>22@100%</td>
<td>+0</td>
<td>h4</td>
<td>0.233</td>
<td>38361379</td>
<td>164641111</td>
</tr>
<tr>
<td>#42</td>
<td>22@100%</td>
<td>+6</td>
<td>g2</td>
<td>0.324</td>
<td>69392367</td>
<td>214173972</td>
</tr>
<tr>
<td>#43</td>
<td>23@100%</td>
<td>-12</td>
<td>c7</td>
<td>0.284</td>
<td>57083578</td>
<td>200998514</td>
</tr>
<tr>
<td>#44</td>
<td>23@100%</td>
<td>-14</td>
<td>b8</td>
<td>0.198</td>
<td>21367458</td>
<td>107916454</td>
</tr>
<tr>
<td>#45</td>
<td>24@100%</td>
<td>+6</td>
<td>b2</td>
<td>1.813</td>
<td>612227091</td>
<td>337687308</td>
</tr>
<tr>
<td>#46</td>
<td>24@100%</td>
<td>-8</td>
<td>b3</td>
<td>0.516</td>
<td>104353321</td>
<td>202235118</td>
</tr>
<tr>
<td>#47</td>
<td>25@100%</td>
<td>+4</td>
<td>g2</td>
<td>0.247</td>
<td>30154429</td>
<td>122082708</td>
</tr>
<tr>
<td>#48</td>
<td>25@100%</td>
<td>+28</td>
<td>f6</td>
<td>0.896</td>
<td>178317496</td>
<td>199015062</td>
</tr>
<tr>
<td>#49</td>
<td>26@100%</td>
<td>+16</td>
<td>e1</td>
<td>1.693</td>
<td>461193080</td>
<td>272411742</td>
</tr>
<tr>
<td>#50</td>
<td>26@100%</td>
<td>+10</td>
<td>d8</td>
<td>6.572</td>
<td>2078435981</td>
<td>316256235</td>
</tr>
<tr>
<td>#51</td>
<td>27@100%</td>
<td>+6</td>
<td>e2</td>
<td>1.819</td>
<td>402752110</td>
<td>221414024</td>
</tr>
<tr>
<td>#52</td>
<td>27@100%</td>
<td>+0</td>
<td>a3</td>
<td>2.105</td>
<td>460196375</td>
<td>218620605</td>
</tr>
<tr>
<td>#53</td>
<td>28@100%</td>
<td>-2</td>
<td>d8</td>
<td>16.745</td>
<td>5428631290</td>
<td>324194164</td>
</tr>
<tr>
<td>#54</td>
<td>28@100%</td>
<td>-2</td>
<td>c7</td>
<td>20.264</td>
<td>6858449925</td>
<td>338454891</td>
</tr>
<tr>
<td>#55</td>
<td>29@100%</td>
<td>+0</td>
<td>g6</td>
<td>56.476</td>
<td>14231640162</td>
<td>251994478</td>
</tr>
<tr>
<td>#56</td>
<td>29@100%</td>
<td>+2</td>
<td>h5</td>
<td>7.647</td>
<td>1421292685</td>
<td>185862780</td>
</tr>
<tr>
<td>#57</td>
<td>30@100%</td>
<td>-10</td>
<td>a6</td>
<td>14.908</td>
<td>4174615559</td>
<td>280025191</td>
</tr>
<tr>
<td>#58</td>
<td>30@100%</td>
<td>+4</td>
<td>g1</td>
<td>8.619</td>
<td>1762391815</td>
<td>204477528</td>
</tr>
<tr>
<td>#59</td>
<td>34@100%</td>
<td>+64</td>
<td>e8</td>
<td>0.129</td>
<td>627770</td>
<td>4866434</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>141.618</td>
<td>38425530030</td>
<td>271332246</td>
</tr>
</table>





## Play against Edax4.4

<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a> is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

No opening books used.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do.

I used <a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a> for its testcases.

### Egaroucid played Black

<table>
<tr>
<td>Level</td>
<td>Egaroucid Win</td>
<td>Draw</td>
<td>Edax Win</td>
<td>Egaroucid Win Ratio</td>
</tr>
<tr>
<td>1</td>
<td>643</td>
<td>21</td>
<td>336</td>
<td>0.66</td>
</tr>
<tr>
<td>5</td>
<td>609</td>
<td>51</td>
<td>340</td>
<td>0.64</td>
</tr>
<tr>
<td>10</td>
<td>587</td>
<td>112</td>
<td>301</td>
<td>0.66</td>
</tr>
<tr>
<td>15</td>
<td>63</td>
<td>11</td>
<td>26</td>
<td>0.71</td>
</tr>
</table>





### Egaroucid played White

<table>
<tr>
<td>Level</td>
<td>Egaroucid Win</td>
<td>Draw</td>
<td>Edax Win</td>
<td>Egaroucid Win Ratio</td>
</tr>
<tr>
<td>1</td>
<td>638</td>
<td>19</td>
<td>343</td>
<td>0.65</td>
</tr>
<tr>
<td>5</td>
<td>580</td>
<td>44</td>
<td>376</td>
<td>0.61</td>
</tr>
<tr>
<td>10</td>
<td>458</td>
<td>135</td>
<td>407</td>
<td>0.53</td>
</tr>
<tr>
<td>15</td>
<td>44</td>
<td>24</td>
<td>32</td>
<td>0.58</td>
</tr>
</table>



