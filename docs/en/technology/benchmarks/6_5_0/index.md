# Egaroucid 6.5.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

### Egaroucid for Console 6.5.0 Windows x64 SIMD


#### Core i9 13900K @ 32 threads

<div class="table_wrapper"><table>
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
<td>20</td>
<td>a2</td>
<td>+38</td>
<td>0.031</td>
<td>16997628</td>
<td>548310580</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>+0</td>
<td>0.071</td>
<td>24207723</td>
<td>340953845</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>+6</td>
<td>0.103</td>
<td>53386603</td>
<td>518316533</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>g3</td>
<td>-12</td>
<td>0.173</td>
<td>89368561</td>
<td>516581277</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>b8</td>
<td>-14</td>
<td>0.081</td>
<td>14737343</td>
<td>181942506</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>+6</td>
<td>0.417</td>
<td>365913891</td>
<td>877491345</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.184</td>
<td>72808508</td>
<td>395698413</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>+4</td>
<td>0.09</td>
<td>28623049</td>
<td>318033877</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>+28</td>
<td>0.357</td>
<td>177552046</td>
<td>497344666</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>+16</td>
<td>0.427</td>
<td>246001308</td>
<td>576115475</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>+10</td>
<td>1.635</td>
<td>1238528233</td>
<td>757509622</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>e2</td>
<td>+6</td>
<td>0.749</td>
<td>445087184</td>
<td>594241901</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>+0</td>
<td>0.866</td>
<td>541177162</td>
<td>624915891</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>4.012</td>
<td>3195739778</td>
<td>796545308</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>4.812</td>
<td>4406176438</td>
<td>915664263</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>+0</td>
<td>13.025</td>
<td>9772271726</td>
<td>750270382</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>+2</td>
<td>1.727</td>
<td>836776216</td>
<td>484525892</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>3.369</td>
<td>2322235752</td>
<td>689295266</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>+4</td>
<td>2.313</td>
<td>1276275073</td>
<td>551783429</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>e8</td>
<td>+64</td>
<td>0.459</td>
<td>5864536</td>
<td>12776766</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>34.901</td>
<td>25129728758</td>
<td>720028903</td>
</tr>
    </table></div>





### Egaroucid for Console 6.5.0 Windows x64 Generic

Without speedup by SIMD

#### Core i9 13900K @ 32 threads

<div class="table_wrapper"><table>
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
<td>20</td>
<td>a2</td>
<td>+38</td>
<td>0.039</td>
<td>15642677</td>
<td>401094282</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>+0</td>
<td>0.086</td>
<td>23864262</td>
<td>277491418</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>+6</td>
<td>0.146</td>
<td>54012600</td>
<td>369949315</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>c7</td>
<td>-12</td>
<td>0.145</td>
<td>43346280</td>
<td>298939862</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>b8</td>
<td>-14</td>
<td>0.102</td>
<td>16542190</td>
<td>162178333</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>+6</td>
<td>0.611</td>
<td>365766568</td>
<td>598635954</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.264</td>
<td>91211537</td>
<td>345498246</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>+4</td>
<td>0.108</td>
<td>25244982</td>
<td>233749833</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>+28</td>
<td>0.426</td>
<td>146189774</td>
<td>343168483</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>+16</td>
<td>0.602</td>
<td>260644739</td>
<td>432964682</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>+10</td>
<td>2.184</td>
<td>1029002849</td>
<td>471155150</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>e2</td>
<td>+6</td>
<td>1.198</td>
<td>541359040</td>
<td>451885676</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>+0</td>
<td>0.898</td>
<td>385656725</td>
<td>429461831</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>4.844</td>
<td>2703465246</td>
<td>558105954</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>7.622</td>
<td>4862726309</td>
<td>637985608</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>+0</td>
<td>19.685</td>
<td>10018926638</td>
<td>508962491</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>+2</td>
<td>2.359</td>
<td>875422326</td>
<td>371098908</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>5.299</td>
<td>2766188517</td>
<td>522020856</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>+4</td>
<td>3.317</td>
<td>1351288396</td>
<td>407382694</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>e8</td>
<td>+64</td>
<td>0.442</td>
<td>5998341</td>
<td>13570907</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>50.377</td>
<td>25582499996</td>
<td>507821029</td>
</tr>
    </table></div>






## Play against Edax 4.4

<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a> is one of the best Othello AI in the world.

If I set the game from the very beginning, same line appears a lot. To avoid this, I set the game from many different near-draw lines.

I used <a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a> for its testcases.

No opening books used.

If Egaroucid Win Ratio is over 0.5, then Egaroucid wins more than Edax do. "Black" and "White" means Egaroucid played black/white. In all conditions, Egaroucid is stronger than Edax.

<div class="table_wrapper"><table>
<tr>
<th>Level</th>
<th>Egaroucid win</th>
<th>Draw</th>
<th>Edax Win</th>
<th>Egaroucid Win Ratio</th>
</tr>
<tr>
<td>1</td>
<td>1386(Black: 686 White: 700)</td>
<td>42(Black: 14 White: 28)</td>
<td>572(Black: 300 White: 272)</td>
<td>0.704</td>
</tr>
<tr>
<td>5</td>
<td>1250(Black: 639 White: 611)</td>
<td>87(Black: 51 White: 36)</td>
<td>663(Black: 310 White: 353)</td>
<td>0.647</td>
</tr>
<tr>
<td>10</td>
<td>1041(Black: 571 White: 470)</td>
<td>237(Black: 117 White: 120)</td>
<td>722(Black: 312 White: 410)</td>
<td>0.58</td>
</tr>
<tr>
<td>15</td>
<td>473(Black: 248 White: 225)</td>
<td>192(Black: 84 White: 108)</td>
<td>335(Black: 168 White: 167)</td>
<td>0.569</td>
</tr>
<tr>
<td>21</td>
<td>81(Black: 48 White: 33)</td>
<td>60(Black: 28 White: 32)</td>
<td>59(Black: 24 White: 35)</td>
<td>0.555</td>
</tr>
    </table></div>


