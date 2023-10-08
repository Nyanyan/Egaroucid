# Egaroucid 6.5.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

### Egaroucid for Console 6.5.0 Windows x64 SIMD


#### Core i9 13900K @ 32 threads

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
<td>20</td>
<td>a2</td>
<td>+38</td>
<td>0.031</td>
<td>16810926</td>
<td>542287935</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>+0</td>
<td>0.071</td>
<td>25036144</td>
<td>352621746</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>+6</td>
<td>0.103</td>
<td>49918294</td>
<td>484643631</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>g3</td>
<td>-12</td>
<td>0.176</td>
<td>89061638</td>
<td>506032034</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>b8</td>
<td>-14</td>
<td>0.085</td>
<td>15857193</td>
<td>186555211</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>+6</td>
<td>0.43</td>
<td>371554287</td>
<td>864079737</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.196</td>
<td>80412104</td>
<td>410265836</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>+4</td>
<td>0.092</td>
<td>27932379</td>
<td>303612815</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>+28</td>
<td>0.354</td>
<td>170191048</td>
<td>480765672</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>+16</td>
<td>0.448</td>
<td>253442941</td>
<td>565720850</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>+10</td>
<td>1.668</td>
<td>1242208061</td>
<td>744729053</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>e2</td>
<td>+6</td>
<td>0.915</td>
<td>653672402</td>
<td>714396067</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>+0</td>
<td>0.893</td>
<td>552818476</td>
<td>619057643</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>3.846</td>
<td>2937361450</td>
<td>763744526</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>5.018</td>
<td>4753294738</td>
<td>947248851</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>+0</td>
<td>12.8</td>
<td>9689724152</td>
<td>757009699</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>+2</td>
<td>1.665</td>
<td>826217280</td>
<td>496226594</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>3.704</td>
<td>2685336290</td>
<td>724982799</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>+4</td>
<td>2.373</td>
<td>1340478083</td>
<td>564887519</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>e8</td>
<td>+64</td>
<td>0.466</td>
<td>5868477</td>
<td>12593298</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>35.334</td>
<td>25787196363</td>
<td>729812542</td>
</tr>
</table>



### Egaroucid for Console 6.5.0 Windows x64 Generic

Without speedup by SIMD

#### Core i9 13900K @ 32 threads

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
<td>20</td>
<td>a2</td>
<td>+38</td>
<td>0.041</td>
<td>15751806</td>
<td>384190390</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>+0</td>
<td>0.086</td>
<td>24013659</td>
<td>279228593</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>+6</td>
<td>0.153</td>
<td>55601281</td>
<td>363407065</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>c7</td>
<td>-12</td>
<td>0.145</td>
<td>42769423</td>
<td>294961537</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>b8</td>
<td>-14</td>
<td>0.104</td>
<td>16173485</td>
<td>155514278</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>+6</td>
<td>0.606</td>
<td>356613153</td>
<td>588470549</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.269</td>
<td>93592479</td>
<td>347927431</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>+4</td>
<td>0.111</td>
<td>26024586</td>
<td>234455729</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>+28</td>
<td>0.446</td>
<td>154644228</td>
<td>346735937</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>+16</td>
<td>0.61</td>
<td>267500487</td>
<td>438525388</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>+10</td>
<td>2.216</td>
<td>1019525524</td>
<td>460074694</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>e2</td>
<td>+6</td>
<td>1.228</td>
<td>546410230</td>
<td>444959470</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>+0</td>
<td>0.983</td>
<td>436952174</td>
<td>444508824</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>4.889</td>
<td>2663778933</td>
<td>544851489</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>6.967</td>
<td>4289178140</td>
<td>615642046</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>+0</td>
<td>19.88</td>
<td>10090792924</td>
<td>507585157</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>+2</td>
<td>2.417</td>
<td>901480253</td>
<td>372974866</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>5.77</td>
<td>3005355685</td>
<td>520858870</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>+4</td>
<td>3.468</td>
<td>1439739018</td>
<td>415149659</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>e8</td>
<td>+64</td>
<td>0.435</td>
<td>5903260</td>
<td>13570712</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>50.824</td>
<td>25451800728</td>
<td>500783109</td>
</tr>
</table>




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
<td>521(Black: 268 White: 253)</td>
<td>172(Black: 79 White: 93)</td>
<td>307(Black: 153 White: 154)</td>
<td>0.607</td>
</tr>
<tr>
<td>21</td>
<td>92(Black: 54 White: 38)</td>
<td>48(Black: 23 White: 25)</td>
<td>60(Black: 23 White: 37)</td>
<td>0.58</td>
</tr>
</table>



