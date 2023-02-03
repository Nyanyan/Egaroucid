# Egaroucid 5.8.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

I used Core i9-11900K for testing.

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
<td>38</td>
<td>0.208</td>
<td>30240013</td>
<td>145384677</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>0</td>
<td>0.282</td>
<td>30934637</td>
<td>109697294</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>6</td>
<td>0.323</td>
<td>38954689</td>
<td>120602752</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>c7</td>
<td>-12</td>
<td>0.764</td>
<td>113328845</td>
<td>148336184</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>d2</td>
<td>-14</td>
<td>0.4</td>
<td>24596008</td>
<td>61490020</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>6</td>
<td>2.957</td>
<td>597485227</td>
<td>202057905</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.975</td>
<td>101741641</td>
<td>104350401</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>4</td>
<td>0.568</td>
<td>55996875</td>
<td>98586047</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>28</td>
<td>3.955</td>
<td>605660062</td>
<td>153137815</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>16</td>
<td>4.756</td>
<td>798738457</td>
<td>167943325</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>10</td>
<td>13.33</td>
<td>2343428530</td>
<td>175801090</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>e2</td>
<td>6</td>
<td>5.417</td>
<td>532335784</td>
<td>98271328</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>0</td>
<td>4.371</td>
<td>490500268</td>
<td>112216945</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>45.686</td>
<td>4995871831</td>
<td>109352358</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>57.814</td>
<td>8094332723</td>
<td>140006446</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>0</td>
<td>153.868</td>
<td>19667033928</td>
<td>127817570</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>2</td>
<td>17.839</td>
<td>1737263912</td>
<td>97385722</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>40.738</td>
<td>5202620937</td>
<td>127709287</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>4</td>
<td>20.25</td>
<td>2560167511</td>
<td>126428025</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>g8</td>
<td>64</td>
<td>0.026</td>
<td>2056</td>
<td>79076</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>374.527</td>
<td>48021233934</td>
<td>128218350</td>
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
<th>Level</th>
<th>Egaroucid win</th>
<th>Draw</th>
<th>Edax Win</th>
<th>Egaroucid Win Ratio</th>
</tr>
<tr>
<td>1</td>
<td>489</td>
<td>24</td>
<td>487</td>
<td>0.5</td>
</tr>
<tr>
<td>5</td>
<td>582</td>
<td>59</td>
<td>359</td>
<td>0.62</td>
</tr>
<tr>
<td>10</td>
<td>585</td>
<td>119</td>
<td>296</td>
<td>0.66</td>
</tr>
<tr>
<td>11</td>
<td>552</td>
<td>132</td>
<td>316</td>
<td>0.64</td>
</tr>
</table>




### Egaroucid played White

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
<td>534</td>
<td>27</td>
<td>439</td>
<td>0.55</td>
</tr>
<tr>
<td>5</td>
<td>549</td>
<td>44</td>
<td>407</td>
<td>0.57</td>
</tr>
<tr>
<td>10</td>
<td>443</td>
<td>124</td>
<td>433</td>
<td>0.51</td>
</tr>
<tr>
<td>11</td>
<td>501</td>
<td>115</td>
<td>384</td>
<td>0.57</td>
</tr>
</table>
