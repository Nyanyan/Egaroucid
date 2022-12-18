# Egaroucid 5.9.0 Benchmarks

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a> is a common benchmark for endgame searching. Computer completely solves each testcase, and find the best move. This benchmark evaluates the exact time for searching and the speed (NPS: Nodes Per Second).

I used Core i9-11900K for testing.

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
<td>20</td>
<td>a2</td>
<td>38</td>
<td>0.146</td>
<td>18499320</td>
<td>126707671</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>0</td>
<td>0.256</td>
<td>26526311</td>
<td>103618402</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>6</td>
<td>0.35</td>
<td>40548462</td>
<td>115852748</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>c7</td>
<td>-12</td>
<td>0.574</td>
<td>82911207</td>
<td>144444611</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>d2</td>
<td>-14</td>
<td>0.301</td>
<td>15270420</td>
<td>50732292</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>6</td>
<td>2.533</td>
<td>498339937</td>
<td>196739019</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.837</td>
<td>76840991</td>
<td>91805246</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>4</td>
<td>0.435</td>
<td>37334630</td>
<td>85826735</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>28</td>
<td>1.838</td>
<td>204129724</td>
<td>111060785</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>16</td>
<td>2.496</td>
<td>256316623</td>
<td>102690954</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>10</td>
<td>7.966</td>
<td>1179213508</td>
<td>148030819</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>a3</td>
<td>6</td>
<td>2.557</td>
<td>249226421</td>
<td>97468291</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>0</td>
<td>5.363</td>
<td>560569333</td>
<td>104525327</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>30.118</td>
<td>4883501669</td>
<td>162145616</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>28.803</td>
<td>6314578504</td>
<td>219233361</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>0</td>
<td>138.86</td>
<td>23384774474</td>
<td>168405404</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>2</td>
<td>11.095</td>
<td>1140351709</td>
<td>102780685</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>13.494</td>
<td>1668140404</td>
<td>123620898</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>4</td>
<td>15.759</td>
<td>1921933011</td>
<td>121957802</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>g8</td>
<td>64</td>
<td>0.023</td>
<td>28904</td>
<td>1256695</td>
</tr>
<tr>
<td>All</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>263.804</td>
<td>42559035562</td>
<td>161328242</td>
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
<td>497</td>
<td>24</td>
<td>479</td>
<td>0.51</td>
</tr>
<tr>
<td>5</td>
<td>573</td>
<td>52</td>
<td>375</td>
<td>0.6</td>
</tr>
<tr>
<td>10</td>
<td>536</td>
<td>131</td>
<td>333</td>
<td>0.62</td>
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
<td>532</td>
<td>22</td>
<td>446</td>
<td>0.54</td>
</tr>
<tr>
<td>5</td>
<td>535</td>
<td>54</td>
<td>411</td>
<td>0.57</td>
</tr>
<tr>
<td>10</td>
<td>468</td>
<td>109</td>
<td>423</td>
<td>0.53</td>
</tr>
</table>

