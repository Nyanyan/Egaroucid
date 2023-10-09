# Egaroucid 6.5.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

### Egaroucid for Console 6.5.0 Windows x64 SIMD


#### Core i9 13900K @ 32並列

<table>
<tr>
<th>番号</th>
<th>深さ</th>
<th>最善手</th>
<th>評価値</th>
<th>時間(秒)</th>
<th>ノード数</th>
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
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>34.901</td>
<td>25129728758</td>
<td>720028903</td>
</tr>
</table>





### Egaroucid for Console 6.5.0 Windows x64 Generic

SIMDによる高速化をしていないバージョンです。

#### Core i9 13900K @ 32並列

<table>
<tr>
<th>番号</th>
<th>深さ</th>
<th>最善手</th>
<th>評価値</th>
<th>時間(秒)</th>
<th>ノード数</th>
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
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>50.377</td>
<td>25582499996</td>
<td>507821029</td>
</tr>
</table>





## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a>との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには<a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a>に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

バージョン6.3.0までは引き分けを省いて(勝ち)/(勝ち+負け)で勝率を計算していましたが、一般的ではなかったので、バージョン6.4.0からは引き分けを0.5勝として(勝ち+0.5*引き分け)/(勝ち+引き分け+負け)で計算しました。

<table>
<tr>
<th>レベル</th>
<th>Egaroucid勝ち</th>
<th>引分</th>
<th>Edax勝ち</th>
<th>Egaroucid勝率</th>
</tr>
<tr>
<td>1</td>
<td>1386(黒: 686 白: 700)</td>
<td>42(黒: 14 白: 28)</td>
<td>572(黒: 300 白: 272)</td>
<td>0.704</td>
</tr>
<tr>
<td>5</td>
<td>1250(黒: 639 白: 611)</td>
<td>87(黒: 51 白: 36)</td>
<td>663(黒: 310 白: 353)</td>
<td>0.647</td>
</tr>
<tr>
<td>10</td>
<td>1041(黒: 571 白: 470)</td>
<td>237(黒: 117 白: 120)</td>
<td>722(黒: 312 白: 410)</td>
<td>0.58</td>
</tr>
<tr>
<td>15</td>
<td>473(黒: 248 白: 225)</td>
<td>192(黒: 84 白: 108)</td>
<td>335(黒: 168 白: 167)</td>
<td>0.569</td>
</tr>
<tr>
<td>21</td>
<td>81(黒: 48 白: 33)</td>
<td>60(黒: 28 白: 32)</td>
<td>59(黒: 24 白: 35)</td>
<td>0.555</td>
</tr>
</table>



