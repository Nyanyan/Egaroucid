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
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>35.334</td>
<td>25787196363</td>
<td>729812542</td>
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
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>50.824</td>
<td>25451800728</td>
<td>500783109</td>
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
<td>521(黒: 268 白: 253)</td>
<td>172(黒: 79 白: 93)</td>
<td>307(黒: 153 白: 154)</td>
<td>0.607</td>
</tr>
<tr>
<td>21</td>
<td>92(黒: 54 白: 38)</td>
<td>48(黒: 23 白: 25)</td>
<td>60(黒: 23 白: 37)</td>
<td>0.58</td>
</tr>
</table>



