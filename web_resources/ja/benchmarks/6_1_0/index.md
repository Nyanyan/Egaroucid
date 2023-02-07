# Egaroucid 6.1.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

使用CPUはCore i9-11900Kです。

### Egaroucid for Console 6.1.0 Windows x64 SIMD

<table>
<tr>
<th>番号</th>
<th>深さ</th>
<th>最善手</th>
<th>手番の評価値</th>
<th>探索時間(秒)</th>
<th>訪問ノード数</th>
<th>NPS</th>
</tr>
<tr>
<td>#40</td>
<td>20@100%</td>
<td>a2</td>
<td>+38</td>
<td>0.13</td>
<td>34046159</td>
<td>261893530</td>
</tr>
<tr>
<td>#41</td>
<td>22@100%</td>
<td>h4</td>
<td>+0</td>
<td>0.233</td>
<td>38361379</td>
<td>164641111</td>
</tr>
<tr>
<td>#42</td>
<td>22@100%</td>
<td>g2</td>
<td>+6</td>
<td>0.324</td>
<td>69392367</td>
<td>214173972</td>
</tr>
<tr>
<td>#43</td>
<td>23@100%</td>
<td>c7</td>
<td>-12</td>
<td>0.284</td>
<td>57083578</td>
<td>200998514</td>
</tr>
<tr>
<td>#44</td>
<td>23@100%</td>
<td>b8</td>
<td>-14</td>
<td>0.198</td>
<td>21367458</td>
<td>107916454</td>
</tr>
<tr>
<td>#45</td>
<td>24@100%</td>
<td>b2</td>
<td>+6</td>
<td>1.813</td>
<td>612227091</td>
<td>337687308</td>
</tr>
<tr>
<td>#46</td>
<td>24@100%</td>
<td>b3</td>
<td>-8</td>
<td>0.516</td>
<td>104353321</td>
<td>202235118</td>
</tr>
<tr>
<td>#47</td>
<td>25@100%</td>
<td>g2</td>
<td>+4</td>
<td>0.247</td>
<td>30154429</td>
<td>122082708</td>
</tr>
<tr>
<td>#48</td>
<td>25@100%</td>
<td>f6</td>
<td>+28</td>
<td>0.896</td>
<td>178317496</td>
<td>199015062</td>
</tr>
<tr>
<td>#49</td>
<td>26@100%</td>
<td>e1</td>
<td>+16</td>
<td>1.693</td>
<td>461193080</td>
<td>272411742</td>
</tr>
<tr>
<td>#50</td>
<td>26@100%</td>
<td>d8</td>
<td>+10</td>
<td>6.572</td>
<td>2078435981</td>
<td>316256235</td>
</tr>
<tr>
<td>#51</td>
<td>27@100%</td>
<td>e2</td>
<td>+6</td>
<td>1.819</td>
<td>402752110</td>
<td>221414024</td>
</tr>
<tr>
<td>#52</td>
<td>27@100%</td>
<td>a3</td>
<td>+0</td>
<td>2.105</td>
<td>460196375</td>
<td>218620605</td>
</tr>
<tr>
<td>#53</td>
<td>28@100%</td>
<td>d8</td>
<td>-2</td>
<td>16.745</td>
<td>5428631290</td>
<td>324194164</td>
</tr>
<tr>
<td>#54</td>
<td>28@100%</td>
<td>c7</td>
<td>-2</td>
<td>20.264</td>
<td>6858449925</td>
<td>338454891</td>
</tr>
<tr>
<td>#55</td>
<td>29@100%</td>
<td>g6</td>
<td>+0</td>
<td>56.476</td>
<td>14231640162</td>
<td>251994478</td>
</tr>
<tr>
<td>#56</td>
<td>29@100%</td>
<td>h5</td>
<td>+2</td>
<td>7.647</td>
<td>1421292685</td>
<td>185862780</td>
</tr>
<tr>
<td>#57</td>
<td>30@100%</td>
<td>a6</td>
<td>-10</td>
<td>14.908</td>
<td>4174615559</td>
<td>280025191</td>
</tr>
<tr>
<td>#58</td>
<td>30@100%</td>
<td>g1</td>
<td>+4</td>
<td>8.619</td>
<td>1762391815</td>
<td>204477528</td>
</tr>
<tr>
<td>#59</td>
<td>34@100%</td>
<td>e8</td>
<td>+64</td>
<td>0.129</td>
<td>627770</td>
<td>4866434</td>
</tr>
<tr>
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>141.618</td>
<td>38425530030</td>
<td>271332246</td>
</tr>
</table>






## Edax4.4との対戦

現状世界最強とも言われるオセロAI、<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a>との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには<a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a>に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。

### Egaroucidが黒番

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






### Egaroucidが白番

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


