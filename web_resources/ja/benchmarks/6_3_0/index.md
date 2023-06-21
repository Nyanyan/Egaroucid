# Egaroucid 6.2.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

使用CPUはCore i9-11900Kです。

### Egaroucid for Console 6.3.0 Windows x64 SIMD

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
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>76.456</td>
<td>22916539676</td>
<td>299735007</td>
</tr>
</table>

本バージョンで手元の環境ではEdax 4.4よりも高速になりました。






## Edax 4.4との対戦

現状世界最強とも言われるオセロAI、<a href="https://github.com/abulmo/edax-reversi" target="_blank" el=”noopener noreferrer”>Edax 4.4</a>との対戦結果です。

初手からの対戦では同じ進行ばかりになって評価関数の強さは計測できないので、初期局面から8手進めた互角に近いと言われる状態から打たせて勝敗を数えました。このとき、同じ進行に対して両者が必ず先手と後手の双方を1回ずつ持つようにしました。こうすることで、両者の強さが全く同じであれば勝率は50%となるはずです。

テストには<a href="https://berg.earthlingz.de/xot/index.php" target="_blank" el=”noopener noreferrer”>XOT</a>に収録されている局面を使用しました。

bookは双方未使用です。

Egaroucid勝率が0.5を上回っていればEgaroucidがEdaxに勝ち越しています。また、カッコ内の数字はEgaroucidが黒番/白番のときのそれぞれの値です。全ての条件でEgaroucidが勝ち越しています。

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
<td>1227(黒: 591 白: 636)</td>
<td>46(黒: 20 白: 26)</td>
<td>727(黒: 389 白: 338)</td>
<td>0.628</td>
</tr>
<tr>
<td>5</td>
<td>1154(黒: 593 白: 561)</td>
<td>87(黒: 45 白: 42)</td>
<td>759(黒: 362 白: 397)</td>
<td>0.603</td>
</tr>
<tr>
<td>10</td>
<td>1050(黒: 599 白: 451)</td>
<td>237(黒: 107 白: 130)</td>
<td>713(黒: 294 白: 419)</td>
<td>0.596</td>
</tr>
<tr>
<td>15</td>
<td>194(黒: 95 白: 99)</td>
<td>76(黒: 38 白: 38)</td>
<td>130(黒: 67 白: 63)</td>
<td>0.599</td>
</tr>
</table>


