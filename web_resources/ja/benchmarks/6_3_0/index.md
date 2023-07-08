# Egaroucid 6.3.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

### Egaroucid for Console 6.3.0 Windows x64 SIMD

本バージョンで手元の環境ではEdax 4.4よりも高速になりました。

#### Core i9 11900K

以前のリリースでも用いたCPUです。

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
<td>0.049</td>
<td>16598652</td>
<td>338748000</td>
</tr>
<tr>
<td>#41</td>
<td>22@100%</td>
<td>h4</td>
<td>+0</td>
<td>0.078</td>
<td>21747252</td>
<td>278810923</td>
</tr>
<tr>
<td>#42</td>
<td>22@100%</td>
<td>g2</td>
<td>+6</td>
<td>0.151</td>
<td>50850284</td>
<td>336756847</td>
</tr>
<tr>
<td>#43</td>
<td>23@100%</td>
<td>g3</td>
<td>-12</td>
<td>0.293</td>
<td>88445051</td>
<td>301860242</td>
</tr>
<tr>
<td>#44</td>
<td>23@100%</td>
<td>b8</td>
<td>-14</td>
<td>0.098</td>
<td>21826343</td>
<td>222717785</td>
</tr>
<tr>
<td>#45</td>
<td>24@100%</td>
<td>b2</td>
<td>+6</td>
<td>0.975</td>
<td>406336171</td>
<td>416755047</td>
</tr>
<tr>
<td>#46</td>
<td>24@100%</td>
<td>b3</td>
<td>-8</td>
<td>0.218</td>
<td>67074299</td>
<td>307680270</td>
</tr>
<tr>
<td>#47</td>
<td>25@100%</td>
<td>g2</td>
<td>+4</td>
<td>0.101</td>
<td>18726289</td>
<td>185408801</td>
</tr>
<tr>
<td>#48</td>
<td>25@100%</td>
<td>f6</td>
<td>+28</td>
<td>0.74</td>
<td>165870291</td>
<td>224149041</td>
</tr>
<tr>
<td>#49</td>
<td>26@100%</td>
<td>e1</td>
<td>+16</td>
<td>0.685</td>
<td>187065270</td>
<td>273087985</td>
</tr>
<tr>
<td>#50</td>
<td>26@100%</td>
<td>d8</td>
<td>+10</td>
<td>4.368</td>
<td>1190461993</td>
<td>272541665</td>
</tr>
<tr>
<td>#51</td>
<td>27@100%</td>
<td>e2</td>
<td>+6</td>
<td>1.544</td>
<td>492558593</td>
<td>319014632</td>
</tr>
<tr>
<td>#52</td>
<td>27@100%</td>
<td>a3</td>
<td>+0</td>
<td>1.359</td>
<td>402202878</td>
<td>295955024</td>
</tr>
<tr>
<td>#53</td>
<td>28@100%</td>
<td>d8</td>
<td>-2</td>
<td>7.906</td>
<td>2858113187</td>
<td>361511913</td>
</tr>
<tr>
<td>#54</td>
<td>28@100%</td>
<td>c7</td>
<td>-2</td>
<td>11.145</td>
<td>4031223396</td>
<td>361706899</td>
</tr>
<tr>
<td>#55</td>
<td>29@100%</td>
<td>g6</td>
<td>+0</td>
<td>30.522</td>
<td>9313861679</td>
<td>305152404</td>
</tr>
<tr>
<td>#56</td>
<td>29@100%</td>
<td>h5</td>
<td>+2</td>
<td>3.927</td>
<td>789099800</td>
<td>200942144</td>
</tr>
<tr>
<td>#57</td>
<td>30@100%</td>
<td>a6</td>
<td>-10</td>
<td>5.891</td>
<td>1561827228</td>
<td>265120901</td>
</tr>
<tr>
<td>#58</td>
<td>30@100%</td>
<td>g1</td>
<td>+4</td>
<td>4.673</td>
<td>1140596937</td>
<td>244082374</td>
</tr>
<tr>
<td>#59</td>
<td>34@100%</td>
<td>e8</td>
<td>+64</td>
<td>0.298</td>
<td>6845851</td>
<td>22972654</td>
</tr>
<tr>
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>75.021</td>
<td>22831331444</td>
<td>304332539</td>
</tr>
</table>









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
<td>302(黒: 162 白: 140)</td>
<td>114(黒: 51 白: 63)</td>
<td>184(黒: 87 白: 97)</td>
<td>0.621</td>
</tr>
</table>


