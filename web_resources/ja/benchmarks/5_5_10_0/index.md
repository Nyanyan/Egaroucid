# Egaroucid 5.10.0 ベンチマーク

## The FFO endgame test suite

<a href="http://radagast.se/othello/ffotest.html" target="_blank" el=”noopener noreferrer”>The FFO endgame test suite</a>はオセロAIの終盤探索力の指標として広く使われるベンチマークです。各テストケースを完全読みし、最善手を計算します。探索時間と訪問ノード数を指標に性能を評価します。NPSはNodes Per Secondの略で、1秒あたりの訪問ノード数を表します。

使用CPUはCore i9-11900Kです。

<table>
<tr>
<td>番号</td>
<td>深さ</td>
<td>最善手</td>
<td>手番の評価値</td>
<td>探索時間(秒)</td>
<td>訪問ノード数</td>
<td>NPS</td>
</tr>
<tr>
<td>#40</td>
<td>20</td>
<td>a2</td>
<td>38</td>
<td>0.152</td>
<td>19437182</td>
<td>127876197</td>
</tr>
<tr>
<td>#41</td>
<td>22</td>
<td>h4</td>
<td>0</td>
<td>0.254</td>
<td>28051331</td>
<td>110438311</td>
</tr>
<tr>
<td>#42</td>
<td>22</td>
<td>g2</td>
<td>6</td>
<td>0.348</td>
<td>41090111</td>
<td>118075031</td>
</tr>
<tr>
<td>#43</td>
<td>23</td>
<td>c7</td>
<td>-12</td>
<td>0.545</td>
<td>86703422</td>
<td>159088847</td>
</tr>
<tr>
<td>#44</td>
<td>23</td>
<td>d2</td>
<td>-14</td>
<td>0.293</td>
<td>15931449</td>
<td>54373546</td>
</tr>
<tr>
<td>#45</td>
<td>24</td>
<td>b2</td>
<td>6</td>
<td>2.52</td>
<td>534802372</td>
<td>212223163</td>
</tr>
<tr>
<td>#46</td>
<td>24</td>
<td>b3</td>
<td>-8</td>
<td>0.924</td>
<td>103987485</td>
<td>112540568</td>
</tr>
<tr>
<td>#47</td>
<td>25</td>
<td>g2</td>
<td>4</td>
<td>0.443</td>
<td>39916702</td>
<td>90105422</td>
</tr>
<tr>
<td>#48</td>
<td>25</td>
<td>f6</td>
<td>28</td>
<td>1.615</td>
<td>167648119</td>
<td>103806884</td>
</tr>
<tr>
<td>#49</td>
<td>26</td>
<td>e1</td>
<td>16</td>
<td>2.482</td>
<td>273867075</td>
<td>110341287</td>
</tr>
<tr>
<td>#50</td>
<td>26</td>
<td>d8</td>
<td>10</td>
<td>7.292</td>
<td>1177832829</td>
<td>161523975</td>
</tr>
<tr>
<td>#51</td>
<td>27</td>
<td>a3</td>
<td>6</td>
<td>2.488</td>
<td>263723106</td>
<td>105998032</td>
</tr>
<tr>
<td>#52</td>
<td>27</td>
<td>a3</td>
<td>0</td>
<td>3.83</td>
<td>431703973</td>
<td>112716442</td>
</tr>
<tr>
<td>#53</td>
<td>28</td>
<td>d8</td>
<td>-2</td>
<td>28.493</td>
<td>5006612436</td>
<td>175713769</td>
</tr>
<tr>
<td>#54</td>
<td>28</td>
<td>c7</td>
<td>-2</td>
<td>27.352</td>
<td>6461535397</td>
<td>236236304</td>
</tr>
<tr>
<td>#55</td>
<td>29</td>
<td>g6</td>
<td>0</td>
<td>123.158</td>
<td>22750219689</td>
<td>184723848</td>
</tr>
<tr>
<td>#56</td>
<td>29</td>
<td>h5</td>
<td>2</td>
<td>10.716</td>
<td>1221714952</td>
<td>114008487</td>
</tr>
<tr>
<td>#57</td>
<td>30</td>
<td>a6</td>
<td>-10</td>
<td>13.915</td>
<td>1997564547</td>
<td>143554764</td>
</tr>
<tr>
<td>#58</td>
<td>30</td>
<td>g1</td>
<td>4</td>
<td>13.976</td>
<td>1964184672</td>
<td>140539830</td>
</tr>
<tr>
<td>#59</td>
<td>34</td>
<td>g8</td>
<td>64</td>
<td>0.038</td>
<td>13079</td>
<td>344184</td>
</tr>
<tr>
<td>全体</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>240.834</td>
<td>42586539928</td>
<td>176829434</td>
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
<td>レベル</td>
<td>Egaroucid勝ち</td>
<td>引分</td>
<td>Edax勝ち</td>
<td>Egaroucid勝率</td>
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



### Egaroucidが白番

<table>
<tr>
<td>レベル</td>
<td>Egaroucid勝ち</td>
<td>引分</td>
<td>Edax勝ち</td>
<td>Egaroucid勝率</td>
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
