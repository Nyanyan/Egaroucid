# Egaroucid Book詳説

オセロAI EgaroucidにおけるBook関連機能の使い方(作者の使い方)を解説します。あくまでも作者個人の使い方ですが、Egaroucidはここに書いてある手法でBookを使うことが想定されています。

このページは日本語のみです。

INSERT_TABLE_OF_CONTENTS_HERE

## Bookとは？

Egaroucidにおいて、Bookとは「予め高い精度で計算しておいた評価値集」のことです。

オセロAIは終盤を高速に終局まで読み切ることができますが、序盤、中盤は終局まで読み切ることができません。そういった場合には評価関数を用いて、終局前の局面の状態から終局結果を予測したものを正しいであろう結果として用います。しかし、当然ながら評価関数は完璧なものではありません。2石損くらいの悪手は判定できないこともしばしばあります。そこで、序盤から中盤にかけて、長い時間をかけて様々な局面の評価値を予め計算しておくという手法を取ります。



## 基本知識

ここではEgaroucidのBookに関する基本知識を解説します。EdaxのBookと設計思想は近いので、EdaxのBookについてよくご存知の方は軽く読む程度で結構です。

### Bookの構造

EgaroucidのBookでは、各ボードについて以下の情報が記録されています。

<div class="table_wrapper"><table>
    <tr>
    	<th>名前</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>ボード</td>
        <td>手番のプレイヤの石の配置、および相手の石の配置</td>
    </tr>
    <tr>
    	<td>評価値</td>
        <td>手番側の目線での評価値(「この盤面で最善手を打てばX石勝ちになる」という意味)</td>
    </tr>
    <tr>
    	<td>レベル</td>
        <td>評価値計算に使ったAIのレベル(先読み手数などがわかるように記録)</td>
    </tr>
    <tr>
    	<td>ライン数</td>
        <td>この局面から終局まででいくつの局面がBookに登録されているか(Book精度に関係)</td>
    </tr>
    <tr>
    	<td>リーフの評価値</td>
        <td>合法手でBookに登録されていない局面に繋がる手のうちの多分最善手の評価値(Book学習に使用)</td>
    </tr>
    <tr>
    	<td>リーフの手</td>
        <td>合法手でBookに登録されていない局面に繋がる手のうちの多分最善手</td>
    </tr>
    <tr>
    	<td>リーフのレベル</td>
        <td>リーフの評価値と手を計算したAIのレベル</td>
    </tr>
</table></div>

実際にAIがBookを使うときには、各合法手を打ってみてからBookを検索し、登録されている(相手にとっての)評価値とその手をメモしておきます。全ての合法手を展開したら、相手にとっての評価値が一番小さい手(=自分にとっての評価値が一番大きい手)を最善手として使用します。合法手のうち1つでもBookに登録された局面があれば、Bookに登録された局面だけの中から着手を選びます。また、同じ評価値の手が複数ある場合には、最善手の中でランダムに打つ手を選びます。



### Bookの精度



## Bookを学習してみる

### 標準Bookを追加学習

### 1からBookを学習

### EdaxのBookを追加学習



## Bookの体裁を整える

### Book修正

### Book削減



## Bookファイルの構造

Egaroucidのbookは独自フォーマットのバイナリファイル(リトルエンディアン)で保存されています。過去のフォーマットも含めて構造を説明します。一般ユーザにとって有益な情報ではないと思います。

### egbk3フォーマット

拡張子は<code>.egbk3</code>です。

最新のフォーマットです。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>"EGAROUCID"</td>
        <td>9</td>
        <td>固定の文字列"EGAROUCID"</td>
    </tr>
    <tr>
    	<td>Bookのバージョン</td>
        <td>1</td>
        <td>egbk3フォーマットの場合は3で固定</td>
    </tr>
    <tr>
    	<td>登録局面数</td>
        <td>4</td>
        <td>bookに登録された局面の数</td>
    </tr>
    <tr>
    	<td>局面情報</td>
        <td>25*登録局面数</td>
        <td>登録されている局面のデータ(下記参照)</td>
    </tr>
    </table></div>

登録局面ごとに、以下のデータが保存されています。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>手番の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>相手の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>評価値</td>
        <td>1</td>
        <td>その局面の評価値</td>
    </tr>
    <tr>
    	<td>レベル</td>
        <td>1</td>
        <td>局面の評価値を計算したAIのレベル</td>
    </tr>
    <tr>
    	<td>ライン数</td>
        <td>4</td>
        <td>その局面の先にいくつの局面がbookに登録されているかを示す値</td>
    </tr>
    <tr>
    	<td>リーフの評価値</td>
        <td>1</td>
        <td>bookに未登録の手のうち、一番良さそうな手の評価値</td>
    </tr>
    <tr>
    	<td>リーフの手</td>
        <td>1</td>
        <td>bookに未登録の手のうち、一番良さそうな手</td>
    </tr>
    <tr>
    	<td>リーフのレベル</td>
        <td>1</td>
        <td>リーフ計算に用いたAIのレベル</td>
    </tr>
    </table></div>

### egbk2フォーマット

拡張子は<code>.egbk2</code>です。Egaroucid 6.5.0まで使われていたものです。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>"EGAROUCID"</td>
        <td>9</td>
        <td>固定の文字列"EGAROUCID"</td>
    </tr>
    <tr>
    	<td>Bookのバージョン</td>
        <td>1</td>
        <td>egbk2フォーマットの場合は2で固定</td>
    </tr>
    <tr>
    	<td>登録局面数</td>
        <td>4</td>
        <td>bookに登録された局面の数</td>
    </tr>
    <tr>
    	<td>局面情報</td>
        <td>(22+2*リンク数)*登録局面数</td>
        <td>登録されている局面のデータ(下記参照)</td>
    </tr>
</table></div>

登録局面ごとに、以下のデータが保存されています。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>手番の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>相手の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>評価値</td>
        <td>1</td>
        <td>その局面の評価値</td>
    </tr>
    <tr>
    	<td>レベル</td>
        <td>1</td>
        <td>局面の評価値を計算したAIのレベル</td>
    </tr>
    <tr>
    	<td>リンク数</td>
        <td>4</td>
        <td>その局面の合法手のうちbookに登録されている局面の数</td>
    </tr>
    <tr>
    	<td>リーフ情報</td>
        <td>2*リンク数</td>
        <td>登録されているリンクのデータ</td>
    </tr>
</table></div>

リンクごとに、以下のデータが保存されています。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>リンクの評価値</td>
        <td>1</td>
        <td>合法手の評価値</td>
    </tr>
    <tr>
    	<td>リンクの手</td>
        <td>1</td>
        <td>登録されている合法手</td>
    </tr>
</table></div>

### egbkフォーマット

拡張子は<code>.egbk</code>です。Egaroucid 6.2.0まで使われていたものです。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>登録局面数</td>
        <td>4</td>
        <td>bookに登録された局面の数</td>
    </tr>
    <tr>
    	<td>局面情報</td>
        <td>17*登録局面数</td>
        <td>登録されている局面のデータ(下記参照)</td>
    </tr>
</table></div>

登録局面ごとに、以下のデータが保存されています。

<div class="table_wrapper"><table>
    <tr>
    	<th>項目</th>
       	<th>データ量(バイト)</th>
       	<th>内容</th>
    </tr>
    <tr>
    	<td>手番の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>相手の石の配置</td>
        <td>8</td>
        <td>64bitを使って64マスのそれぞれに手番の石があるかを格納します(MSBがa1)</td>
    </tr>
    <tr>
    	<td>評価値</td>
        <td>1</td>
        <td>その局面の評価値に64を足したもの</td>
    </tr>
</table></div>
