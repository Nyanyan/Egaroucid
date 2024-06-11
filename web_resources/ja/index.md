# オセロAI Egaroucid

<div style="text-align:center">世界最強レベルAI搭載 オセロ研究支援アプリ</div>





Egaroucidは世界最強クラスのオセロAIを搭載した無料のオセロ研究・対戦用アプリです。発音は [ɪɡɑɻˈəʊsid] (えがろーしっ＼ど　えがろ＼うしっど) を想定しています。

作者はオセロAIのコンテスト"[CodinGame Othello](https://www.codingame.com/multiplayer/bot-programming/othello-1)"で世界1位です(2023年9月4日現在)


<div class="download_button_container"><a class="download_button_a" href="./download/">
<div class="download_button"><img height="22pt" src="img/download.png" alt="ダウンロードアイコン">ダウンロード</div>
</a></div>


## ラインナップ

全てフリーソフトとして公開しています。EgaroucidとEgaroucid for Consoleは同一の強い思考エンジンですが、Egaroucid for Webは簡易的な思考エンジンを搭載しています。

<div class="table_wrapper"><table>
    <tr>
        <th>名称</th>
        <th>動作環境</th>
    </tr>
    <tr>
        <td>[Egaroucid](./download/)</td>
        <td>Windows</td>
    </tr>
    <tr>
        <td>[Egaroucid for Console](./console/)</td>
        <td>Windows MacOS Linux</td>
    </tr>
    <tr>
        <td>[Egaroucid for Web](./web/)</td>
        <td>任意のWebブラウザ</td>
    </tr>
    </table></div>



<div class="centering_box">
	<img class="pic2" src="img/egaroucid.png" alt="Egaroucid">
    <img class="pic2" src="img/egaroucid_for_console.png" alt="Egaroucid for Console">
    <img class="pic2" src="img/egaroucid_for_web.png" alt="Egaroucid for Web">
</div>



## 特徴

<ul>
    <li>正確で高速な独自の評価関数</li>
    <li>高速な探索</li>
    <li>AIとの対局
        <ul>
            <li>独自GUIでの対局</li>
            <li>Go Text Protocol (GTP)対応GUIを用いた対局(Egaroucid for Console)
				<ul>
                    <li>GoGui</li>
                    <li>Quarry</li>
                </ul>
            </li>
        </ul>
    </li>
    <li>対局解析</li>
    <li>評価値・うみがめ数の表示</li>
    <li>各種入出力</li>
    <li>bookを自動/手動で作成/修正</li>
    <li>Egaroucid/Edax形式bookの追加・統合</li>
    <li>定石名の表示</li>
    <li>[Siv3D](https://siv3d.github.io/)による綺麗なGUI</li>
</ul>


## 導入する

### Egaroucid

[ダウンロードページ](./download/)より自分の環境に合ったものをダウンロードし、インストーラを実行してください。

### Egaroucid for Console

[コンソール版導入ページ](./console/)の解説に従ってダウンロードまたはビルドをしてください。

### Egaroucid for Web

ダウンロードやインストールの必要はありません。[Web版ページ](./web/)で今すぐ遊べます。



## 公式ドキュメント

Egaroucid公式として公開している色々な資料です。日本語のみで書いてあります。

<ul>
    <li>[使い方](./usage/): ソフトの使い方を機能一つ一つ解説しています</li>
    <li>[Book詳説](./usage/book/): Bookの使い方を詳しく解説しています</li>
    <li>[技術解説](./technology/explanation/): オセロAI制作の役に立ちそうな専門的な話を解説しています</li>
</ul>


## 紹介記事

Egaroucidを紹介していただいた記事です

<ul>
    <li>[世界最強の「オセロAI」を作った大学生　創作の原点は「1冊の絵本」](https://web.quizknock.com/othello-ai): QuizKnock Webメディアで丁寧に紹介していただきました</li>
    <li>[Egaroucid: An Othello app review](https://www.worldothello.org/news/354/egaroucid-an-othello-app-review): 世界オセロ連盟のニュースで紹介していただきました</li>
</ul>



## その他のドキュメント

Egaroucidに関連するその他の資料です。読み物が多めです。

<ul>
    <li>[オセロAI世界1位になってオセロAIをｶﾝｾﾞﾝﾆﾘｶｲｼﾀ話](https://qiita.com/Nyanyan_Cube/items/195bdc47bb1d7c6f8b24): オセロAI制作の初期にコンテストで1位になった話です</li>
    <li>[みんな、とにかくオセロAIを作るんだ](https://qiita.com/Nyanyan_Cube/items/1839732d7bdf74caff21): オセロAIを作ると何が良いのかを解説しています</li>
    <li>[オセロAIの教科書](https://note.com/nyanyan_cubetech/m/m54104c8d2f12): オセロAI制作を初歩から解説したマガジンです。全編無料です。サンプルコードもあります。ちょっと古い記事なので、近々更新したいです…</li>
	<li>[オセロAI世界1位が最弱オセロAIを作った話(読み物編)](https://note.com/nyanyan_cubetech/n/n2674dd6f5973): 負けオセロAI [Egaroucen](https://www.egaroucen.nyanyan.dev/)(自分調べで世界最弱)制作の物語です</li>
    <li>[オセロAI世界1位が最弱オセロAIを作った話(技術編)](https://qiita.com/Nyanyan_Cube/items/43d8627539c73459635c): 負けオセロAI [Egaroucen](https://www.egaroucen.nyanyan.dev/)(自分調べで世界最弱)の技術解説です</li>
    <li>[世界1位のオセロAIをSiv3Dで人生初アプリ化した話](https://qiita.com/Nyanyan_Cube/items/bfb2bc3ba7a93c83f2d1): オセロAIを実際に画面上で動くソフトウェアとして公開した話です</li>
</ul>
### Othello is Solvedに関する資料

2023年10月30日付で、[Othello is Solved](https://doi.org/10.48550/arXiv.2310.19387)という論文がarXivに投稿されました。私はこの論文の著者ではありませんが、専門ど真ん中なこともあり、この論文を読んで勝手に解説を書きました。Egaroucidに直接関係するものではありませんが、論文の読解にはEgaroucid開発の経験が役立ちましたのでここで紹介します。おそらく日本語でアクセスできる解説の中では一番正確かつ詳しいと思います。

<ul>
    <li>[Othello is Solved (Hiroki Takizawa)](https://doi.org/10.48550/arXiv.2310.19387): 原文(英語)です。</li>
    <li>[「オセロが解けた」を白黒ハッキリさせようじゃないか](https://note.com/ipsj/n/n86f6dbfbfc7a): 情報処理学会の学会誌に特別解説として執筆した記事です。論文解説として短くまとめました。</li>
    <li>[Othello is Solved 論文解説 (私見)](https://qiita.com/Nyanyan_Cube/items/a373da3157cdd117afcc): 1万7千字という長大な文章ですが、論文について詳しく解説しています。</li>
</ul>




## バグ報告・新機能の提案などを募集しています

[こちらのGoogleフォーム](https://docs.google.com/forms/d/e/1FAIpQLSd6ML1T1fc707luPEefBXuImMnlM9cQP8j-YHKiSyFoS-8rmQ/viewform)より意見を受け付けています。

<ul>
    <li>不具合を発見した</li>
    <li>Egaroucidの翻訳をしたい</li>
    <li>こんな機能が欲しい</li>
    <li>こうしたらもっと良くなりそう</li>
    <li>ユーザテストに参加したい</li>
</ul>

など、様々な意見をお待ちしております。



## OSSへの貢献を歓迎します

EgaroucidはGPL-3.0ライセンスの下で作られたオープンソースソフトウェアです。[GitHub](https://github.com/Nyanyan/Egaroucid)にて全てのコードを公開しています。プルリクエストなどによる貢献を歓迎しています。



## Egaroucidを使用したアプリを作る場合

GPL-3.0ライセンスの下で自由に使っていただいて構いません。GPLの感染が気になる場合にはご相談ください。



利用した場合には(義務ではありませんが)利用報告をいただけると喜びます。



## 謝辞

開発に多大なる貢献をしていただいた方々に感謝します(順不同、敬称略)。

<ul>
    <li>UIデザイン
        <ul>
            <li>金子映像</li>
        </ul>
    </li>
    <li>技術提供
        <ul>
            <li>奥原俊彦</li>
        </ul>
    </li>
    <li>技術的アドバイス
        <ul>
            <li>Jon Marc Hornstein</li>
        </ul>
    </li>
    <li>定石名提供
        <ul>
            <li>うえのん</li>
            <li>Matthias Berg</li>
        </ul>
    </li>
    <li>Book提供
        <ul>
            <li>Gunnar Andersson</li>
        </ul>
    </li>
    <li>翻訳
        <ul>
            <li>ZhangHengge (中国語)</li>
        </ul>
    </li>
    <li>ユーザテスト
        <ul>
            <li>わんりゅー</li>
            <li>大賀菜央</li>
            <li>倉橋哲史</li>
            <li>出本大起</li>
            <li>まてぃか</li>
            <li>Nettle蕁麻</li>
            <li>okojoMK</li>
            <li>高田征吾</li>
            <li>まだらぬこ</li>
            <li>長野泰志</li>
            <li>trineutron</li>
            <li>クルトン</li>
        </ul>
    </li>
</ul>




## Egaroucid関連リンク

<ul>
    <li>[Egaroucid GitHub レポジトリ](https://github.com/Nyanyan/Egaroucid)</li>
	<li>[自作最弱オセロAI](https://www.egaroucen.nyanyan.dev/)</li>
    <li>[作者Webサイト](https://nyanyan.dev/ja/)</li>
    <li>[作者Twitter](https://twitter.com/takuto_yamana)</li>
</ul>



## 他のオセロAI

近年のオセロAIを紹介します。

<ul>
    <li>[Edax 4.4](https://github.com/abulmo/edax-reversi) 強豪かつ有名なオセロAIです。EgaroucidはEdaxのアイデアを多く参考にしています。</li>
	<li>[Edax 4.5](https://github.com/okuhara/edax-reversi-AVX) Edax 4.4をAVXに最適化したものです。とても速く、Egaroucidでも一部の工夫を参考にしています。</li>
	<li>[Master Reversi](http://t-ishii.la.coocan.jp/hp/mr/) 強豪かつ高速なオセロAIです。</li>
	<li>[FOREST](https://lapagedolivier.fr/forest.htm) 深層学習を利用した評価関数でαβ法を行うオセロAIです。評価関数の精度が非常に良いです。</li>
    <li>[dekunobou](https://dekunobou.jj1guj.net/) 遺伝的アルゴリズムで評価関数を調整したユニークなオセロAIです。dekunobouの開発者は将棋AIの開発でも有名です。</li>
    <li>[WZebra](http://www.radagast.se/othello/download.html) 2005年頃まで開発されていたオセロAIです。Egaroucidでは作者の許可の下、WZebraのbookを元に値を修正して標準付属bookを作っています。すでに開発は止まっていますが敬意を表して紹介します。</li>
    <li>[Logistello](https://skatgame.net/mburo/log.html) 1997年に当時の世界チャンピオン村上健氏に勝利したオセロAIです。Logistelloで考案された技術はEgaroucidでも基礎的なところで使用しています。すでに開発は止まっていますが敬意を表して紹介します。</li>
</ul>




## 作者

[山名琢翔](https://nyanyan.dev/ja/)





