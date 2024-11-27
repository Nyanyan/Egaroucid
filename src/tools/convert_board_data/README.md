# Convert Board Data

評価関数学習時に使う

様々なフォーマットから統一されたボードデータフォーマットに変換する



## 棋譜から

* ```all_expand_transcript.py```を実行
  * ```expand_transcript.cpp```をラップしてある
  * パス時にバグがある学習データを除くため、N手までにバグがあったらデータを捨てるという処理を行えるようにした



## 開始ボード+棋譜から

* ```all_expand_transcript_with_starting_board.py```を実行
  * ```expand_transcript_with_starting_board.cpp```をラップしてある
  * 以下のような形式が改行区切りで収録されたテキストファイルを読む
  * ```------------------OOOX----XOOX---XOOOO----XXOXO----------------- X b4g5f7e7f8g3c2d7e8c8c7b3a4c1d2e2h6a3h4g7g4h5h3g8a2b8e1f1g1d8h8a5a6b7d1h1h2h7b6g2f2b1b2a1a8a7```



## ボード情報から

* ```board_data_processing.cpp```を使用
  * 適宜フォーマットをあわせる



## Bookから

* ```generate_board_from_book.cpp```を使用



## フォーマット

バイナリ形式で、以下の情報を繰り返している。特にヘッダはない。

| 項目           | データ量(バイト) | 備考                         |
| -------------- | ---------------- | ---------------------------- |
| 手番側の石配置 | 8                |                              |
| 相手側の石配置 | 8                |                              |
| 黒番/白番      | 1                | 0なら黒、1なら白             |
| 最善手         | 1                | MSBがa1、LSBがh8             |
| スコア         | 1                | その手番から見たときのスコア |

