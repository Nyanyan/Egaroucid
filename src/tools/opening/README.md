# Opening

棋譜と定石名の元データから、GUIで使う盤面別の定石名を生成する。

## 日本語の元データ

`openings_japanese.txt`を編集する。1行の形式は`定石名 = F5D6C3...`で、`//`から始まる行はコメントとして扱う。従来の`data/openings_japanese.txt`は使わない。

Issue #640では、[othlogの定石名一覧](https://othlog.hasera.net/book-name/)から未登録の343局面を追加した。既存355局面の名称は維持している。掲載名との相違、棋譜の誤記の補正、対称形で同じ盤面になる項目の扱いは、[照合結果](issue_640_comparison.md)を参照。

## 再生成

リポジトリのルートから次を実行する。Pythonの標準ライブラリだけで実行できる。

```sh
cd src/tools/opening
python opening.py
```

`output/openings.txt`には盤面の定石名を、`output/openings_fork.txt`にはその盤面から最短で到達できる定石名を出力する。非合法手を含む元データはエラーにする。初期盤面の黒白配置を保つ4通りの対称形は、GUIと同じ規則で扱い、定石名の重複登録を避ける。候補名の並び順は固定される。

生成した2ファイルを、次の両方のディレクトリへコピーする。

- `bin/resources/openings/japanese/`
- `src/tools/release_script/format_files/0_common_files/openings/japanese/`

実行用と配布用の内容が一致することを確認する。定石名は起動時にリソースから読み込むため、このデータ更新にGUI本体の再ビルドは必要ない。
