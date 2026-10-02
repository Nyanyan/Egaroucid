# WTHORデータの加工

この実験では、WTHORの人間棋譜を使います。入力は、すでに盤面へ展開された次のデータです。

```text
$EGAROUCID_DATA/train_data/board_data/records1/*.dat
```

`transcript_release/0002` は、この実験の入力には使いません。

## データの読み込みと分割

このディレクトリの `prepare_wthor_data.py` で、WTHORの棋譜を train・validation・test に分割します。
PythonとNumPyが必要です。次のコマンドは、リポジトリルートの `Egaroucid` ディレクトリから実行します。

```powershell
python src\tools\policy_network_human_like_ai\00_data_processing\prepare_wthor_data.py
```

既定では全棋譜を読み、seed 613でシャッフルして、棋譜数の90%・5%・5%に分けます。
validationとtestの端数は切り捨て、残りをtrainへ割り当てます。
同じ棋譜の局面は、すべて同じ分割に所属します。元のWTHORデータは変更しません。

分割結果は、このディレクトリの `output/wthor` に保存します。

- `dataset_split.json`: seed、分割比率、棋譜数、局面数、入力ファイルと局面インデックスのSHA-256。
- `dataset_split.npz`: 各分割に含める局面のインデックス。

別の場所にあるWTHORデータを使う場合は `--board-data-dir`、保存先を変える場合は `--output-dir` を指定します。
動作確認などで件数を抑える場合は `--max-position-samples` を指定します。この上限に収まる棋譜を先頭から丸ごと読み込み、分割します。

## 加工結果を使った学習

学習コマンドには、保存した分割結果を指定します。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
  --split-manifest src\tools\policy_network_human_like_ai\00_data_processing\output\wthor\dataset_split.json `
  --configs 512x4 `
  --epochs 50
```

学習側も既定でWTHORの `records1` を参照します。既存の `--wthor` オプションも引き続き使えます。
`--split-manifest` を指定すると、入力のSHA-256を照合してから保存済みの分割を読み込みます。
その場合、分割のseed・比率・件数上限は加工時の設定を使います。学習時には `--split-mode sampled` や件数上限を併用しないでください。
再評価に使えるよう、学習結果のディレクトリにも分割ファイルを保存します。

既存の学習BATなど、`--split-manifest` を省略した呼び出しでは、このディレクトリの処理で分割を作ってから学習します。
詳しい学習条件は [10_train_policy_network](../10_train_policy_network/README.md) を参照してください。

## 実装の配置

| ファイル | 役割 |
| --- | --- |
| `prepare_wthor_data.py` | WTHORデータの分割結果を保存するコマンド。 |
| `wthor_dataset.py` | 棋譜の識別、棋譜単位の分割、分割ファイルの保存・照合。 |
| `policy_data.py` | 盤面データの読み込み、128入力特徴量・合法手などの作成。学習側からも呼び出します。 |

過去の別データ用スクリプトは `legacy_selected_data` に保存しています。WTHORの加工・学習からは呼び出しません。
