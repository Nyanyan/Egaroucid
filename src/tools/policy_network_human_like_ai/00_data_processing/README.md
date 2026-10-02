# 人間棋譜データの加工

raw_dataから学習用データを作る処理を、次の4段階に分けています。
データセット名は `--dataset`、データ全体の保存先は環境変数 `POLICY_DATA_ROOT` で指定します。
保存先を実行ごとに変える場合は `--data-root` でも指定できます。

| 番号 | スクリプト | 入力 → 出力 |
| --- | --- | --- |
| 01 | `01_convert_raw_data.py` | `raw_data/<dataset>` → `transcript/<dataset>` |
| 02 | `02_split_transcripts.py` | `transcript/<dataset>` → `transcript_split/<dataset>` |
| 03 | `03_generate_board_data.py` | `transcript_split/<dataset>` → `board_split/<dataset>` |
| 04 | `04_augment_board_data.py` | `board_split/<dataset>` → `board_split_increased/<dataset>` |

PythonとNumPyが必要です。各スクリプトは独立して実行でき、02以降は入力形式やデータセット名に依存しません。

## 実行手順

以下は、リポジトリルートの `Egaroucid` ディレクトリで実行するPowerShellの例です。
設定済みの環境変数 `EGAROUCID_DATA` を使い、`POLICY_DATA_ROOT` に今回のデータ保存先を設定します。
保存先やデータセットを変える場合は、先頭の変数を変更します。

```powershell
$env:POLICY_DATA_ROOT = Join-Path $env:EGAROUCID_DATA '20261002_policy_network_human_like_data'
$dataset = 'WTHOR_1977_2024'
$scripts = 'src\tools\policy_network_human_like_ai\00_data_processing'
```

### 01: raw_dataを棋譜テキストに変換

```powershell
python "$scripts\01_convert_raw_data.py" --dataset "$dataset" --format wthor-csv
```

`wthor-csv` は、ヘッダーなしの棋譜1列CSVと、`transcript` 列を含むCSVの両方を読み込みます。
大文字・空白を正規化し、1行1棋譜のUTF-8テキストへ変換します。パスは棋譜に含めず、盤面再生時に判定します。
棋譜が元から1行ずつ書かれたTXTの場合は、`--format transcript-text` を使えます。

出力名は `0000000.txt`、`0000001.txt` のような7桁連番です。
既定では1ファイルに10,000局を保存します。変更する場合は `--games-per-file` を指定してください。
各棋譜に元データ内で一意のIDを付け、対応する `.game_ids` に保存します。
また、元ファイル名・行番号と変換後の場所を `games.jsonl` に記録します。

### 02: 棋譜単位で分割

```powershell
python "$scripts\02_split_transcripts.py" --dataset "$dataset"
```

既定値はseed 613、train 90%・val 5%・test 5%です。
分割した棋譜を `train`、`val`、`test` サブディレクトリの連番TXTへ保存します。
`--seed`、`--train-ratio`、`--val-ratio`、`--test-ratio` で条件を変更できます。

同一内容の棋譜が複数ある場合は、すべて同じ組へ割り当てます。重複した棋譜自体は削除しません。
比率はこの棋譜グループ数に適用するため、棋譜の行数による比率は少しずれる場合があります。
valとtestのグループ数の端数を切り捨て、残りをtrainに含めます。

### 03: 分割済み棋譜からボードデータを生成

```powershell
python "$scripts\03_generate_board_data.py" --dataset "$dataset"
```

各棋譜を初期局面から再生し、各着手の直前の盤面を、その棋譜が所属する組に保存します。
パスは合法手がない場合だけ認めます。不正着手や終局前に終わった棋譜は、棋譜全体を除外し、
棋譜ID・理由・棋譜を `rejected_games.jsonl` に記録します。元の棋譜テキストは残ります。

ボードデータはヘッダーなしの19バイト形式です。
手番側の石、相手側の石、手番の色、着手、最終得点を `<QQbbb` の順で保存します。
石配置はa1が最上位ビット、h8が最下位ビットです。最終得点は手番側から見た石差とし、終局時の空きマスを勝者へ加算します。
各局面の元の棋譜IDは、対応する `.game_ids` に保存します。

既定では `train/0000000.dat` などの各ファイルに最大1,000,000局面を保存します。
`--positions-per-file` で変更できます。棋譜がファイルをまたいでもIDは変わりません。

### 04: 回転・鏡映で8倍化

```powershell
python "$scripts\04_augment_board_data.py" --dataset "$dataset"
```

元の盤面、90度・180度・270度の回転、4通りの鏡映を作ります。
手番側と相手側の石配置、および教師ラベルの着手を同じ変換で移し、手番の色・得点・棋譜IDを保ちます。
対称な盤面で同じ結果が生じた場合も残すため、局面数は正確に8倍になります。

train・val・testをそれぞれ8倍化し、所属する組を維持します。水増し後の局面を再分割することはありません。
既定の出力上限は1ファイル1,000,000局面です。処理は分割して読み書きするため、8倍化した全データをメモリへ載せずに生成できます。

## 生成結果と再実行

各段階の `manifest.json` に、設定、入力元、件数、出力ファイルのサイズとSHA-256を記録します。
後続の段階はこれらを照合してから処理を開始します。
既存の出力が空でなければ停止します。`--replace` を指定すると、処理完了後に以前の出力を日時付きの隣接バックアップへ移し、生成結果を配置します。
raw_dataは変更しません。

2026年10月2日に指定のWTHORデータで実行した結果は次のとおりです。

| 組 | 分割した棋譜 | 有効な棋譜 | ボード局面 | 8倍化後の局面 |
| --- | ---: | ---: | ---: | ---: |
| train | 121,140 | 120,768 | 7,232,087 | 57,856,696 |
| val | 6,727 | 6,707 | 401,650 | 3,213,200 |
| test | 6,727 | 6,706 | 401,545 | 3,212,360 |
| 合計 | 134,594 | 134,181 | 8,035,282 | 64,282,256 |

413局は終局前に棋譜が終わっていたため、03で除外しました。
ボードデータを元の棋譜順に並べ直し、既存のWTHOR `records1` と全局面の石配置・色・着手・得点が一致することを確認しています。

## 学習側の指定

生成済みの分割を使う場合は、学習コマンドへ `--board-split-dir` を渡します。

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
  --board-split-dir "$env:POLICY_DATA_ROOT\board_split_increased\$dataset" `
  --configs 512x4 --epochs 50
```

水増し前のデータを使う場合は、`board_split_increased` を `board_split` に変えます。
学習側は保存済みのtrain・val・testをそのまま読み込み、棋譜IDの重複が組をまたがないことも確認します。
この指定では、旧形式用の `--wthor`、`--data-root`、`--split-manifest`、件数上限、`--split-mode sampled` を併用しません。
使用したデータの情報は学習出力の `prepared_dataset.json` に保存します。

## 新しい入力形式を追加する場合

新形式の読み込み処理を `raw_formats.py` に追加し、`RAW_FORMATS` に登録します。
読み込み処理は「正規化した棋譜文字列と元の行番号」を返します。
01が共通のTXT、棋譜ID、元ファイルとの対応、manifestへ変換するため、02〜04を変更する必要はありません。

旧データ用の選別・変換スクリプトと `prepare_wthor_data.py`、それら専用のテストは、ローカルの `legacy/` に保存しています。
`legacy/` はGitの管理対象外です。データの加工には01〜04を使ってください。
現行の学習・評価が使う共通処理とテストは、このディレクトリで引き続き管理します。
