# 人間棋譜データの加工

raw_dataから学習用データを作る処理を、次の4段階に分けています。
データセット名は `--dataset`、データ全体の保存先は環境変数 `POLICY_DATA_ROOT` で指定します。
保存先を実行ごとに変える場合は `--data-root` でも指定できます。

| 番号 | スクリプト | 入力 → 出力 |
| --- | --- | --- |
| 01 | `01_convert_raw_data.py` | `raw_data/<dataset>` → `transcript/<dataset>` |
| 02 | `02_split_transcripts.py` | `transcript/<dataset>` → `transcript_split/<dataset>` |
| 03 | `03_generate_board_data.py` | `transcript_split/<dataset>` → `board_split/<dataset>` |
| 03a（解析） | `03_analyze_transcripts.py` | `transcript_split/<dataset>` → `cache/<dataset>_LOSS_WEIGHTED`（解析・集計・グラフ） |
| 03b（重み付け） | `03_generate_weighted_board_data.py` | `cache/<cache-name>`＋元の棋譜 → `board_split/<cache-name>` |
| 04 | `04_augment_board_data.py` | `board_split/<dataset>` → `board_split_increased/<dataset>` |

PythonとNumPyが必要です。各スクリプトは独立して実行でき、02以降は入力形式やデータセット名に依存しません。
03aのグラフ作成にはMatplotlibも必要です。

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

### 03a: 全棋譜の解析と石損分布の集計

重み付きデータを作る場合は、通常の03の代わりに03a・03bを順番に実行します。既存の分割済み棋譜を入力にするので、01・02の再実行は不要です。
まず03aで解析を完了させ、キャッシュと石損分布を保存します。この段階ではボードデータを生成しません。

```powershell
$env:POLICY_DATA_ROOT = Join-Path $env:EGAROUCID_DATA '20261002_policy_network_human_like_data'
$dataset = 'WTHOR_1977_2024'
$scripts = 'src\tools\policy_network_human_like_ai\00_data_processing'
$env:EGAROUCID_EXE = Join-Path (Get-Location) 'bin\versions\Egaroucid_for_Console_7_8_1_Windows_SIMD\Egaroucid_for_Console_7_8_1_SIMD.exe'
python "$scripts\03_analyze_transcripts.py" --dataset "$dataset"
```

保存先は `POLICY_DATA_ROOT`、エンジンは `EGAROUCID_EXE` から読みます。エンジンの環境変数を省略すると、上記のリポジトリ内の実行ファイルを使います。環境固有の絶対パスをコードへ書く必要はありません。

解析にはEgaroucidの `analyze` を使い、**level 10・1スレッドのプロセスを既定で32個**常駐させます。空いたプロセスへ次の棋譜を割り当てます。同一棋譜の重複分は解析を共有しますが、統計とデータ出力では元の棋譜IDごとに数えます。`--workers` でプロセス数を変更できます。

石損は `max(PlayedのScore, AlternativeのScore) - PlayedのScore` とします。Alternativeがない手の石損は0です。両方とも着手側から見た評価値なので、白番でも符号を反転しません。Judge欄の分類は使いません。

次のファイルを `$env:POLICY_DATA_ROOT/cache/WTHOR_1977_2024_LOSS_WEIGHTED` 内に保存します。

| ファイル | 内容 |
| --- | --- |
| `analyze.sqlite3` | 全手のanalyze出力、棋譜、解析時間、完了日時、エンジン条件 |
| `statistics/loss_distribution.csv` | 全体・train・val・test別に、0〜128石損の件数と割合（％） |
| `statistics/loss_distribution.json` | 同じ分布、分母の局面数、入力元、使用したキャッシュ条件 |
| `statistics/loss_distribution.png` | 全体の割合の棒グラフと、各分割の割合を比較する対数グラフ |
| `statistics/rejected_games.jsonl` | 不正・未完了棋譜のIDと除外理由 |

割合の分母は、それぞれの組の有効な全局面数です。0石損も含めるため、局面が存在する組では各石損の割合の合計が100％になります。不正着手や終局前に切れた棋譜は、既存の03と同じく局全体を除外します。

キャッシュは1局の解析が完了するたびに確定します。中断後は同じコマンドで再開でき、保存済みの局は再解析しません。`analyses.raw_zlib` には、両候補の着手・評価値・深さ・確率・集計を含むanalyze出力をUTF-8＋zlib形式で保存します。前のミス抽出スクリプトで作ったキャッシュも再利用できます。
旧名の `cache/WTHOR_1977_2024_MISS` がある場合は、キャッシュを使用する処理を終了してから、フォルダ全体を `WTHOR_1977_2024_LOSS_WEIGHTED` に改名してください。解析結果の形式は変わらないため、再解析は不要です。

エンジン本体・評価ファイル・ハッシュ用ファイルのSHA-256と探索設定をキャッシュの識別に使います。これらが変わると別の条件で解析し、以前の結果もDBに残します。既定の `--hash-level 25` は同梱の `resources/hash/hash25.eghs` を使います。別の値を指定する場合は、対応するハッシュ用ファイルが必要です。

各プロセスの内部の探索キャッシュは、対局をまたいで保持します。同じ局面や探索中の部分木が現れると、エンジンが深さ・探索精度などの条件を確認して、保存済みの評価値や上下限を再利用します。盤面と棋譜だけを初期化し、`clearcache` は送信しません。長時間動かしても新しい局面を登録できるよう、エンジン標準の置換・世代管理は有効にしてあります。

内部の探索キャッシュは32プロセスそれぞれが持ち、プロセス終了時に失われます。対局単位の解析結果は引き続きSQLiteへ保存します。以前の「対局ごとに内部キャッシュを消去する」設定で保存した結果も、他の条件がすべて同じなら再利用し、元の解析条件をDBに残します。進捗表示の `cache` はSQLiteの対局単位のヒット数です。

進捗は既定で5秒ごとに件数・経過時間・ETAを表示します。解析のETAは未キャッシュの棋譜を対象に計算し、最初の1局が終わるまでは `--:--:--` と表示します。`--progress-seconds` で表示間隔、`--timeout-sec` で1コマンドの制限時間（既定600秒）を変更できます。
同じログに、石損 `1-2`、`3-4`、`5-6`、`7+` の割合を表示します。各区間は両端を含み、互いに重複しません。分母の `positions` は、0石損とキャッシュ再利用分を含む完了済みの全局面数です。同一棋譜の重複分も元データの件数どおり数えます。再開時は先にキャッシュを集計し、新しい解析が終わるたびに加算します。まだ集計対象がない場合は割合を `--` と表示し、最終集計では各分割の割合も表示します。
解析・集計が完了するとstatisticsを配置します。再実行では以前のstatisticsを日時付きで退避して更新します。SQLiteキャッシュはそのまま保持します。

### 03b: キャッシュから全局面の重み付きデータを生成

解析が完了したら、次を実行します。03bはキャッシュを読み取るだけなので、Egaroucidの実行ファイルは不要です。

```powershell
$cacheName = 'WTHOR_1977_2024_LOSS_WEIGHTED'
python "$scripts\03_generate_weighted_board_data.py" --cache-name "$cacheName"
```

`--cache-name` には `cache` 直下のフォルダ名をそのまま渡します。元の棋譜名は `statistics/loss_distribution.json` の `dataset` から取得します。出力先にもキャッシュ名をそのまま使い、接尾辞を追加しません。03bの `--dataset` は `--cache-name` の別名で、こちらもキャッシュフォルダ名を指定します。

各局面の重みを **`2 ** (石損 / 2)`** とします。0石損は1倍、2石損は2倍、4石損は4倍です。奇数の石損にも同じ式を適用し、1石損は約1.414倍になります。上限で切り詰める処理は行いません。
石損で局面を除外せず、有効な棋譜の全局面を1件ずつ保存します。元のtrain・val・testと棋譜ID、実際の着手、終局得点を保ちます。

出力先は `$env:POLICY_DATA_ROOT/board_split/WTHOR_1977_2024_LOSS_WEIGHTED` です。
各組に従来の19バイト形式の `.dat` と `.game_ids` を保存し、同じファイル名の `.weights` を追加します。
`.weights` は局面と同じ順序のfloat64リトルエンディアンで、1局面につき8バイトです。データを重みの回数だけ複製する必要がないため、大きな石損でも件数は増えません。

既定では直近のstatisticsが示す解析条件を使います。statisticsがない場合は、キャッシュ名の末尾 `_LOSS_WEIGHTED` を除いた名前を元の棋譜名とし、解析条件が1種類なら利用できます。
条件を明示する場合は、JSONの `configuration` またはSQLiteの `configurations.id` を `--configuration` に渡します。
必要な棋譜のキャッシュが不足している場合は停止し、03aの実行を案内します。自動で追加解析は行いません。

既定では1ファイルに最大1,000,000局面を保存します。`--positions-per-file` で変更できます。
出力は全件の変換が終わってから配置します。完成済みの同名出力を作り直す場合は `--replace` を付けると、以前の出力を日時付きで退避します。

8倍化する場合は、04へ重み付きデータセット名を渡します。回転・鏡映で作る8局面それぞれに、元の重みを付けます。

```powershell
python "$scripts\04_augment_board_data.py" --dataset "$cacheName"
```

学習側も `.weights` を読み、Kerasの `sample_weight` として使います。重みの比率を保つため、全train局面の平均重みで各組の重みを割ってから学習へ渡します。ファイル内の重みは変更しません。詳細は [学習側のREADME](../10_train_policy_network/README.md) を参照してください。

以前の `03_generate_miss_board_data.py --min-loss N` も残しています。こちらはN石損以上の手だけを抽出する用途で、全局面を使う場合は03a・03bを使用してください。

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
