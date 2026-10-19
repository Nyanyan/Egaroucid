# WTHORの人間の着手との一致度評価

## Othello Questの各フォルダでGPWの30プレイヤを評価する

`run_quest_gpw_human_match.bat` は、`othllo_quest_data/board_data` 直下の各フォルダを順番に評価します。対象プレイヤとモデルは、下記のGPW大会の全30プレイヤと同じです。通常NN・石損A・石損Bのモデルとエンジンは大会記録から読み込み、SHA-256を照合します。TensorFlowを使えるPythonが必要です。

このディレクトリで次のコマンドを実行すると、各フォルダからそれぞれ10万局面を無作為抽出して評価します。抽出から集計までフォルダごとに処理するため、異なるフォルダのデータは混ざりません。抽出はレコード単位の非復元抽出で、同じ盤面の別レコードが選ばれた場合は出現回数どおりに数えます。既定のseedは613です。

```powershell
.\run_quest_gpw_human_match.bat --positions 100000 --workers 32
```

2026年10月8日時点では6フォルダに延べ6,788,419局面があり、そこから各10万局面、計60万局面を選びます。`--positions` を省略しても各10万局面になります。10万局面未満のフォルダがある場合はエラーで停止します。

結果は `output/quest_gpw30_n100000_seed613` に保存します。その中の `accuracy_matrix.csv` が30プレイヤ×6フォルダの一致率（%）です。`summary.csv` には180行の結果として、一致数・評価局面数・一致率・完全一致率・95% Wilson区間を保存します。通常の一致率は盤面を保つ対称変換で等価な手も一致とし、`exact_top1_accuracy` は元の着手位置との完全一致率です。

各フォルダの予測手と探索キャッシュも、結果フォルダ内の同名サブフォルダに保存します。中断した場合は同じコマンドで再開すると、保存済みの探索結果と完了済みフォルダを再利用します。進捗はコンソールと `run_status.json`、完了したフォルダの一致率は実行中も更新する `summary.csv` で確認できます。全レベルを新たに探索するため、WTHORの既存キャッシュを使う評価よりも計算量が増えます。

探索ログには「データセット1/6」のように対象を表示します。そのパーセントは当該フォルダのエンジン探索の今回実行分で、「この探索工程の推定残り」もその工程だけの推定です。後続のNN推論・融合・一致率集計や、他のフォルダの所要時間は含みません。全体は評価を完了したフォルダ数で表示し、全体ETAは未算出と明記します。

探索の1件は「1つの盤面・手番 × 1エンジン × 1レベル」です。たとえば1000–1100フォルダの10万レコードには89,657種類の盤面・手番があり、初回の探索件数は `89,657 × 2エンジン × 7レベル = 1,255,198件` になります。同じ状態の探索結果を再利用しても、一致率の集計では元レコードの出現回数を保ちます。再開時はキャッシュ再利用分を今回の探索件数から除きます。ETAはエンジンごとの残件数と平均処理速度から求めた値の大きい方で、レベル・局面による探索時間の違いは補正していません。

入力ファイル・モデル・件数だけを確認する場合や、一つのフォルダで少数局面を試す場合は、次のコマンドを使います。`--dry-run` は推論・探索・出力を行いません。`--workers` は両エンジン合計の探索プロセス数で、最小値は2です。

```powershell
.\run_quest_gpw_human_match.bat --dry-run
.\run_quest_gpw_human_match.bat --dataset 20261005_reversi_800-1000 --positions 16 --workers 2 --output-dir output/quest_gpw30_smoke
```

`--dataset` を省略すると全フォルダが対象になります。入力の親フォルダは `--board-data-root`、保存先は `--output-dir`、Pythonは環境変数 `POLICY_PYTHON` で変更できます。対象フォルダや件数を変える際は、別の保存先を指定してください。

なお、現在のQuestデータには対局の両者の着手が含まれ、botの着手も除外されていません。また、フォルダ間で同じ対局が含まれる場合があります。そのため、結果は各フォルダに記録された着手全体との一致率として解釈してください。

## 最新のGPW大会の30プレイヤを評価する

`run_gpw_human_match.bat` を実行すると、完了済みの `40_test_strength/output/gpw30_independent_xot_50sets` に参加した全30プレイヤの一致率を計測します。実行場所は任意です。TensorFlowを使えるPythonが必要で、別のPythonを使う場合は環境変数 `POLICY_PYTHON` にその実行ファイルを指定します。

対象はEgaroucid・EdaxのL1・3・5・9・13・17・21、通常NN＋L21、石損A＋通常NN、石損B＋石損Aの各5構成、ランダム方策です。各融合のαは0.2・0.4・0.6・0.8・1.0で、先に書いた方策の比率です。石損Aは `(石損+1)^2`、石損Bは `2^(石損/2)` の学習モデルです。大会記録のプレイヤ名・融合式・モデルのSHA-256を照合してから評価します。

計測は先例を踏襲します。全体はseed 613で抽出済みの共通test 100,000局面、石損1・2・4・6・10・20以上は水増し前test全401,545局面からの該当全件です。石損条件の件数は順に114,946・94,189・52,751・32,990・13,624・2,098です。Top-1は盤面を保つ対称性で等価な手も一致とし、完全一致率と95% Wilson区間も保存します。

既存6レベルの探索結果は先例のキャッシュを読み取り、新規のL3だけを1スレッド・bookなしで追加探索します。共有プロセス数の既定値は32で、`--workers 8` などで変更できます。NNの3モデルは今回の標本で再推論します。ランダム方策も先例どおりseed 57を使い、全体標本では状態・人間の手をソートした順、石損条件では元testの順に各出現を抽選します。石損の6条件では同じ抽選結果を共有します。

結果は `output/gpw30_human_match_20261008` に保存します。`accuracy_matrix.csv` は30行×7条件の百分率、`summary.csv` は210行の一致数・母数・一致率などの詳細です。`summary.json` に設定と出典、`*_predictions.npz` に個々の予測、`verification.json` に合法手・対称性・一致数の検算結果を記録します。進捗はコンソールと `run_status.json`、標本ごとの集計は `progress.csv` で確認できます。

中断した場合は同じコマンドで再開できます。保存済みのL3探索と検算済みの標本を再利用し、条件が違う出力先への上書きは停止します。保存先は `--output-dir` で変更できます。動作確認には次の引数を使ってください。

```bat
run_gpw_human_match.bat --dry-run
run_gpw_human_match.bat --smoke-states 32 --workers 2
```

`--dry-run` は実データ・モデル・既存キャッシュの検証まで実行します。`--smoke-states` は各標本の状態数を絞り、既定では本番と別の `_smoke_32` フォルダーへ保存します。小規模確認が終わったら、引数なしで本番の全件計測を開始できます。

## 石損数＋1モデルの混合を評価する

`run_loss_plus_one_blends.bat` は、今回の石損数＋1モデルを、Egaroucidレベル21・通常学習NN・前回の指数重みNNとそれぞれ混合して評価します。
αは0・0.2・0.4・0.6・0.8・1で、0は相手単体、1は今回のモデル単体です。
先例と同じ対数確率の加重和を使います。NN同士では `P_new^α * P_existing^(1−α)` を合法手で正規化し、レベル21との混合では相手の対数確率として探索スコア÷温度（既定の参照実験では1）を使います。

評価対象は先例と同じseed 613のtest標本100,000件と、水増し前test全体から抽出した石損1・2・4・6・10・20石以上の全件です。条件は重複する累積集合です。石損は既存のレベル10解析を使います。3相手×6α×7条件の126行を出力し、α=1の重複行も残します。
Top-1は盤面を保つ対称性で等価な手を一致とし、完全一致率も併記します。評価時は全局面を等しく数えます。

BATは実行場所に依存せず、モデル3個と先例のデータ・探索キャッシュを既定パスから読み込みます。`--dry-run` は全入力とキャッシュの確認だけ、`--smoke-states 32` は状態数を絞った動作確認です。本番評価は引数なしで実行します。モデルや保存先は `--new-model`・`--normal-model`・`--exponential-model`・`--output-dir` で変更できます。
既定の結果保存先は `output/loss_plus_one_blends_20261006` です。`summary.csv` に126行の詳細、`accuracy_matrix.csv` に18行×7条件の一致率、`summary.json` に入力情報と混合式、`verification.json` に保存予測の独立検算結果を保存します。進捗はコンソールと `run_status.json`・`progress.csv` で確認できます。既存の結果がある保存先は上書きせず停止します。

## ミスした着手に絞って27モデルを評価する

`evaluate_mistake_test_models.py` は、石損が **1・2・4・6石以上**のtestデータ全件について、27モデルのTop-1着手一致率を測定します。各閾値は独立した区間ではなく、6石以上の局面が4石以上・2石以上・1石以上にも含まれる累積条件です。追加の標本抽出や対称変換による水増しは行いません。

石損の基準には、既存のEgaroucidレベル10の棋譜解析を使います。値は `max(実着手の評価値, 最善の別着手の評価値) − 実着手の評価値` です。これは同レベルの探索評価値による石損であり、全局面を終局まで完全探索した真の石損とは区別してください。

対象は通常NNとレベル21のα混合6種、Egaroucidレベル1・5・9・13・17、Edaxレベル1・5・9・13・17・21、石損重み付きNNのα混合5種、通常NNと石損重み付きNNのβ混合4種、ランダム方策の計27モデルです。αとβの定義は下の既存評価と共通です。

このディレクトリから、まず前処理、続いて評価を実行してください。以下のモデルパスと参照キャッシュは、2026年10月5日の27モデル実験に合わせています。

```powershell
python ../00_data_processing/prepare_mistake_test_data.py `
  --output-dir output/mistake_test_20261005/datasets

python evaluate_mistake_test_models.py `
  --datasets-dir output/mistake_test_20261005/datasets `
  --reference-dir ../../../../../experiment_bat/20261002/results/human `
  --normal-model ../../../../../experiment_bat/20261002/results/train/selected_model.h5 `
  --weighted-model ../10_train_policy_network/trained/prepared_conv8_bn_e10_20261003_221904_955/selected_model.h5 `
  --workers 32 --policy-batch-size 512 `
  --output-dir output/mistake_test_20261005/evaluation
```

前処理は元のtestを棋譜から再生し、盤面・着手・棋譜IDを全件照合してから、解析キャッシュの石損で抽出します。解析済みのtestがキャッシュに揃っていることが必要です。学習・検証データの解析は追加実行しません。入力は `POLICY_DATA_ROOT`、または `EGAROUCID_DATA/20261002_policy_network_human_like_data` から読み込みます。

2026年10月5日に作成した `WTHOR_1977_2024` の件数は次のとおりです。元のtestは401,545局面・6,706棋譜です。

| 石損の条件 | test局面数 | 該当する棋譜数 | 元のtestに占める割合 |
| --- | ---: | ---: | ---: |
| 1石以上 | 114,946 | 6,706 | 28.6259% |
| 2石以上 | 94,189 | 6,706 | 23.4566% |
| 4石以上 | 52,751 | 6,277 | 13.1370% |
| 6石以上 | 32,990 | 5,819 | 8.2158% |

各データセットは `datasets/WTHOR_1977_2024_MISS_<閾値>/test/` に保存します。通常の `.dat` と `.game_ids` に加え、元test内の位置を示す `.source_indices` と石損を示す `.losses` が対応します。件数と抽出元の情報は `datasets/dataset_counts.csv` と `.json` に保存します。

評価では、4種類のデータに含まれる同じ盤面の推論と探索を共有します。一方、集計では元の局面レコードの出現回数をそのまま数えます。過去の探索キャッシュを読み取り専用で複製し、設定を照合したうえで不足する局面だけを追加探索します。エンジン内の探索表は各プロセスで再利用するため、過去と今回では探索履歴が異なる場合があります。そのため、使用した評価ファイルとモデルのSHA-256、キャッシュの参照先、再利用方針を結果JSONに記録します。

主な出力は次のとおりです。

| ファイル | 内容 |
| --- | --- |
| `evaluation/mistake_test_accuracy_matrix.csv` | 27モデル×4条件の一覧。百分率、一致数、評価件数 |
| `evaluation/mistake_test_summary.csv` | 108条件の詳細。一致率、完全一致率、95% Wilson区間 |
| `evaluation/mistake_test_summary.json` | 全結果、モデルとデータの識別情報、実行条件 |
| `evaluation/progress.json` / `.csv` | 実行中の集計。完了・途中・未計算を区別 |
| `evaluation/predictions.npz` | 元testのインデックス、石損、正解着手、27モデルの予測着手 |

`top1_accuracy` は、盤面を保つ対称変換で等価な着手も一致とする既存の定義です。`exact_top1_accuracy` は元の着手位置との完全一致率です。ランダム方策はseed 57で元testの順序に従って各レコードの合法手から1回ずつ抽選し、その着手を4条件で共有します。その期待一致率も `expected_random_accuracy` に保存します。

評価が中断した場合は、同じコマンドと出力先で再開できます。保存済みの探索結果を再利用し、残りを計算します。評価コマンドに `--dry-run` を付けると、データとモデルの識別情報を確認して終了します。前処理を作り直す場合の `--replace` は、以前のデータセットをバックアップしてから置き換えます。

完了後は次のコマンドで、予測手の合法性・対称性、12種類のエンジン設定とキャッシュの一致、全108条件の一致数を独立に検算できます。

```powershell
python verify_mistake_test_results.py `
  --datasets-dir output/mistake_test_20261005/datasets `
  --output-dir output/mistake_test_20261005/evaluation
```

成功すると `evaluation/verification.json` を保存します。検算に失敗した場合はエラーで停止します。

前処理から検算まで続けて実行する場合は、次のコマンドを使えます。既存のデータセットがあれば照合して再利用し、最後に `evaluation/RESULTS.md` へ27モデルの結果表を保存します。

```powershell
python run_mistake_test_experiment.py `
  --output-dir output/mistake_test_20261005 `
  --reference-dir ../../../../../experiment_bat/20261002/results/human `
  --normal-model ../../../../../experiment_bat/20261002/results/train/selected_model.h5 `
  --weighted-model ../10_train_policy_network/trained/prepared_conv8_bn_e10_20261003_221904_955/selected_model.h5 `
  --workers 32
```

処理全体の状態は `run_status.json` の `state` に記録します。`complete` は評価と独立検算の両方が完了した状態で、`failed` の場合は `error` に原因が残ります。同じ出力先に対して複数の評価プロセスを同時に起動しないでください。

## 通常のtestから標本を抽出して評価する

`run_random_wthor_blend_experiment.py` を実行すると、回転・反転などの対称変換で増やす前のtestデータから、ランダムに100,000局面を選んで次の設定で評価します。局面数 `N` を指定すると、抽出件数を変更できます。

- alpha = 0, 0.2, 0.4, 0.6, 0.8, 1 の混合方策。
- レベル **1, 5, 9, 13, 17** のEgaroucid for Console。レベル21の結果はalpha = 0で確認できます。
- レベル **1, 5, 9, 13, 17, 21** のEdax。

人間の着手データには、環境変数 `POLICY_DATA_ROOT` 配下の水増し前のデータ `board_split/<データセット名>/test` を使用します。データセット名は `--dataset` で選択でき、既定値は `WTHOR_1977_2024` です。`EGAROUCID_DATA` を使う場合、既定の参照先はその配下の `20261002_policy_network_human_like_data/board_split/WTHOR_1977_2024/test` です。

`POLICY_DATA_ROOT` を省略した場合は、`EGAROUCID_DATA` 配下の `20261002_policy_network_human_like_data` をデータルートにします。データ加工側と参照先をそろえる場合は、次のように設定できます。

```powershell
$env:POLICY_DATA_ROOT = Join-Path $env:EGAROUCID_DATA '20261002_policy_network_human_like_data'
```

このディレクトリから、次のコマンドを実行します。

```powershell
python run_random_wthor_blend_experiment.py `
  --dataset WTHOR_1977_2024 `
  --weights ../10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin `
  --workers 32 --egaroucid-threads 1 --edax-threads 1 `
  --output-dir output/human_match_WTHOR_1977_2024
```

上記の重みファイルは、`--weights` を省略した場合の既定値でもあります。別の学習済みモデルを評価する場合は、そのモデルの重みファイルを指定してください。コマンド末尾に `--dry-run` を追加すると、探索を行わずに標本の読み込みと実行構成を確認できます。

一致率は既定でtop-1だけを測定し、top-3は計算・表示・CSV・JSON出力を行いません。top-3も必要な場合は `--top3` を追加してください。比較用エンジンには既定で1手、`--top3` 指定時は3手を要求します。混合方策の尤度計算に使うEgaroucidレベル21は、どちらの場合も全合法手を探索します。候補手数が変わるため、top-3の設定を切り替える場合は別の出力先を指定してください。

`--workers` はEgaroucidとEdaxを合わせた探索プロセス数の上限です。両エンジンの全12設定を最初から動かすため、12以上を指定してください。32スレッドの環境で各エンジンを1スレッドで動かす場合は、`--workers 32` で合計32プロセス以内になります。

Policy Networkの推論後に行う集計準備では、合法手と盤面の対称性を4,096状態ずつまとめて計算します。この結果はEgaroucid・Edaxの途中集計と最終検算で共有します。準備の開始・完了時と `--progress-interval-sec` の間隔で、処理件数、進捗率、経過時間、処理速度、推定残り時間を表示します。準備にかかった時間は結果JSONの `timing.agreement_preparation_sec` に記録します。

入力はすでに分割済みの `test` なので、追加分割は行いません（`--data-split all`）。このtest全体から、入力レコード単位で重複なしに100,000件を無作為抽出します。抽出シード値は613で、同じデータ・seedなら同じ標本を再現できます。`WTHOR_1977_2024` では、対称変換前の401,545局面が抽出元です。抽出後も回転・反転によるコピーは追加しません。

件数を変更する場合は `python run_random_wthor_blend_experiment.py 30000 ...` のように局面数 `N` を指定し、seedを変更する場合は `--sample-seed` を使います。入力件数より大きいNを指定するとエラーになるため、小さなデータで動作確認する場合はNも小さくしてください。別のデータセットは `--dataset` で選択できます。モデル単体の評価スクリプトも、入力ディレクトリには同じ参照先と `--dataset` を使用します。

任意の盤面データを直接指定する場合は、`--board-data-dir` が環境変数と `--dataset` より優先されます。従来の未分割データに対して再分割を行う場合に限り、`--data-split test` などを指定してください。

Edaxの実行ファイルの既定パスは `bin/versions/edax_4_6/wEdax-x86-64.exe` で、実行ファイルと同じディレクトリ内の `data/eval.dat` を評価データとして使用します。これらのパスは `--edax-exe` と `--edax-eval-file` で変更できます。

`--edax-workers` は、共有する `--workers` 枠の内数としてEdaxの上限を指定します。両エンジンを並行して起動し、完了した探索の枠を再利用します。進捗は1つのログにまとめ、混合方策・Egaroucid for Console・Edaxの一致率を同じ更新で表示します。片方のエンジンが完了した後も、その結果を表示し続けます。逐次実行だった旧版から移行する場合は、出力先に `output/human_match_parallel` などの新しいディレクトリを指定してください。

Egaroucid for Consoleのレベル21はalpha = 0と同じ結果なので、進捗・最終表示・CSV・JSONの比較結果から重複行を省きます。混合方策に必要なレベル21の探索は引き続き行います。この表示変更によって探索条件は変わらないため、すでに並行実行版を使っている場合は、同じ出力先と実行条件でキャッシュを再利用できます。

Edaxでは `-n` でスレッド数を指定し、22ビットのハッシュテーブルを使用します。また、定石ブックと先読み（pondering）は無効にします。探索に失敗した場合の処理は、`--edax-timeout-sec` と `--edax-retries` で調整できます。

Edaxのhintには、`<-63` や `>+63` のように評価値の境界を示す記号が付く場合があります。[Edax 4.6の出力処理](https://github.com/abulmo/edax-reversi/blob/v4.6/src/search.c#L1038) に合わせ、数値部分を探索スコアとして使用し、記号はキャッシュのhint文字列にも残します。たとえば `<-63` の数値は-63として扱い、-64などの厳密な石差を推測しません。評価値中の `>` と、行頭にあるコマンド入力待ちの `>` は区別します。

盤面と手番の組み合わせごとに、各レベルで1回だけ探索します。ただし、人間の着手サンプルに重複がある場合は、その出現回数を集計に反映します。EdaxのキャッシュはEgaroucidのキャッシュと分けて保存し、実行ファイルと評価データのハッシュ、および探索設定を記録します。実験条件を変更する場合は、別の出力ディレクトリを使用してください。`RESULTS.md` にある過去の結果には当時の条件が記載されており、再計算は行いません。

## 学習済みの新モデルだけを追加評価する

前回の実験にlevel 21のキャッシュが揃っている場合は、`evaluate_wthor_blend_from_cache.py` を使います。
前回のsummaryに記録された入力データ、抽出件数、seed、温度、Top-N設定を引き継ぎ、新モデルを含む正のalphaだけを評価します。
通常の設定ではalpha = 0.2, 0.4, 0.6, 0.8, 1が対象です。Egaroucid・Edax単体とalpha = 0の結果は再計算しません。

このディレクトリから、学習完了後に次を実行してください。モデルと参照先は対象の実験に合わせて指定します。

```powershell
python evaluate_wthor_blend_from_cache.py `
  --reference-dir ../../../../../experiment_bat/20261002/results/human `
  --model ../10_train_policy_network/trained/<学習の保存先>/selected_model.h5 `
  --policy-batch-size 512 `
  --output-dir output/new_model_from_cache
```

レベル21のSQLiteキャッシュは読み取り専用で開き、全合法手の評価値を利用します。
標本のインデックスとレコードのSHA-256を前回と照合し、標本の相違やキャッシュの不足があれば停止します。エンジンによる追加探索は行いません。
末尾に `--dry-run` を加えると、学習中でも全標本とキャッシュを検証できます。この場合、モデル推論と結果保存は行いません。

出力先には空のディレクトリを指定してください。新モデルの一致率と尤度は `cached_wthor_blend_summary.csv` と `.json` に保存します。
重み付きデータで学習したモデルも、評価には前回と同じ標本を使い、各局面を同じ重みで集計します。

## 通常モデルと重み付きモデルのβ混合を評価する

`evaluate_wthor_model_blend.py` は、`model_blend_beta_0.2`・`0.4`・`0.6`・`0.8` の着手一致率をまとめて評価します。対戦用と同じ `P_normal^(1-beta) * P_weighted^beta` を合法手で正規化し、既存の対称性を考慮した一致率で集計します。β=0は通常モデル、β=1は重み付きモデルです。

このディレクトリから、現在の2モデルを次のコマンドで評価できます。

```powershell
python evaluate_wthor_model_blend.py `
  --reference-dir ../../../../../experiment_bat/20261002/results/human `
  --normal-model ../../../../../experiment_bat/20261002/results/train/selected_model.h5 `
  --weighted-model ../10_train_policy_network/trained/prepared_conv8_bn_e10_20261003_221904_955/selected_model.h5 `
  --betas 0.2,0.4,0.6,0.8 `
  --output-dir output/model_blend_beta_0.2_0.8
```

前回のsummaryと実験設定から同じ10万局面を復元し、標本のSHA-256を照合します。探索エンジンと探索キャッシュは不要です。既定ではTop-1のみ、`--top3` を追加するとTop-3も出力します。`--dry-run` ではモデルファイルと標本を確認し、推論・結果保存は行いません。

結果は指定先の `wthor_model_blend_summary.csv` と `.json` に保存します。出力先は空のディレクトリを指定してください。JSONには2モデルのパス・SHA-256と標本の情報も記録します。

## 一致率と尤度

Top-1と、任意で有効にするTop-3の一致率には、従来どおり盤面の対称性を考慮した定義を使用します。両プレイヤーの石の配置を保つ対称変換によって、人間が実際に選んだ手を予測手に対応させられる場合、一致とみなします。比較用のエンジン単体の評価では、各エンジンの `hint 1`（`--top3` 指定時は `hint 3`）の出力順序をそのまま使用します。Egaroucidのレベル21は、混合方策に必要な全合法手のヒントを計算します。

尤度は、**人間が実際に選んだその手に、各手法がどれだけの確率を与えたか**を表します。alphaごとに、次の式で計算します。

```text
z(move) = alpha * network_logit(move)
          + (1 - alpha) * Egaroucid_level21_score(move) / score_temperature
p(move) = exp(z(move)) / sum(exp(z(legal_move)))
```

実装では、**全合法手**を対象に数値的に安定した方法でlog-softmaxを計算し、その後、人間が選んだ手の値を取り出します。この指標では、対称性によって等価になる手の確率を合算しません。そのため、4つの手の確率が等しい初期盤面では、一致率は100%ですが、人間が選んだ手の尤度は25%になります。

混合方策のCSVとJSONには、alphaごとに次のフィールドを出力します。

| フィールド | 意味 |
| --- | --- |
| `mean_likelihood` | 人間が選んだ手の確率の算術平均。範囲は0〜1 |
| `mean_likelihood_percent` | 上記の平均を百分率で表した値 |
| `mean_log_likelihood` | 人間が選んだ手の確率の自然対数を平均した値。単位は局面あたりのnat |
| `negative_log_likelihood` | 平均対数確率の符号を反転した値。小さいほどよい |

有効な局面サンプルは、すべて同じ重みで集計します。同じ盤面でも人間が選んだ手が異なるサンプルは、それぞれ集計対象になります。ここで求める平均値は、全サンプルの確率を掛け合わせた値ではありません。

alpha = 1 は方策ネットワーク単体を、alpha = 0 は混合方策で使用する評価値のsoftmax方策を評価します。一方、比較用のエンジン単体の行には一致率だけを出力します。上位の候補手だけでは、全合法手の確率分布を正規化できないためです。

サンプルがない場合、尤度のフィールドはnullになります。また、方策ネットワーク単体の評価では、人間が選んだ手の確率が厳密に0の場合、対数指標をnullとして出力します。これは、JSONの標準にない無限大の値を出力しないためです。

## 出力と検証

- `random_wthor_blend_summary.csv`: 混合方策の一致率と尤度。
- `random_wthor_console_level_summary.csv`: Egaroucidのレベル別一致率。
- `random_wthor_edax_level_summary.csv`: Edaxのレベル別一致率。
- `random_wthor_blend_summary.json`: すべての結果、定義、設定、処理時間。
- `progress_egaroucid.json` / `progress_edax.json`: 全17参加者の途中集計。集計済み件数、Top-1一致率、計算対象の方策の平均尤度・NLL、未算出・暫定・完了の状態を保存。

途中集計は、Policy Networkの推論と集計準備が終わった時点、および探索の進捗表示時に更新します。各方策はhintが得られた局面から集計し、α=1のNN単体も、他のαと同じlevel 21の途中標本を使います。件数0の一致率はnullで保存します。進捗JSONと最終CSVは一時ファイルへ書いてから置き換えるため、実験用BATのサマリーから読み取る際に書きかけの数値が混ざりません。

方策ネットワーク単体を評価する `evaluate_wthor_human_match.py` も、合法手に条件づけて確率を正規化したうえで、JSONと全体集計・10手ごとの区間集計のCSVに尤度を出力します。一致率は既定でtop-1のみで、top-3も必要な場合は `--top-n 1,3` を指定します。上記の比較には `run_random_wthor_blend_experiment.py` を使用してください。

旧版のTop-N評価器、シャード実行・結合、重複解析、旧ブレンド評価器とその専用テストは `legacy/` に退避しています。このフォルダはGitの管理対象外です。現行の `evaluate_wthor_blend_human_match.py` には、現行評価から参照される盤面読み込み・標本抽出・順位計算などの共通処理だけを残しています。実行用CLIは `run_random_wthor_blend_experiment.py` と `evaluate_wthor_human_match.py` を使用してください。

```powershell
python -m unittest discover -s . -p 'test_*.py'
$env:WTHOR_EDAX_INTEGRATION = '1'
python -m unittest test_wthor_edax
```

任意で実行する結合テストでは、ローカルのEdax実行ファイルを使用し、黒番・白番それぞれの終盤局面を全6レベルで探索します。
