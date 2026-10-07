# WTHORの人間の着手との一致度評価

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
