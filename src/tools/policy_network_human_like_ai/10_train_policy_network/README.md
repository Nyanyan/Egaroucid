# 分割済みデータによるポリシーネットワーク学習

関連issue: #613

`train_policy_network.py` は、前処理済みのtrainで学習し、valでモデルを選択した後、testで最終評価します。
データ作成の手順は [00_data_processing](../00_data_processing/README.md) を参照してください。

## データの場所と切り替え

データの場所は環境変数から取得します。環境固有の絶対パスは、コードや文書へ書かず、実行環境に設定してください。

| 設定 | 内容・既定値 |
| --- | --- |
| `POLICY_DATA_ROOT` | `board_split_increased` を含むデータルート |
| `EGAROUCID_DATA` | `POLICY_DATA_ROOT` が未設定の場合、この配下の `20261002_policy_network_human_like_data` を使う |
| `POLICY_DATASET` | データセット名。既定値は `WTHOR_1977_2024` |
| `--data-root` / `--dataset` | 対応する環境変数より優先する引数 |
| `--board-split-dir` | stage 03/04のデータセットディレクトリを直接指定する場合に使用 |

通常の読み込み先は次の構成です。

```text
$POLICY_DATA_ROOT/board_split_increased/$POLICY_DATASET/
  manifest.json
  train/*.dat, *.game_ids
  val/*.dat, *.game_ids
  test/*.dat, *.game_ids
```

既存の `EGAROUCID_DATA` を使う場合は、データセット名だけ設定すれば起動できます。
データルートを明示したい場合も、環境変数から組み立てられます。

```powershell
$env:POLICY_DATA_ROOT = Join-Path $env:EGAROUCID_DATA '20261002_policy_network_human_like_data'
$env:POLICY_DATASET = 'WTHOR_1977_2024'
```

別のデータセットへ切り替えるときは、`POLICY_DATASET` または `--dataset` を変更してください。
水増し前のstage 03を使う場合は、`--board-split-dir` に `$env:POLICY_DATA_ROOT/board_split/<データセット名>` を渡します。

石損で重みを付けたデータは、前処理の `03_generate_weighted_board_data.py` で生成します。
各 `.dat` に対応する `.weights` の値を自動的に読み、Kerasへ `sample_weight` として渡します。
各局面を1件ずつ使い、石損0・2・4・6に対する重みは1・2・4・8です。奇数の石損にも `2 ** (石損 / 2)` を適用します。

ファイルには元の重みを保存します。学習時は全train局面の平均重みでtrain・val・testの重みを割り、比率を保ったままlossの桁を抑えます。
val・testの情報は、この正規化係数に使いません。係数は `prepared_dataset.json` に記録します。
シャッフルや件数上限を指定した抽出でも、局面と重みの対応を保ちます。重みの欠損、件数不一致、非正値・非有限値は開始時に検出します。
Kerasのtrain/val lossには重みが反映されます。一方、モデル選択と最終評価に使う既存のtop-N一致率・cross entropyは、各局面を等しく数える指標です。

```powershell
python train_policy_network.py --board-split-dir "$env:POLICY_DATA_ROOT/board_split/WTHOR_1977_2024_LOSS_WEIGHTED"
```

04で8倍化した重み付きデータを使う場合は、`--dataset WTHOR_1977_2024_LOSS_WEIGHTED` を指定します。

「石損数＋1」の重みで学習する場合は `train_conv8_bn_loss_plus_one.bat` を実行します。
0・1・2・4・6石損の重みは1・2・3・5・7倍です。
`train_loss_plus_one.py` が既存の指数重みから整数の石損を復元・検証し、別のデータセットを作ります。
局面と棋譜IDはそのままコピーし、SHA-256で一致を確認します。全trainの平均による正規化は従来と同じです。
既定条件はConv8＋BN、10エポック、バッチ512、seed 613、float32、全件使用です。
`--run-dir` で保存先を指定でき、その下の `logs/train.log` に進捗、`train/selected_model.h5` に最終モデルを保存します。
再開時は同じ引数に `--resume-training` を追加します。各エポック終了時のチェックポイントから再開します。

## 学習と最終評価

分割は前処理で確定したものを使います。学習時に棋譜や8倍化済み局面を再分割しません。
開始時にmanifestのSHA-256、ファイル一覧、件数、棋譜IDを確認し、分割をまたぐ同一棋譜を検出した場合は停止します。

`WTHOR_1977_2024` の2026年10月2日時点のmanifestには、次の件数が記録されています。

| 用途 | 棋譜数 | 8倍化後の局面数 |
| --- | ---: | ---: |
| train | 120,768 | 57,856,696 |
| val | 6,707 | 3,213,200 |
| test | 6,706 | 3,212,360 |

入力は手番側64マスと相手側64マスの計128特徴量です。
全局面の特徴量をメモリに展開すると大きくなるため、19バイトの局面レコードをメモリマップし、バッチごとに変換します。
trainは各エポックでブロックの順番とブロック内の局面をシャッフルします。ブロックは約65,536局面です。
valとtestは固定順序で評価します。

各構成では指定したエポック数まで学習し、最終エポックのモデルを保存します。
複数構成を指定した場合は、valの1位一致率、上位3手一致率、パラメータ数の少なさの順で代表モデルを選びます。
その後、保存した代表モデルを読み直し、testを一度だけ評価します。
`run_training_experiments.py` は構成ごとに独立した実行を行うため、各実行でtest評価を行います。
構成間で代表モデルを選び、testを一度だけ評価したい場合は、学習スクリプトの `--configs` に複数構成を渡してください。

## 実行例

以下は、このディレクトリで実行する例です。
`train_policy_network.py` と `run_training_experiments.py` の既定値はConv8+BN、10エポック、バッチサイズ512です。

```powershell
python train_policy_network.py
python train_policy_network.py --configs 512x4 --epochs 10
python train_policy_network.py --dataset another_dataset --configs conv8_bn --epochs 10
python train_policy_network.py --configs 512x4,conv8_bn --epochs 10
```

件数上限の既定値はなく、train・val・testを全件使用します。
短い動作確認に限り、各分割からseedに基づいて重複なしで抽出できます。分割の所属は変わりません。

```powershell
python train_policy_network.py --configs 64x3 --epochs 1 `
  --max-train-samples 65536 --max-val-samples 8192 --max-test-samples 8192
```

上限を指定した結果は抽出した局面の評価です。test全件で確認する場合は `--max-test-samples` を省略してください。

Windowsでは次のBATファイルでも学習できます。

| BATファイル | モデル | 既定エポック数 | 既定バッチサイズ |
| --- | --- | ---: | ---: |
| `train_prepared.bat` | Conv8＋BN | 10 | 512 |
| `train_dense512.bat` | Dense 512×4、LeakyReLU 0.03 | 10 | 8192 |
| `train_conv4.bat` | Conv4 | 10 | 512 |
| `train_conv6.bat` | Conv6 | 10 | 512 |
| `train_conv8.bat` | Conv8 | 10 | 512 |
| `train_conv8_bn.bat` | Conv8＋BN | 10 | 512 |

モデル別のBATは共通の `train_prepared.bat` を呼び、同じ環境変数とデータ分割を使います。
`EPOCHS`、`BATCH_SIZE`、`SEED` は環境変数で変更できます。
`POLICY_PYTHON` でTensorFlowを導入したPythonを選択し、`POLICY_NO_PAUSE=1` で終了時のキー入力待ちを省略できます。
`train_other_paper_models.bat` はConv4、Conv6、Conv8を順に実行し、失敗した場合は停止します。

## 出力

通常の保存先は `trained/<実行日時>`、BATでは `trained/prepared_<モデル名・エポック数・実行日時>` です。
`--output-dir` で変更できます。既存結果との混在を防ぐため、空のディレクトリを指定してください。

| ファイル | 内容 |
| --- | --- |
| `prepared_dataset.json` | データセット名、manifestのSHA-256、分割ごとの使用件数・抽出seed・抽出インデックスのSHA-256 |
| `<モデル名>/history.csv`, `keras_log.csv` | 各エポックのtrain/val損失 |
| `<モデル名>/model.h5`, `summary.json` | 最終エポックのモデルと結果 |
| `<モデル名>/split_topn_accuracy.json`, `.csv` | モデル選択に使ったval一致率 |
| `results.json`, `.csv` | 各構成の比較結果 |
| `selected_model.h5`, `selected_summary.json` | 採用モデルとval/test結果 |
| `test_metrics.json`, `test_topn_accuracy.csv` | 採用後のtest件数、上位n手一致率、交差エントロピー |
| `selected_policy_network_weights.bin` | 全結合モデルのC++向け重み。CNNはH5を使用 |

testの交差エントロピーは、合法手マスク前の出力確率と記録された着手から計算します。
BATの実行ログと最大メインメモリ使用量は `train_log` に保存します。GPUメモリ量は含みません。

旧records1読み込み・再分割を行っていた学習コード、旧BAT、旧READMEは、ローカルの `legacy/prepared_split_migration` に退避しています。
現在の学習で使わない次のファイルは、`legacy/unused_training` へ移動しています。

| 移動したもの | 内容 |
| --- | --- |
| `evaluate_policy_topn.py` | 旧records1向けの単独評価スクリプト |
| `evaluate_model_size_experiment.py`, `test_evaluate_model_size_experiment.py` | 旧モデルサイズ比較とそのテスト |
| `RESULTS.md`, `model_size_results.csv`, `model_size_results.json` | 旧モデル比較の記録 |
| `policy_network.hpp`, `policy_network_sample.cpp` | 現在の処理から参照されないC++推論サンプル |
| `trained/` 配下の過去36実行 | 対戦・評価処理から参照されていない過去の学習結果 |

対戦・評価処理で使用中の3モデルと、今回のConv4動作確認結果は元の `trained` に残しています。
移動先では元の相対パスを保ち、`archived_files.json` に記録したSHA-256で全597ファイルの内容が変わっていないことを確認しました。
`legacy` はGitのignore対象です。今後の学習出力もignore対象とし、既存の対戦用 `wthor_final_arch_512x4_e50` のみ引き続きGitで管理します。

## モデルのアーキテクチャ

`--configs` で、既存の全結合モデルと論文のCNNを選べます。複数指定する場合はカンマで区切ります。
たとえば `--configs 512x4,conv8_bn` では、同じ棋譜分割を使って両方のモデルを学習します。

CNNは Liskowski・Jaśkowski・Krawiec の
[*Learning to Play Othello With Deep Neural Networks*](https://doi.org/10.1109/TG.2018.2799997)
（2018年、Table I・V、pp. 357–359）の構成に基づきます。

| 指定名 | 畳み込み層のチャネル数 | Batch Normalization |
| --- | --- | --- |
| `conv4` | 64 → 64 → 128 → 128 | なし |
| `conv6` | 64 → 64 → 128 → 128 → 256 → 256 | なし |
| `conv8` | 64 → 64 → 128 → 128 → 256 → 256 → 256 → 256 | なし |
| `conv8_bn` | Conv8と同じ | あり |

`Conv8+BN` と `conv8-bn` も `conv8_bn` の別名として使えます。
各CNNは、手番側と相手側の石を表す8×8×2の入力に対して、3×3の畳み込み、stride 1、ゼロパディング、ReLUを使います。
途中の盤面サイズは8×8に保ち、プーリングは使いません。
最後は全結合128ユニット（ReLU）と、全結合60ユニット（softmax）です。
重みはHe初期化、バイアスはゼロ初期化とし、dropoutは使いません。

論文の60出力は、初期配置で埋まっている中央4マスを除いた着手候補に対応します。
本実装も60出力で学習し、既存の評価・推論処理に渡す際は、中央4マスへ確率0を挿入して64出力に変換します。
この変換には学習パラメータがなく、入力の128特徴量と既存の着手番号もそのまま使えます。

BNの具体的な配置と数値設定は論文本文に指定がないため、本実装では各畳み込み層と隠れ全結合層に
「線形演算 → BN → ReLU」を適用します。BNは `momentum=0.99`、`epsilon=0.001` とし、BNの前のバイアスを省略します。
論文で案内されている[著者のリポジトリ](https://github.com/pliskowski/deep-othello)は、2026年10月1日の確認時点では空で、実装の照合はできていません。

学習にはAdamと前処理で保存した分割を使います。対称変換による8倍化は前処理の04で行います。
論文のSGD＋momentum、学習率スケジュール、baggingは再現していません。
CNNでも `--learning-rate` と `--l2` を指定できます。`--alpha` と `--dropout` は全結合モデルにだけ適用されます。

CNNを含む実行では、学習・予測の既定バッチサイズはともに512です。明示した値がある場合はそちらを使います。
全結合モデルだけの場合は、従来どおり学習4096・予測8192です。

## 一致率

一致率は、合法手だけを残したポリシーネットワーク出力で計算します。

各局面について、正方形の盤面に対する8通りの変換
（そのまま、90度・180度・270度回転、縦・横・2本の対角線を軸にした反転）を調べます。
そのうち、手番側・相手側の石配置を変えない変換だけを一致判定に使います。
WTHORで実際に打たれた手と、その手をこれらの変換で移した合法手を同一の着手として扱います。
それ以外の対称的な位置にある手は別の着手です。

- 1位一致率: 同一とみなす着手のいずれかが、合法手内で最も確率の高い手だった局面サンプルの割合です。
- 上位n手一致率: 同一とみなす着手のいずれかが、合法手内の確率上位n手に入った局面サンプルの割合です。

盤面に該当する対称性がなければ、実際に打たれた1手だけが一致対象になります。

## 動作確認（2026年10月2日）

`WTHOR_1977_2024` のtrain・val・test全件を使い、64×3モデルを1エポック学習して確認しました。
これは読み込みから最終評価までの動作確認であり、BAT既定の50エポック学習結果ではありません。

```powershell
python -u run_with_resource_log.py `
  --log train_log/prepared_full_check.log `
  --summary train_log/prepared_full_check_resource.json `
  -- python -u train_policy_network.py --configs 64x3 --epochs 1 `
  --output-dir train_log/prepared_full_check
```

| 項目 | 結果 |
| --- | ---: |
| test評価局面数 | 3,212,360 |
| test Top-1一致率 | 37.499% |
| test Top-3一致率 | 68.635% |
| test Top-5一致率 | 84.737% |
| test交差エントロピー | 2.037945 |
| 読み込み・学習・評価の合計時間 | 84.93秒 |
| 最大メインメモリ使用量 | 3742.89 MiB |

さらに、Conv4のBATをtrain 512局面・val/test各256局面、1エポックで実行し、モデル保存とtest評価の完了を確認しました。
整理後の学習側の自動テスト15件と、前処理側の自動テスト18件も成功しています。旧モデルサイズ比較の5件は、対象コードとともに `legacy` へ移動しました。
