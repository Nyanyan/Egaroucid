# Questデータだけを使ったゼロからの学習

`train_quest_scratch.py` はConv8＋Batch Normalizationをランダム初期化し、指定した棋力帯のQuestデータで学習します。モデルの構成は通常WTHORモデルと同じです。初期重みは各層の初期化関数で生成し、学習済みモデルから重みを読み込みません。

## 800未満の学習と評価を実行する

このディレクトリで `run_quest_scratch.bat` を実行すると、10エポック学習した後、採用したモデルをQuestとWTHORで評価します。学習は `output/quest_under800_20261010`、評価結果は `output/quest_under800_20261010_analysis` に保存します。既存のファインチューニング結果は別のディレクトリに残ります。

```powershell
.\run_quest_scratch.bat
```

TensorFlowを導入したPythonを使います。必要なら環境変数 `POLICY_PYTHON` に実行ファイルのパスを指定してください。データの場所は前回と同じく `POLICY_DATA_ROOT` または `EGAROUCID_DATA` から解決します。

## 学習・評価の条件

学習はAdam、学習率0.001、バッチ512、seed 613、mixed_float16で行います。前回のファインチューニングの学習率は0.0001だったため、初期重みだけが異なる比較ではありません。

800未満の学習分割は114,026局面です。回転・反転による8通りの変換で、1エポックあたり912,208局面を学習します。検証は元の6,016局面を使い、検証損失が最小のエポックを採用します。学習・検証・テストは棋譜単位で分離し、テストをモデル選択には使いません。

`evaluate_scratch.py` は、Questのtest全6,692局面と、前回と同じWTHORの10万局面で着手一致率を求めます。合法手のTop-1を選び、盤面を保つ対称変換で等価な手も一致と数えます。保存済み予測を再集計し、前回と局面レコード・対称性マスクが一致することも確認します。

## 保存内容と再開

`initialization.json` に初期重みのハッシュ、初期化seed、更新前のオプティマイザの状態を記録します。`training_identity.json` にはデータ・コード・設定、`history.csv` には各エポックの損失、`selected_summary.json` には採用エポックとモデルのハッシュを保存します。

中断後は同じBATを再実行できます。学習途中なら同じ実行のエポック終了時のチェックポイントから再開し、完了済みならモデルのハッシュを照合して評価へ進みます。設定を変えて新しい学習を行う場合は、`train_quest_scratch.py --output-dir <新しい保存先>` を使ってください。

評価結果は `RESULTS.md`、`comparison.csv`、`report_data.json` にまとめます。比較表には通常WTHORモデル・800未満へのファインチューニング・今回のゼロ学習を掲載します。

初期化の検証は `python -m unittest -v test_scratch_initialization.py` で実行できます。
