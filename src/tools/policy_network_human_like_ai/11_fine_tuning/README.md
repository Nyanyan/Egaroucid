# WTHORモデルをオセロクエストの棋力帯に合わせて追加学習する

`fine_tune_policy.py` は、通常のWTHOR学習で保存したモデルを読み込み、指定したオセロクエストの棋力帯で全層をファインチューニングします。既定の初期モデルは、既存30モデルの通常NNに使っているConv8＋BNです。

## 実行する

このディレクトリで、次のコマンドを実行します。

```powershell
python fine_tune_policy.py --rating-band under_800 --output-dir output/quest_under800
```

Windowsのコンソールで進捗を見たい場合は、同じ引数をBATへ渡せます。

```powershell
.\train_quest.bat --rating-band under_800 --output-dir output/quest_under800
```

`POLICY_PYTHON` にTensorFlowを導入したPythonのパスを指定できます。BATは終了後にキー入力を待つので、エラーが出た場合も表示を確認できます。

棋力帯は `--rating-band 800-1000` など、前処理で作成したデータセット名で変更します。レート範囲は下限を含み、上限を含みません。`under_800` は800未満です。各局面には、実際に着手するプレイヤーのレートを使います。

データは `POLICY_DATA_ROOT/othello_quest_split/board_split/<棋力帯>` から読み込みます。`POLICY_DATA_ROOT` が未設定なら、`EGAROUCID_DATA/20261002_policy_network_human_like_data` を使います。`--quest-root` で `othello_quest_split` の場所を、`--board-split-dir` で棋力帯のディレクトリを直接指定できます。元モデルは `--base-model` で変更できます。元モデルの隣には学習時の `prepared_dataset.json`、`selected_summary.json`、`training_identity.json` が必要です。

## 学習条件とデータの分割

既定値は10エポック、Adamの学習率0.0001、バッチサイズ512、seed 613です。保存済みモデルの重みと層ごとの計算精度を引き継ぎ、オプティマイザは新しく初期化します。Batch Normalizationを含め、全層を更新します。

学習局面だけを8通りの回転・反転で増やします。変換はバッチの読み込み時に行うため、元データへの書き込みはありません。`--augmentation 1` で変換を無効にできます。検証・テストでは元の局面を使います。

開始時にmanifestのハッシュ、ファイルの件数、棋譜IDの分割を確認します。同じ棋譜がtrain・val・testをまたいでいたら停止します。テスト分割はこの整合性確認に使い、学習やモデル選択には渡しません。

各エポックの検証損失を比較し、最小のモデルを `selected_model.h5` に保存します。同値なら早いエポックを採用します。テストの着手一致率と対戦レートは、このモデルを確定してから別の評価で求めます。

## 進捗と再開

`train.log` に学習ログ、`training_progress.json` にエポック・進捗・推定残り時間、`history.csv` に損失を保存します。`run_status.json` の `state` が `complete`、`returncode` が0なら正常終了です。採用エポックとモデルのSHA-256は `selected_summary.json` に残ります。

中断した場合は、同じコマンドに `--resume` を加えます。エポック終了時のモデルとオプティマイザから再開します。モデル・データ・実装・設定のいずれかが変わっている場合は、新しい出力先を指定してください。同じ出力先への二重起動はOSのファイルロックで防ぎます。

短い動作確認には `--epochs 1 --max-train-samples 64 --max-val-samples 32` を使えます。本番では件数上限を省略してください。`--dry-run` はデータと初期モデルのハッシュ・設定を確認して終了します。

```powershell
python -m unittest discover -s . -p 'test_*.py'
```

## 800未満モデルを評価して既存30モデルと比較する

2026年10月10日の実行は `output/quest_under800_20261010` に保存しています。
通常WTHORモデルから10エポック追加学習し、検証損失が最小だった10エポック目を採用しました。

この実行を評価するスクリプトは次のとおりです。結果は `output/quest_under800_20261010_analysis` に保存します。

| スクリプト | 処理 |
| --- | --- |
| `evaluate_finetuned.py` | 新モデルをQuestのtest全6,692局面と既存WTHOR testの10万局面で評価 |
| `run_strength_vs_30.py` / `run_strength.bat` | 既存30モデルと各50セット、先後交換で合計3,000局を追加対戦 |
| `evaluate_quest_baselines.py` / `run_quest_evaluation.bat` | 同じQuest test全6,692局面で既存30モデルを評価 |
| `verify_strength.py` | 既存対戦の不変性と、新規対戦の先後・石差・勝敗・集計を検算 |
| `plot_results.py` | 検証済み一致率とEloからWTHOR用・Quest用のPNG/SVGを出力 |
| `finish_experiment.py` / `run_finish.bat` | 対戦完了後に検算とEloの95%区間推定を実行し、一致率評価の完了を待って作図・結果文書の保存を実行 |

対戦には `40_test_strength/output/gpw30_independent_xot_50sets` の43,500局を引き継ぎます。モデル、対戦条件、パッケージ、開始局面、元の対戦記録を照合してから、新しい3,000局だけを追加します。各相手との50セットは相手の組ごとに独立抽出したXOT局面を使い、各局の置換表を消去します。

Questの比較評価では、完了済みの1400未満テスト評価から条件の一致した探索キャッシュを別ファイルへコピーし、不足する局面だけを探索します。集計には今回の800未満testだけを使います。参照キャッシュは読み取り専用で開き、元ファイルを変更しません。探索中の置換表はワーカー内で再利用するため、過去の探索と新しい探索では履歴が異なる場合があります。

新モデルと既存30モデルのQuest予測は、保存後に局面レコードと対称性マスクが一致することを検証します。WTHORでは元の10万局面の抽出ハッシュと既存30モデルの予測・集計を照合します。一致率は同じテスト集合内で比較できます。

完了すると `comparison_31_models.csv`、`report_data.json`、`RESULTS.md`、`elo_vs_agreement_wthor.png/.svg`、`elo_vs_agreement_quest.png/.svg` が揃います。Eloは全31モデルの平均を1500にした相対値で、オセロクエストの実レートとは異なります。

2026年10月10日の評価は18:53に完了しました。途中、Egaroucidのレベル21で1局面が30分の制限を超えたため、`run_quest_recovery.bat` から保存済みキャッシュを使って再開しました。再開用の `resume_quest_evaluation.py` は応答待ちの上限だけを4時間に延ばし、探索レベル・スレッド数・hint数・評価局面は維持します。変更の記録は `quest_baselines/runtime_recoveries` に保存します。
