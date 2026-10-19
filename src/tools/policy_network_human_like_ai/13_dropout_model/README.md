# WTHOR学習済みモデルへの推論時Dropout

既存の通常WTHORモデル（Conv8＋BN）の重みを読み込み、推論時だけDropoutを適用して、オセロクエストの着手一致率を計測します。
畳み込み層8個と隠れ全結合層1個のReLU直後に、同じ率のDropoutを追加します。
元の重みとBNの平均・分散を引き継ぎ、BNは推論モードに固定します。
高いdropout率での半精度計算のオーバーフローを避けるため、全条件の推論計算をfloat32にそろえます。

## 実行

```powershell
.\run_dropout_sweep.bat
```

既定の率は `0.0・0.2・0.4・0.6・0.8` です。任意の率も指定できます。

```powershell
.\run_dropout_sweep.bat --dropout-rates 0.0 0.1 0.3 0.5
```

元モデルの既定値は、このワークスペースの `experiment_bat/20261002/results/train/selected_model.h5` です。
`--base-model` で変更できます。元モデルと同じ場所に学習記録の `selected_summary.json`、`training_identity.json`、`prepared_dataset.json` が必要です。
データルートは `POLICY_DATA_ROOT`、未設定なら `EGAROUCID_DATA/20261002_policy_network_human_like_data` を使用します。
`--quest-root` で `othello_quest_split` の場所を直接指定できます。

## 評価条件

評価対象は800以上2200未満の200刻み・7帯です。各帯の水増し前testを全件使います。
着手したプレイヤーのレートで分類され、棋譜単位でtrain・val・testが分離された前処理済みデータを参照します。

各局面で1回、Dropoutを有効にして推論します。出力を何回も平均する処理は行いません。
乱数seedは既定で613＋レート帯の番号（0～6）です。各帯ごとにモデルと乱数を初期化するため、中断・再開しても同じ条件で計測できます。

着手は合法手内の最大値で選びます。高いdropout率ではsoftmaxの確率が0へ丸められる場合があるため、順位にはsoftmax前の値を使います。
盤面を保つ回転・鏡映で等価な人間の着手も一致として数えます。同点では着手番号の小さい手を選びます。
率0.0は同じ重みをfloat32で計算するDropoutなしモデルと同じ着手を選ぶことをtest全局面で照合します。また、各帯の評価後に全重みとBNの統計値が元モデルと一致することを確認します。

## 出力と再開

既定の保存先は `output/inference_dropout_<実行日時>/` です。

| ファイル | 内容 |
| --- | --- |
| `RESULTS.md` | 7帯×各dropout率の一致率比較表 |
| `accuracy_matrix.csv` | 比較表のCSV |
| `summary.csv` / `summary.json` | 一致数・局面数・一致率・モデルと局面のSHA-256 |
| `experiment_identity.json` | 元モデル・データ・率・seed・コードの記録 |
| `run_status.json` | 全体の完了数、経過時間、推定残り時間 |
| `models/dropout_*.h5` | 元の重みを保持し、推論時Dropoutを有効にしたモデル |
| `quest/dropout_*/<帯>/predictions.npz` | 各局面の選択手・正解マスク・棋譜ID |

H5はKeras標準の層だけで構成し、`load_model(path, compile=False)` で読み込めます。保存したモデルも推論時にDropoutが有効です。
コンソールには進捗と推定残り時間を表示します。`Ctrl+C` またはウィンドウを閉じると評価を中断します。
再開時は同じ引数と保存先に `--resume` を追加します。完了済みの帯は記録と予測ファイルのSHA-256を確認して再利用します。

```powershell
.\run_dropout_sweep.bat --output-dir output/inference_dropout_run --resume
python -u test_dropout_sweep.py
```

旧学習用スクリプトは `legacy/unused_training` に退避しています。現在の実行経路では学習処理を呼び出しません。
