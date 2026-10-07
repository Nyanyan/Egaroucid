# 対戦レートと人間の着手一致率の描画

`plot_rating_vs_human_match.py`は、集計済みの`report_data.json`を読み込み、20261007_2のまとめレポートと同じ6種類の図をPNG・SVGで出力します。描画だけを行うため、学習・推論・対戦を再実行する必要はありません。

## 実行

このフォルダの`run_rating_vs_human_match.bat`を実行してください。既定ではワークスペースの`experiment_bat/20261007_2/report_data.json`を読み込み、[output/20261007_2](output/20261007_2)へ保存します。実行時のカレントディレクトリには依存しません。

Pythonから実行する場合は、次のコマンドをこのフォルダで使います。

```powershell
python .\plot_rating_vs_human_match.py
```

必要なライブラリは`numpy`と`matplotlib`です。BATでは`POLICY_PYTHON`にPython実行ファイルを指定できます。自動実行で終了時のキー入力待ちを省く場合は、`POLICY_NO_PAUSE=1`を設定してください。

別の実験の集計JSONを描画するときは、入力と出力先を指定します。

```powershell
python .\plot_rating_vs_human_match.py `
  --input ".\input\report_data.json" `
  --output-dir ".\output\my_experiment"
```

レートと一致率の図だけを出力することもできます。以下のオプションはBATにも渡せます。

```powershell
.\run_rating_vs_human_match.bat `
  --figures elo_vs_agreement mistake_elo_vs_agreement `
  --formats png svg
```

## 配色とブレンド係数

| 系列 | 色 | 図中の記号 |
| --- | --- | --- |
| Egaroucid | 緑 `#249972` | E：レベル |
| Edax | 青 `#2468B4` | D：レベル |
| 通常NN＋Egaroucid L21 | 赤 `#D62728` | N：通常NN側のα |
| 通常NN＋石損NN | オレンジ `#D55E00` | B：石損NN側のα |
| ランダム | 灰 `#747C85` | R |

石損NNは、学習時に`（石損＋1）²`の重みを用いたモデルです。通常NNとの融合は`P(a) ∝ P通常NN(a)^(1−α) × P石損NN(a)^α`で、B0.0は通常NN単体、B1.0は石損NN単体です。石損NN単体も、融合のαが1である同じ系列として接続します。

通常NN＋Egaroucid L21の系列は、`P(a) ∝ P通常NN(a)^α × P_L21(a)^(1−α)`です。この系列のαは通常NN側の係数です。N1.0とB0.0は同じ通常NNを指し、赤とオレンジの線が共有する端点になります。全モデルを並べる図では、この端点をN1.0として一度だけ表示します。石損NN系列だけの閾値別パネルでは、オレンジのB0.0として表示します。

## 95%信頼区間

各折れ線に、線と同じ色で一致率の95% Wilson区間を帯として表示します。帯の濃さの既定値は10%（不透明度`alpha=0.10`、透過率90%）です。点の横線はEloの95%区間、縦線は一致率の95%区間です。十字の誤差バーとその端の線は、常に不透明（`alpha=1.0`）で表示します。

帯の濃さは次のオプションで変更できます。`0`は完全に透明、`1`は不透明です。十字の表示には影響しません。

```powershell
python .\plot_rating_vs_human_match.py --ci-alpha 0.10
```

以前の`--ci-transparency`も使えます。こちらは透過率なので、`--ci-transparency 0.90`で上の指定と同じ薄さになります。

区間の値は入力JSONの保存値を使用します。20261007_2のElo区間はXOT局面単位の20,000回BCaブートストラップ、一致率の区間は局面単位のWilson区間です。一致率の区間には棋譜内相関の補正がありません。

帯は各点の下限・上限を、レベルまたはαの順に直線で接続したものです。点ごとの95%区間を可視化しており、曲線全体を95%の確率で覆う同時信頼帯や、2モデルの差に対する信頼区間ではありません。Eloの区間は横線で個別に示し、帯へ合成していません。

## 出力

| ファイル名（PNG・SVG） | 内容 |
| --- | --- |
| `model_overview` | モデル別のEloと全体の一致率 |
| `elo_vs_agreement` | Eloと一致率の関係：全体10万局面と2石以上の石損条件 |
| `mistake_model_comparison` | 石損条件ごとの全モデル比較 |
| `threshold_trends` | 系列別・ブレンド係数別の閾値による変化 |
| `agreement_heatmap` | 全体・石損条件別の一致率一覧 |
| `mistake_elo_vs_agreement` | すべての石損条件でのEloと一致率の関係 |

`plot_manifest.json`には、その実行で使った入力パス・SHA-256・系列・配色・区間の濃さ・ラベル位置・出力一覧を保存します。出力画像の解像度は既定200 dpiで、`--dpi`で変更できます。

## タイトル・ラベルなどの文字と位置の変更

`adjust_rating_vs_human_match_labels.bat`を実行すると、6種類すべての図を順に編集できます。レート対一致率の2種類から表示し、一つのウィンドウを閉じると次の図へ進みます。全体10万局面の右隣は、2石以上の石損条件です。レート対一致率の各パネルには全モデルのラベルを表示し、どの点ラベルも個別に動かせます。共有端点は同じモデルなので、各パネルに一つ表示します。

保存済みの位置がないラベルは、点やほかのラベルと重なりにくい位置へ自動配置します。保存済みの位置は優先し、自動配置では動かしません。

1. ラベルをマウスの左ボタンでドラッグして移動します。
2. 文字をダブルクリックすると、表示内容を自由に変更できます。図全体のタイトル、各パネルのタイトル、軸名、凡例、目盛り名、モデル名、数値ラベル、注記が対象です。クリックで選んでから`E`キーを押しても編集できます。
3. `S`または`Ctrl+S`で、文字と点ラベルの位置をJSONへ保存し、PNG・SVGも更新します。
4. `R`で、そのウィンドウを開いたときの文字・位置へ戻します。保存したい場合は再度`S`を押してください。

保存せずウィンドウを閉じた場合は、既存の位置ファイルと画像を変更しません。点から離したラベルには、対応する点への引き出し線が付きます。ツールバーの拡大・移動モード中はラベルをドラッグできないので、調整前にそのモードを解除してください。

通常のBATに`--adjust-labels`を付けても同じ画面を開けます。1種類だけを編集する例です。

```powershell
.\run_rating_vs_human_match.bat --adjust-labels --figures elo_vs_agreement
```

設定は既定で`output/20261007_2/label_positions.json`へ保存します。点ラベルは`figures`、タイトル・軸名などは`texts`に記録します。図名やパネルの識別子ごとに保存するため、同じ名前の文字でも設定が混ざりません。文字を変更しても、対応するモデルや元データ、目盛りの数値座標は変わりません。次回の通常描画でも、このファイルを自動的に読み込みます。以前の位置ファイルも引き続き読み込めます。

`--label-positions`で別の位置ファイルを指定できます。`--reset-labels`を付けると保存済み位置を無視して既定位置から開きます。編集画面で`S`を押すまでは保存済みファイルを変更しません。新しい位置ファイルはversion 2形式で、`legacy`の旧図専用ファイルとは別です。

編集画面にはQtまたはTkが使えるPythonが必要です。Qtを優先し、使えない場合はTkで起動します。この環境のWindows Python 3.9とQtで動作を確認しています。通常の画像生成にはGUIを使いません。

## 入力JSONの形式

20261007_2の`report_data.json`と共通の形式を使います。描画に必要なのは以下の項目です。

- `models`：`model_id`、表示名`model`、短縮名`short`、`group`、`elo`、`elo_low`、`elo_high`、`results`。
- 各モデルの`results`：`dataset`、`positions`、`top1_accuracy`、`top1_ci95_lower`、`top1_ci95_upper`。一致率とその区間は0〜1の値です。
- `dataset_counts`：`dataset`、`label`、`positions`、`minimum_loss`。先頭は全体データで`minimum_loss=null`、以降は石損条件です。
- `series_paths`：系列名からモデルIDの配列への対応。配列の順序を線の接続順として使います。共有する端点は複数の系列に含められます。

系列名は`Egaroucid`、`Edax`、`normal`、`squared_blend`、`random`です。モデル数や石損条件の数は固定していません。各モデルの結果は配列の位置ではなく`dataset`で対応づけ、件数の不一致・欠損・不正な区間を検出したら描画を中止します。`elo_vs_agreement`の右パネルは`minimum_loss=2`のデータを選びます。この条件がない入力はエラーにして、別の石損条件を表示することを防ぎます。元のレート・一致率を再推定したり、異なる実験の測定値を混ぜたりする処理はありません。

## 既存の実験と過去の図

古い実験用のMarkdownと`run_blend_experiment_*.bat`は`legacy`へ移しました。操作手順は[legacy/MODEL_BLEND_EXPERIMENT.md](legacy/MODEL_BLEND_EXPERIMENT.md)にあります。旧BATの呼び出し先を更新し、既存実験の実行・再開に必要なPython処理と出力先はそのまま使えるようにしています。今回の6種類の図を描く入口は`run_rating_vs_human_match.bat`です。

7月の固定CSVを描画していたスクリプト・テスト・CSV・ラベル位置・画像・結果説明は[legacy](legacy/README.md)に保存しています。新しいグラフのラベル編集は、このフォルダの`adjust_rating_vs_human_match_labels.bat`から行えます。

## 確認用コマンド

```powershell
python -m unittest test_plot_rating_vs_human_match test_label_editor -v
```

入力データの対応づけ、共有端点と接続順、色、95%区間の上下限と濃さ、PNG・SVGの生成、ラベルのドラッグ・保存・再読込・リセットを確認します。
