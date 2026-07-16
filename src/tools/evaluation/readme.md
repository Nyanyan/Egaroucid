# Evaluation

評価関数を作る

## 現在おすすめの評価関数: 7.7beta + FM dim16 linearft

2026-07-16時点のおすすめモデルは `7.7beta + FM dim16 linearft` です。通常ビルドでは `bin/resources/eval.egev4` を読みます。`.egev4` は、7.7beta線形評価にFactorization Machineを追加したモデル形式として正しい拡張子です。

保存用パッケージは `model/20260716_7_7_fm_dim16_linearft` にあります。ここには最終モデル、中間モデル、学習ログ、データ変換ログ、データマニフェスト、生の対戦ログ、学習ツールのソーススナップショットを保存しています。詳しい採用レポートは `src/tools/evaluation/report/20260716_eval77_fm_linearft_release_report_ja.md` を参照してください。

### 用語

* FM: Factorization Machine。特徴同士の組み合わせの効果を評価値に足すモデルです。
* phase: 石数帯ごとの学習単位です。今回のモデルでは phase 6 から 59 を学習しました。
* holdout: 学習に使わず、誤差確認だけに使う検証データです。
* MAE: 平均絶対誤差です。小さいほど教師データに近い評価です。

### 使用した学習データ

生成済み学習バッチは次の3つです。

* `ignored/eval_experiments/20260710_eval77_fm_simdopt/eval77_batch_phase6_21_same_split`
* `ignored/eval_experiments/20260710_eval77_fm_simdopt/eval77_batch_phase22_40_same_split`
* `ignored/eval_experiments/20260710_eval77_fm_simdopt/eval77_batch_phase41_59_same_split`

学習には生成バッチID `0,1,2` を使い、holdoutには生成バッチID `3` を使いました。レコード数の上限は設定していません。合計は、学習 1,294,463,367 records、holdout 430,993,584 records です。

元のboard-data record IDの一覧とphaseごとの正確な使用ファイルは `model/20260716_7_7_fm_dim16_linearft/data_manifests/phase*_minibatch_summary.txt` に保存しています。生成済み学習バッチ本体は約234.66GBあるため、保存用パッケージにはコピーしていません。

### 学習手順

1. 7.7beta線形モデルをEGEV4に変換し、FMベクトルを0で初期化します。

```powershell
python model/20260716_7_7_fm_dim16_linearft/training_tools/output_egev4_7_7_fm_init_experiment.py `
  --input model/20260716_7_7_fm_dim16_linearft/models/source_7_7_beta_linear.egev2 `
  --output ignored/eval_experiments/20260710_eval77_fm_simdopt/eval77_zero_dim16.egev4 `
  --dim 16
```

2. `eval_optimizer_7_7_fm_stream_holdout_experiment.cpp` でFMベクトルだけを学習します。7.7betaの線形重みは固定します。

主な条件は `dim=16`、`epochs=12`、`lr=0.0002`、`loss=pseudo-huber`、`residual_clip=16`、`huber_delta=4`、`l2=0.00001`、`seed=20260710` です。実測学習時間は約22.08時間でした。

3. `eval_optimizer_7_7_fm_linear_finetune_experiment.cpp` でFMベクトルを固定し、線形重みだけを微調整します。

主な条件は `epochs=12`、`lr=0.00002`、`loss=pseudo-huber`、`residual_clip=16`、`huber_delta=4`、`linear_l2=0.00001`、`seed=20260713` です。実測学習時間は約10.36時間でした。

4. `merge_egev4_phases.cpp` でphaseごとの出力を結合し、最終的な `eval.egev4` を作ります。

学習ログは `model/20260716_7_7_fm_dim16_linearft/training_logs` に保存しています。生の対戦ログは `model/20260716_7_7_fm_dim16_linearft/battle_logs` に保存しています。



## データの変換

* ```Egaroucid/train_data/transcript/recordsX```内にf5d6形式の棋譜を収録する
  * 連番で収録する
* ```tools/generate_board_data```の```all_expand_transcript.py```あたりを実行して棋譜をボードデータに変換する
* ```data_translate.py```で```Egaroucid/train_data/board_data/recordsX```から```Egaroucid/train_data/bin_data/日付/フェーズ```内にデータを変換する
  * コマンドライン引数はないが、60フェーズに固定してある。
  * すべてのデータを一気に変換するようになっている
  * ```data_board_to_idx.cpp```をラップしてある
    * ```evaluation_definition.hpp```でインデックスの定義をしてある



## 学習

* ```eval_optimizer.py```で学習できる
  * コマンドライン引数は```[start_phase] [end_phase]```
  * ```opt_log.txt```に学習ログを出力する
  * ```eval_optimizer_phase.py```をラップしてある
    * 学習時間や学習率、学習に使うデータはここで設定する
    * ```eval_optimizer_cuda.cu```をラップしてある
      * ```evaluation_definition.hpp```でインデックスの定義をしてある
* 学習済みモデルは```trained```フォルダに保存される



## 出力

* ```output_egev.cpp```で出力可能
  * コマンドライン引数は``` [n_phases]```
  * ```trained/フェーズ.txt```を読んで、```trained/eval.egev```に出力する。



## その他

* ```count_n_games.py```でデータ数(対局数)をカウントできる
  * コマンドライン引数はなし
  * ```Egaroucid/train_data/transcript```内の指定された```recordsX```の```番号.txt```に書かれた棋譜を全部カウントする
  * ```train_data/board_data/log.txt```に対局数は記録してあるのでそれを見ると良いが。
* ```test_loss_wrapper.py```でegevファイルを使ってテストデータ(36番と38番)でテストできる
  * ```test_loss.cpp```をラップしてある
* ```plot_loss.py```でMAE/MSEをプロットできる
