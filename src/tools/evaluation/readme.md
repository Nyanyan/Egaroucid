# Evaluation

評価関数を作る



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
* FM版は```eval_optimizer_cuda_fm.cu```で学習できる
  * 出力は```trained/[phase]_fm.txt```（先頭に線形項、続いてFM因子）
  * 出現回数は```trained/weight_[phase]_fm.txt```に出力される
  * FMバイナリ出力は```trained/eval_fm.egev3```（先頭に作成日時YYYYMMDDhhmmssを格納）
* 学習済みモデルは```trained```フォルダに保存される



## 更新式メモ（非FM / FM）

* 損失はどちらもサンプルごとに ```L=(y-ŷ)^2```、残差 ```e=y-ŷ```

### 非FM（線形和）

* 予測:
  * ```ŷ = Σ_i w_i x_i```
* 勾配:
  * ```∂L/∂w_i = -2 e x_i```
* 実装上は「足し込み更新」に合わせて ```g_i = 2 e x_i``` を集計し、Adamで
  * ```m_i ← β1 m_i + (1-β1) g_i```
  * ```v_i ← β2 v_i + (1-β2) g_i^2```
  * ```w_i ← w_i + lr_t * m_i / (sqrt(v_i)+ε)```

### FM（2次相互作用あり）

* 予測:
  * ```ŷ = Σ_i w_i x_i + 1/2 Σ_f [ (Σ_i v_{i,f}x_i)^2 - Σ_i (v_{i,f}x_i)^2 ]```
* 勾配:
  * ```∂L/∂w_i = -2 e x_i```
  * ```∂ŷ/∂v_{i,f} = x_i (Σ_j v_{j,f}x_j - v_{i,f}x_i)```
  * ```∂L/∂v_{i,f} = -2 e x_i (Σ_j v_{j,f}x_j - v_{i,f}x_i)```
* 実装上の更新方向:
  * ```g_i = 2 e x_i```
  * ```g_{i,f} = 2 e x_i (Σ_j v_{j,f}x_j - v_{i,f}x_i)```
* Adamは線形項とFM因子で別々の ```m```, ```v``` を持って同様に更新



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