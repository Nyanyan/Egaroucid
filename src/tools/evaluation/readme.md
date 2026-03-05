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


* 損失はいずれもサンプルごとに $L = (y - \hat{y})^2$、残差 $e = y - \hat{y}$

### 非FM（線形和）


* 予測: $\hat{y} = \sum_i w_i x_i$
* 勾配: $\frac{\partial L}{\partial w_i} = -2 e x_i$
* 実装上は「足し込み更新」に合わせて $g_i = 2 e x_i$ を集計し、Adamで
  $m_i \leftarrow \beta_1 m_i + (1-\beta_1) g_i$
  $v_i \leftarrow \beta_2 v_i + (1-\beta_2) g_i^2$
  $w_i \leftarrow w_i + \mathrm{lr}_t \cdot \frac{m_i}{\sqrt{v_i}+\varepsilon}$

### FM（2次相互作用あり）


* 予測: $\hat{y} = \sum_i w_i x_i + \frac{1}{2} \sum_f \left[ \left( \sum_i v_{i,f} x_i \right)^2 - \sum_i (v_{i,f} x_i)^2 \right ]$
* 勾配: $\frac{\partial L}{\partial w_i} = -2 e x_i$
  $\frac{\partial \hat{y}}{\partial v_{i,f}} = x_i \left( \sum_j v_{j,f} x_j - v_{i,f} x_i \right)$
  $\frac{\partial L}{\partial v_{i,f}} = -2 e x_i \left( \sum_j v_{j,f} x_j - v_{i,f} x_i \right)$
* 実装上の更新方向: $g_i = 2 e x_i$
  $g_{i,f} = 2 e x_i \left( \sum_j v_{j,f} x_j - v_{i,f} x_i \right)$
* Adamは線形項とFM因子で別々の $m$, $v$ を持って同様に更新



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