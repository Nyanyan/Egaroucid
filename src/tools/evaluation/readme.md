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