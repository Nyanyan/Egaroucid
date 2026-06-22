# Policy-Value Network Results / ポリシー・バリューネットワーク結果

Related issue: #613

関連 issue: #613

Run date: 2026-06-22

実行日: 2026-06-22

Environment:

環境:

- TensorFlow 2.10.0
- GPU: NVIDIA GeForce RTX 3090
- Data root / データ root: `$EGAROUCID_DATA/train_data/board_data`
- Training records / 学習 records: `records259` through `records310`

## Method / 方法

The network uses the same corrected side-to-move player/opponent input as the
policy network. The value target is board-data `score / 64`, from the
player-to-move perspective.

入力は policy network と同じ修正版の手番側/相手側 bitboard です。value 教師は
board data の `score / 64` で、手番側目線として使っています。

The architecture follows the AlphaZero-style shared trunk plus policy/value
heads. Reference: Silver et al., "Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm", arXiv:1712.01815,
https://arxiv.org/abs/1712.01815.

構造は AlphaZero 風の共有 trunk + policy/value 2 head です。参考: Silver ら
"Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
Algorithm", arXiv:1712.01815。

## Hyper-Parameter Search / ハイパーパラメータ探索

All search runs used 500,000 sampled training positions and 100,000 sampled
validation positions from records 259-310. Activation was LeakyReLU with
`alpha=0.03`.

探索では records 259-310 から 500,000 train 局面、100,000 validation 局面を
サンプリングしました。activation は LeakyReLU `alpha=0.03` です。

| config | params | value loss weight | val loss | val policy top-1 | val policy top-3 | val policy top-5 | val value MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pv96w01 | 37,313 | 0.10 | 2.7560 | 17.337% | 41.913% | 58.131% | 0.2812 |
| pv96w025 | 37,313 | 0.25 | 2.7721 | 17.323% | 41.850% | 58.092% | 0.2704 |
| pv128w01 | 57,921 | 0.10 | 2.6195 | 18.075% | 43.476% | 60.607% | 0.2766 |
| pv128w025 | 57,921 | 0.25 | 2.6652 | 17.793% | 43.001% | 59.817% | 0.2690 |
| pv128w05 | 57,921 | 0.50 | 2.6938 | 17.757% | 43.108% | 60.066% | 0.2651 |
| pv96d4w025 | 46,625 | 0.25 | 2.8902 | 16.556% | 39.655% | 55.287% | 0.2782 |

`pv128w01` was selected because it preserved the strongest policy accuracy
while keeping value error close to the heavier value-loss settings.

`pv128w01` は policy 精度が最も良く、value 誤差も value loss を重くした設定に
近かったため採用しました。

## Final Training / 最終学習

Command:

コマンド:

```powershell
python -u src\tools\policy_value_network\train_policy_value_network.py --configs pv128w01:128:3:0.1 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_value_network\trained\final_issue613_pv128w01
```

Result:

結果:

| config | params | epochs | best epoch | val loss | val policy top-1 | val policy top-3 | val policy top-5 | val value MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pv128w01 | 57,921 | 24 | 22 | 2.3437 | 20.165% | 47.061% | 65.119% | 0.2648 |

The value MAE is measured on normalized `score / 64`; in disc-difference units
this is about 16.95 discs.

value MAE は正規化後の `score / 64` 上の値です。石差換算では約 16.95 石です。

Final training metrics:

最終 epoch の train 指標:

- train policy top-1: 20.371%
- train policy top-3: 47.618%
- train policy top-5: 65.502%
- train value MAE: 0.2666

Artifacts:

成果物:

- `src/tools/policy_value_network/trained/final_issue613_pv128w01/best_policy_value_network_weights.bin`
- `src/tools/policy_value_network/trained/final_issue613_pv128w01/best_model.h5`
- `src/tools/policy_value_network/trained/final_issue613_pv128w01/best_summary.json`

The `trained/` directory is ignored by git.

`trained/` ディレクトリは git 管理外です。

## WTHOR Human-Game Evaluation / WTHOR 人間棋譜評価

The evaluation used WTHOR human-game board data, stored in the current
training-data tree under the `records1` directory. Generated evaluation file
names use `wthor`. The policy head was masked to legal moves before ranking.
The value head was compared with board-data `score / 64`.

評価には WTHOR 人間棋譜 board data を使いました。現在の training-data tree では
`records1` ディレクトリに格納されています。生成される評価ファイル名には `wthor`
を使います。policy head は合法手だけに mask してから順位付けし、value head は
board data の `score / 64` と比較しました。

Command:

コマンド:

```powershell
python -u src\tools\policy_value_network\evaluate_wthor_policy_value.py --model src\tools\policy_value_network\trained\missing_model_for_binary_eval.h5 --weights src\tools\policy_value_network\trained\final_issue613_pv128w01\best_policy_value_network_weights.bin --batch-size 65536 --predict-batch-size 8192 --output-dir src\tools\policy_value_network\trained\wthor_policy_value_eval --verbose
```

Evaluation summary:

評価サマリ:

- Positions / 評価局面数: 7,537,415
- Invalid policy records / 不正 policy レコード: 0
- Illegal-label records / 非合法ラベルレコード: 0
- Model / モデル: `src/tools/policy_value_network/trained/final_issue613_pv128w01/best_policy_value_network_weights.bin`

Policy top-N accuracy:

policy top-N 一致率:

| top N | hits | positions | accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 5,117,530 | 7,537,415 | 67.895% |
| 2 | 6,283,926 | 7,537,415 | 83.370% |
| 3 | 6,874,876 | 7,537,415 | 91.210% |
| 4 | 7,258,119 | 7,537,415 | 96.295% |
| 5 | 7,407,968 | 7,537,415 | 98.283% |
| 8 | 7,525,375 | 7,537,415 | 99.840% |
| 10 | 7,535,037 | 7,537,415 | 99.968% |
| 16 | 7,537,410 | 7,537,415 | 99.9999% |

Policy top-N by phase:

phase 別 policy top-N:

| phase | positions | top-1 | top-3 | top-5 | top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opening 4-20 discs / 序盤 4-20 石 | 2,139,606 | 63.266% | 87.325% | 98.417% | 99.993% |
| midgame 21-44 discs / 中盤 21-44 石 | 3,020,105 | 60.884% | 89.568% | 97.215% | 99.928% |
| endgame 45-64 discs / 終盤 45-64 石 | 2,377,704 | 80.966% | 96.791% | 99.518% | 99.999% |

Value metrics:

value 指標:

| scope | positions | MAE | RMSE | mean error | disc MAE | disc RMSE | sign accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall / 全体 | 7,537,415 | 0.2811 | 0.3552 | -0.0239 | 17.994 | 22.730 | 59.241% |
| opening 4-20 discs / 序盤 4-20 石 | 2,139,606 | 0.2964 | 0.3786 | -0.0326 | 18.971 | 24.233 | 51.649% |
| midgame 21-44 discs / 中盤 21-44 石 | 3,020,105 | 0.2828 | 0.3576 | -0.0265 | 18.098 | 22.889 | 58.820% |
| endgame 45-64 discs / 終盤 45-64 石 | 2,377,704 | 0.2653 | 0.3292 | -0.0127 | 16.981 | 21.070 | 66.610% |

Generated evaluation artifacts:

生成された評価成果物:

- `src/tools/policy_value_network/trained/wthor_policy_value_eval/wthor_policy_value_evaluation.json`
- `src/tools/policy_value_network/trained/wthor_policy_value_eval/wthor_policy_value_topn_accuracy.csv`
- `src/tools/policy_value_network/trained/wthor_policy_value_eval/wthor_policy_value_topn_accuracy_by_phase.csv`
- `src/tools/policy_value_network/trained/wthor_policy_value_eval/wthor_policy_value_metrics.csv`
- `src/tools/policy_value_network/trained/wthor_policy_value_eval/wthor_policy_value_metrics_by_phase.csv`

## C++ Sample Checks / C++ サンプル確認

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_value_network\policy_value_network_sample.cpp -o src\tools\policy_value_network\policy_value_network_sample.exe
```

Initial board:

初期局面:

```powershell
src\tools\policy_value_network\policy_value_network_sample.exe src\tools\policy_value_network\trained\final_issue613_pv128w01\best_policy_value_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 8
```

- value from side to move / 手番側目線 value: -0.019981
- disc-diff estimate / 石差推定: -1.278795

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| e6 | 0.292917 |
| d3 | 0.265648 |
| c4 | 0.254701 |
| f5 | 0.186311 |

After transcript `f5d6c3`:

棋譜 `f5d6c3` 後:

```powershell
src\tools\policy_value_network\policy_value_network_sample.exe src\tools\policy_value_network\trained\final_issue613_pv128w01\best_policy_value_network_weights.bin --transcript f5d6c3 --top 8
```

- side to move / 手番: white
- value from side to move / 手番側目線 value: -0.019423
- disc-diff estimate / 石差推定: -1.243053

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| d3 | 0.276506 |
| f3 | 0.275595 |
| g5 | 0.188193 |
| f4 | 0.160180 |
