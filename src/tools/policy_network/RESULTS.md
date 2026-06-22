# Policy Network Results / ポリシーネットワーク結果

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

## Input Correction / 入力修正

The first implementation used fixed black/white inputs. That was wrong for the
board-data layout. The implementation was corrected to use the side-to-move
`player` bitboard and the `opponent` bitboard directly.

最初の実装は固定の黒/白入力でしたが、board data の形式としては誤りでした。手番側
`player` bitboard と相手側 `opponent` bitboard をそのまま使う実装へ修正しました。

All results below are from the corrected player/opponent input.

以下の結果はすべて修正後の player/opponent 入力によるものです。

## Hyper-Parameter Search / ハイパーパラメータ探索

All search runs used 500,000 sampled training positions and 100,000 sampled
validation positions from records 259-310. Activation was LeakyReLU with
`alpha=0.03`.

探索では records 259-310 から 500,000 train 局面、100,000 validation 局面を
サンプリングしました。activation は LeakyReLU `alpha=0.03` です。

| config | params | val loss | val top-1 | val top-3 | val top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 2.9386 | 16.196% | 38.793% | 53.529% |
| 96x3 | 37,216 | 2.7377 | 17.218% | 41.892% | 58.359% |
| 128x3 | 57,792 | 2.6093 | 17.954% | 43.397% | 60.562% |
| 96x4 | 46,528 | 2.8458 | 16.504% | 39.841% | 55.454% |

`128x3` was selected because it had the best validation accuracy while still
being small enough for fast C++ inference.

`128x3` は validation 精度が最も高く、C++ 推論でも十分軽いため採用しました。

## Final Training / 最終学習

Command:

コマンド:

```powershell
python -u src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_network\trained\playerop_final_issue613_128x3
```

Result:

結果:

| config | params | epochs | best epoch | val loss | val top-1 | val top-3 | val top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 24 | 24 | 2.3376 | 20.091% | 46.841% | 64.662% |

Final training metrics:

最終 epoch の train 指標:

- train top-1: 20.286%
- train top-3: 47.441%
- train top-5: 65.313%

Artifacts:

成果物:

- `src/tools/policy_network/trained/playerop_final_issue613_128x3/best_policy_network_weights.bin`
- `src/tools/policy_network/trained/playerop_final_issue613_128x3/best_model.h5`
- `src/tools/policy_network/trained/playerop_final_issue613_128x3/best_summary.json`

The `trained/` directory is ignored by git.

`trained/` ディレクトリは git 管理外です。

## WTHOR Human-Game Evaluation / WTHOR 人間棋譜評価

The evaluation used WTHOR human-game board data, stored in the current
training-data tree under the `records1` directory. The network output was
masked to legal moves before ranking. A hit means the actual human move is
within the top N legal moves. Generated evaluation file names use `wthor`.

評価には WTHOR 人間棋譜 board data を使いました。現在の training-data tree では
`records1` ディレクトリに格納されています。NN 出力を合法手だけに mask してから
順位付けし、実際の人間の手が top N 合法手以内に入れば hit としています。生成される
評価ファイル名には `wthor` を使います。

Command:

コマンド:

```powershell
python -u src\tools\policy_network\evaluate_policy_topn.py --model src\tools\policy_network\trained\playerop_final_issue613_128x3\best_model.h5 --weights src\tools\policy_network\trained\playerop_final_issue613_128x3\best_policy_network_weights.bin --batch-size 65536 --predict-batch-size 8192 --output-dir src\tools\policy_network\trained\playerop_wthor_eval --verbose
```

Evaluation summary:

評価サマリ:

- Positions / 評価局面数: 7,537,415
- Invalid policy records / 不正 policy レコード: 0
- Illegal-label records / 非合法ラベルレコード: 0
- Model / モデル: `src/tools/policy_network/trained/playerop_final_issue613_128x3/best_model.h5`

Overall top-N accuracy:

全体 top-N 一致率:

| top N | hits | positions | accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 5,203,262 | 7,537,415 | 69.032% |
| 2 | 6,318,466 | 7,537,415 | 83.828% |
| 3 | 6,875,000 | 7,537,415 | 91.212% |
| 4 | 7,251,830 | 7,537,415 | 96.211% |
| 5 | 7,425,169 | 7,537,415 | 98.511% |
| 8 | 7,527,098 | 7,537,415 | 99.863% |
| 10 | 7,535,397 | 7,537,415 | 99.973% |
| 16 | 7,537,411 | 7,537,415 | 99.9999% |

By phase:

phase 別:

| phase | positions | top-1 | top-3 | top-5 | top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opening 4-20 discs / 序盤 4-20 石 | 2,139,606 | 65.536% | 86.401% | 98.805% | 99.991% |
| midgame 21-44 discs / 中盤 21-44 石 | 3,020,105 | 61.913% | 90.110% | 97.478% | 99.941% |
| endgame 45-64 discs / 終盤 45-64 石 | 2,377,704 | 81.222% | 96.940% | 99.558% | 99.999% |

Generated evaluation artifacts:

生成された評価成果物:

- `src/tools/policy_network/trained/playerop_wthor_eval/wthor_topn_accuracy.csv`
- `src/tools/policy_network/trained/playerop_wthor_eval/wthor_topn_accuracy_by_phase.csv`
- `src/tools/policy_network/trained/playerop_wthor_eval/wthor_topn_accuracy.json`

## C++ Sample Checks / C++ サンプル確認

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Initial board:

初期局面:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\playerop_final_issue613_128x3\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| e6 | 0.303573 |
| d3 | 0.263853 |
| c4 | 0.246827 |
| f5 | 0.184177 |

After transcript `f5`:

棋譜 `f5` 後:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\playerop_final_issue613_128x3\best_policy_network_weights.bin --transcript f5 --top 10
```

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| d6 | 0.378855 |
| f6 | 0.319004 |
| f4 | 0.291286 |
