# Policy Network Results / 方策ネットワーク結果

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
- Discovered training files / 検出した学習ファイル数: 82
- Total training records / 学習総レコード数: 3,139,338,789

## Hyper-parameter Search / ハイパーパラメータ探索

All search runs used 500,000 sampled training positions and 100,000 sampled
validation positions from records 259-310. Activation was LeakyReLU with
`alpha=0.03`.

探索では records 259-310 から 500,000 train 局面、100,000 validation 局面を
サンプリングしました。activation は LeakyReLU `alpha=0.03` です。

| config | params | best epoch | val loss | val top-1 | val top-3 | val top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 48x3 | 14,032 | 12 | 3.2939 | 11.486% | 28.676% | 40.366% |
| 64x3 | 20,736 | 12 | 3.1455 | 12.766% | 31.404% | 44.292% |
| 80x3 | 28,464 | 11 | 3.0581 | 13.214% | 32.759% | 46.315% |
| 96x3 | 37,216 | 12 | 2.9468 | 13.723% | 33.891% | 47.978% |
| 64x4 | 24,896 | 12 | 3.2261 | 12.141% | 30.022% | 41.983% |
| 112x3 | 46,992 | 12 | 2.8835 | 14.171% | 34.792% | 48.893% |
| 128x3 | 57,792 | 11 | 2.8269 | 14.457% | 35.737% | 50.109% |
| 96x4 | 46,528 | 12 | 3.0433 | 13.231% | 33.018% | 46.511% |

The extra fourth hidden layer did not help at similar parameter counts.
`128x3` was selected for the final model because it gave the best accuracy
while still being small enough for fast C++ inference.

同程度の parameter 数では 4 層目を足しても改善しませんでした。精度が最も高く、
C++ 推論でも十分軽い `128x3` を採用しました。

## Final Training / 最終学習

Command:

コマンド:

```powershell
python -u src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_network\trained\final_issue613_128x3
```

Result:

結果:

| config | params | epochs | best epoch | val loss | val top-1 | val top-3 | val top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 24 | 23 | 2.5817 | 16.984% | 40.129% | 55.331% |

Artifacts:

成果物:

- `src/tools/policy_network/trained/final_issue613_128x3/best_policy_network_weights.bin`
- `src/tools/policy_network/trained/final_issue613_128x3/best_model.h5`
- `src/tools/policy_network/trained/final_issue613_128x3/best_summary.json`

The `trained/` directory is ignored by git.

`trained/` ディレクトリは git 管理外です。

## Records1 Human-Game Evaluation / records1 人間棋譜評価

The evaluation used `records1` board data generated from WTHOR human games.
The network output was masked to legal moves before ranking. A hit means the
actual human move is within the top N legal moves.

評価には WTHOR 人間棋譜から生成された `records1` board data を使いました。
NN 出力は合法手だけに mask してから順位付けしました。一致とは、実際の人間手が
top N 合法手以内に入ったことを意味します。

Command:

コマンド:

```powershell
python -u src\tools\policy_network\evaluate_policy_topn.py --batch-size 65536 --predict-batch-size 8192 --output-dir src\tools\policy_network\trained\records1_eval --verbose
```

Evaluation summary:

評価サマリ:

- Positions / 評価局面数: 7,537,415
- Invalid policy records / 不正 policy レコード: 0
- Illegal-label records / 非合法ラベルレコード: 0
- Model / モデル: `src/tools/policy_network/trained/final_issue613_128x3/best_model.h5`

Overall top-N accuracy:

全体 top-N 一致率:

| top N | hits | positions | accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 4,489,003 | 7,537,415 | 59.556% |
| 2 | 5,894,634 | 7,537,415 | 78.205% |
| 3 | 6,607,201 | 7,537,415 | 87.659% |
| 4 | 7,118,978 | 7,537,415 | 94.449% |
| 5 | 7,327,372 | 7,537,415 | 97.213% |
| 8 | 7,515,103 | 7,537,415 | 99.704% |
| 10 | 7,532,797 | 7,537,415 | 99.939% |
| 16 | 7,537,402 | 7,537,415 | 99.9998% |

By phase:

phase 別:

| phase | positions | top-1 | top-3 | top-5 | top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opening 4-20 discs / 序盤 4-20 石 | 2,139,606 | 57.607% | 83.886% | 97.476% | 99.970% |
| midgame 21-44 discs / 中盤 21-44 石 | 3,020,105 | 48.568% | 84.581% | 95.516% | 99.870% |
| endgame 45-64 discs / 終盤 45-64 石 | 2,377,704 | 75.267% | 94.963% | 99.133% | 99.998% |

Generated evaluation artifacts:

生成された評価成果物:

- `src/tools/policy_network/trained/records1_eval/records1_topn_accuracy.csv`
- `src/tools/policy_network/trained/records1_eval/records1_topn_accuracy_by_phase.csv`
- `src/tools/policy_network/trained/records1_eval/records1_topn_accuracy.json`

## C++ Sample Checks / C++ サンプル確認

Build:

ビルド:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Initial board:

初期局面:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\final_issue613_128x3\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| e6 | 0.286479 |
| c4 | 0.250458 |
| d3 | 0.238813 |
| f5 | 0.223494 |

After transcript `f5`:

棋譜 `f5` 後:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\final_issue613_128x3\best_policy_network_weights.bin --transcript f5 --top 10
```

Top legal moves:

上位合法手:

| move | probability |
| --- | ---: |
| f4 | 0.363545 |
| f6 | 0.357208 |
| d6 | 0.277746 |
