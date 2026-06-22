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
