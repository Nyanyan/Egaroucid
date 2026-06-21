# Policy Network Results

Related issue: #613

Run date: 2026-06-22

Environment:

- TensorFlow 2.10.0
- GPU: NVIDIA GeForce RTX 3090
- Data root: `$EGAROUCID_DATA/train_data/board_data`
- Records: `records259` through `records310`
- Discovered files: 82
- Total records: 3,139,338,789

## Hyper-parameter Search

All search runs used 500,000 sampled training positions and 100,000 sampled
validation positions from records 259-310. Activation was LeakyReLU with
`alpha=0.03`.

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

## Final Training

Command:

```powershell
python -u src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192 --output-dir src\tools\policy_network\trained\final_issue613_128x3
```

Result:

| config | params | epochs | best epoch | val loss | val top-1 | val top-3 | val top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x3 | 57,792 | 24 | 23 | 2.5817 | 16.984% | 40.129% | 55.331% |

Artifacts:

- `src/tools/policy_network/trained/final_issue613_128x3/best_policy_network_weights.bin`
- `src/tools/policy_network/trained/final_issue613_128x3/best_model.h5`
- `src/tools/policy_network/trained/final_issue613_128x3/best_summary.json`

The `trained/` directory is ignored by git.

## C++ Sample Checks

Build:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Initial board:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\final_issue613_128x3\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

Top legal moves:

| move | probability |
| --- | ---: |
| e6 | 0.286479 |
| c4 | 0.250458 |
| d3 | 0.238813 |
| f5 | 0.223494 |

After transcript `f5`:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\final_issue613_128x3\best_policy_network_weights.bin --transcript f5 --top 10
```

Top legal moves:

| move | probability |
| --- | ---: |
| f4 | 0.363545 |
| f6 | 0.357208 |
| d6 | 0.277746 |
