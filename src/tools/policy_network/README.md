# Policy Network

Related issue: #613

This tool trains a compact Othello policy network with `tensorflow.keras`.

Input:

- 64 black-disc bits
- 64 white-disc bits

Output:

- 64-way softmax policy distribution
- Coordinate mapping is the Egaroucid policy index: `a1 -> 63`, `h8 -> 0`

The training data is read from `$EGAROUCID_DATA/train_data/board_data/records259`
through `records310`. Each board record is 19 bytes:

1. `uint64` player-to-move bitboard
2. `uint64` opponent bitboard
3. `int8` player color (`0` black, `1` white)
4. `int8` policy
5. `int8` score

The learner converts the first two bitboards back to fixed black/white inputs
before training.

## Training

Small smoke test:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 16x1 --epochs 1 --max-train-samples 2000 --max-val-samples 512
```

Hyper-parameter search example:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 48x3,64x3,80x3,64x4 --epochs 10 --patience 3 --max-train-samples 300000 --max-val-samples 50000 --batch-size 4096
```

Final training example:

```powershell
python src\tools\policy_network\train_policy_network.py --configs 128x3 --epochs 24 --patience 6 --max-train-samples 2000000 --max-val-samples 200000 --batch-size 8192
```

Artifacts are written under `src/tools/policy_network/trained/<timestamp>/`.
That directory is ignored by git. The C++ sample uses
`best_policy_network_weights.bin`.

The current selected architecture is `128x3` with 57,792 parameters. See
`RESULTS.md` for the search table and final training result.

## C++ Sample

Build:

```powershell
g++ -std=c++17 -O3 src\tools\policy_network\policy_network_sample.cpp -o src\tools\policy_network\policy_network_sample.exe
```

Show policy distribution from a transcript:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\<run>\best_policy_network_weights.bin --transcript f5d6c3 --top 10
```

Show policy distribution from a board string:

```powershell
src\tools\policy_network\policy_network_sample.exe src\tools\policy_network\trained\<run>\best_policy_network_weights.bin --board ---------------------------OX------XO---------------------------X --top 10
```

`BOARD65` is 64 board characters plus a side-to-move character. `X`, `0`, and
`*` mean black; `O` and `1` mean white; `-` and `.` mean empty.
