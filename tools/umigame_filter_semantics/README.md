# Umigame Filter Semantics Regression

This directory contains regression checks for Umigame value filtering.

The checks cover two separate contracts:

- `Errors per Move` is a recursive search condition. It keeps child moves whose
  mover-perspective book value is within the configured loss from the local best
  child at that node.
- `Max Allowed Eval` is a GUI display filter. It decides whether a candidate
  move is displayed, using the candidate's Black-perspective book value. It must
  not prune recursive Umigame search.

## Source-Level Semantics Check

Run this lightweight check after touching the menu mapping or Umigame filtering
code:

```powershell
python ..\tools\umigame_filter_semantics\check_umigame_filters.py
```

It verifies the relevant source contracts and an independent small-tree model of
the filtering rules. It does not parse a real `.egbk3` book.

## Real-Book Regression

Build and run the bounded real-book verifier:

```powershell
python ..\tools\umigame_filter_semantics\run_umigame_regression.py
```

The runner builds `UmigameRegression.vcxproj`, then starts the verifier from the
`bin` directory using `bin\resources\book.egbk3`.

Useful options:

- `--depths 8,12,20,60`
- `--errors 0,2,4`
- `--display inf:inf,0:0,4:2,2:4`
- `--node-limit 100000`
- `--fail-limit 10`
- `--parallel-shards 16`
- `--threads 1`
- `--no-build`

Shards split the BFS node ordinal space by modulo. Each shard checks every
configured depth, error, and display-filter combination for its assigned node
subset; shards are not assigned by depth.

## Full Regression Command Used For PR #617

```powershell
python ..\tools\umigame_filter_semantics\run_umigame_regression.py --no-build --node-limit 1600000 --fail-limit 1 --parallel-shards 16 --threads 1
```

The full run exhausted the reachable test frontier before reaching the node
limit:

- `visited_nodes=1285627`
- `checked_node_cases=15427524`
- `checked_moves=14997884`
- `failures=0`
- `stop_reason=exhausted_reachable_test_frontier`
