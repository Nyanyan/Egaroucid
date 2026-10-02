# WTHOR human move evaluation

Run `run_random_wthor_blend_experiment.py N` to evaluate the same sampled WTHOR
positions with:

- Blended policies at alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.
- Egaroucid for Console at levels **1, 5, 9, 13, 17**. Level 21 is reported as alpha = 0.
- Edax at levels **1, 5, 9, 13, 17, 21**.

Human move data comes from `$POLICY_DATA_ROOT/board_split_increased/<dataset>/test`.
Select the dataset with `--dataset` (default: `WTHOR_1977_2024`). If
`POLICY_DATA_ROOT` is unset, the root is
`$EGAROUCID_DATA/20261002_policy_network_human_like_data`. To share the data
processing root explicitly:

```powershell
$env:POLICY_DATA_ROOT = Join-Path $env:EGAROUCID_DATA '20261002_policy_network_human_like_data'
```

For example, from this directory:

```powershell
python run_random_wthor_blend_experiment.py 10000 `
  --dataset WTHOR_1977_2024 `
  --weights ../10_train_policy_network/trained/model/policy_network_weights.bin `
  --workers 32 --egaroucid-threads 1 --edax-threads 1 `
  --output-dir output/human_match_WTHOR_1977_2024
```

`--workers` is the total limit for Egaroucid and Edax combined, and must be
at least 12 to start all engine/level combinations. For a 32-thread machine
with one search thread per process, use `--workers 32`. The prepared `test` directory is sampled directly
(`--data-split all`, sample seed 613); it is not split again. The policy-only
evaluator uses the same data source and `--dataset` option.
`--board-data-dir` overrides the environment and dataset selection. Use an
explicit `--data-split test` only when resplitting legacy unsplit input.
Edax defaults to `bin/versions/edax_4_6/wEdax-x86-64.exe`, with `data/eval.dat`
beside the executable. Override with `--edax-exe` and `--edax-eval-file`.
`--edax-workers` caps Edax within the shared `--workers` limit; it does not
add extra processes. Both engines start concurrently and reuse freed slots.
Progress reports combine blend, Egaroucid for Console, and Edax results in one
block per update, retaining finished engines in subsequent reports.
When migrating from the old sequential version, use a new output directory such
as `output/human_match_parallel`. Edax uses
`-n` for thread count, a 22-bit hash table, and disables book and pondering.
`--edax-timeout-sec` and `--edax-retries` control failed-search handling.

Each distinct board and side is searched once per level; repeated human samples
retain their multiplicity. Edax caches are separate from Egaroucid caches and
record executable/evaluation-data hashes and search settings. Changing experiment
conditions requires a different output directory. Historical results in
`RESULTS.md` describe their original conditions and are not recomputed.

Console level 21 has the same result as alpha = 0, so its duplicate row is omitted
from progress, final console output, CSV, and JSON comparison results. Level 21
is still searched for the blend. This display change preserves the search
identity: existing parallel-run caches can be reused with the same settings
and output directory.

## Agreement and likelihood

Top-1 and top-3 agreement retain the existing board-symmetry-aware definition:
a prediction matches if a symmetry preserving both players' stones maps the
observed human move to it. Engine baselines use their native `hint 3` output
order. Egaroucid level 21 supplies the all-legal-move hints required by the blend.

Likelihood answers: **What probability did the method assign to the exact move
the human actually played?** For every alpha, compute

```text
z(move) = alpha * network_logit(move)
          + (1 - alpha) * Egaroucid_level21_score(move) / score_temperature
p(move) = exp(z(move)) / sum(exp(z(legal_move)))
```

The implementation computes log-softmax stably over **all legal moves**, then
looks up the observed move. Symmetry-equivalent moves are not summed for this
metric. Thus, on an initial board with four equally likely moves, agreement is
100% but the observed move's likelihood is 25%.

The blend CSV and JSON include these fields for each alpha:

| Field | Meaning |
| --- | --- |
| `mean_likelihood` | Arithmetic mean of observed-move probabilities, from 0 to 1 |
| `mean_likelihood_percent` | The same mean expressed as a percentage |
| `mean_log_likelihood` | Mean natural log probability, in nats per position |
| `negative_log_likelihood` | Negative mean log probability; lower is better |

Every valid position sample has equal weight, including repeated boards with
different human moves. These averages are not the product of all probabilities.
Alpha 1 measures the policy network alone; alpha 0 measures the score-softmax
policy used by the blend. The standalone engine baseline rows only report
agreement: their top-3 hints are insufficient to normalize a full probability
distribution. Empty samples have null likelihood fields. The policy-only
evaluator also emits null log metrics when an observed probability is exactly
zero, rather than writing nonstandard JSON infinities.

## Outputs and validation

- `random_wthor_blend_summary.csv`: blend agreement and likelihood.
- `random_wthor_console_level_summary.csv`: Egaroucid agreement by level.
- `random_wthor_edax_level_summary.csv`: Edax agreement by level.
- `random_wthor_blend_summary.json`: all results, definitions, settings and timings.

The policy-only `evaluate_wthor_human_match.py` also reports likelihood in its
JSON and its overall / ten-move-bucket CSVs, after conditioning probabilities on
legal moves. Use `run_random_wthor_blend_experiment.py` for the comparison above.

The old top-N evaluator, shard runner/merger, duplicate analyzer, blend CLI and
legacy-only tests are archived locally in `legacy/`, which is ignored by Git.
The current `evaluate_wthor_blend_human_match.py` retains only shared board
readers, sampling and ranking helpers needed by active evaluations and launchers.
Use `run_random_wthor_blend_experiment.py` or `evaluate_wthor_human_match.py`
as the current command-line entry points.

```powershell
python -m unittest discover -s . -p 'test_*.py'
$env:WTHOR_EDAX_INTEGRATION = '1'
python -m unittest test_wthor_edax
```

The optional integration test invokes the local Edax binary at all six levels
on late-game positions with both colors to move.
