# WTHOR human move evaluation

Run `run_random_wthor_blend_experiment.py N` to evaluate the same sampled WTHOR
positions with:

- Blended policies at alpha = 0, 0.2, 0.4, 0.6, 0.8, 1.
- Egaroucid at levels **1, 5, 9, 13, 17, 21**.
- Edax at levels **1, 5, 9, 13, 17, 21**.

For example, from this directory:

```powershell
python run_random_wthor_blend_experiment.py 10000 `
  --board-data-dir "$env:EGAROUCID_DATA/train_data/board_data/records1" `
  --weights ../10_train_policy_network/trained/model/policy_network_weights.bin `
  --workers 6 --egaroucid-threads 1 --edax-threads 1 `
  --output-dir output/human_match
```

`--workers` must be at least 6. The default data split is `test` (80/10/10
position split, seed 613). Use `--data-split all` to sample the whole input.
Edax defaults to `bin/versions/edax_4_6/wEdax-x86-64.exe`, with `data/eval.dat`
beside the executable. Override with `--edax-exe` and `--edax-eval-file`.
`--edax-workers` defaults to the Egaroucid process budget. The two engine phases
run sequentially, each using a bounded pool of persistent processes. Edax uses
`-n` for thread count, a 22-bit hash table, and disables book and pondering.
`--edax-timeout-sec` and `--edax-retries` control failed-search handling.

Each distinct board and side is searched once per level; repeated human samples
retain their multiplicity. Edax caches are separate from Egaroucid caches and
record executable/evaluation-data hashes and search settings. Changing experiment
conditions requires a different output directory. Historical results in
`RESULTS.md` describe their original conditions and are not recomputed.

## Agreement and likelihood

Top-1 and top-3 agreement retain the existing board-symmetry-aware definition:
a prediction matches if a symmetry preserving both players' stones maps the
observed human move to it. Engine baselines use their native `hint 3` output
order. Egaroucid level 21 reuses the all-legal-move hints required by the blend.

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
legal moves. The older shard-oriented blend scripts keep their existing output
format; use `run_random_wthor_blend_experiment.py` for the comparison above.

```powershell
python -m unittest discover -s . -p 'test_*.py'
$env:WTHOR_EDAX_INTEGRATION = '1'
python -m unittest test_wthor_edax
```

The optional integration test invokes the local Edax binary at all six levels
on late-game positions with both colors to move.
