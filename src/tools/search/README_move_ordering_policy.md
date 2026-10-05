# Automatic move ordering evaluation

Ordinary search leaves continue to use the main evaluator. Each root search
iteration selects the following static ordering model and auxiliary leaf model:

| Root search | Static ordering | Auxiliary leaves | Auxiliary TT bounds |
|---|---|---|---|
| Midgame, depth below 27 | Main | Main | Ordinary behavior |
| Midgame, depth 27 or higher | Offset | Main | Ordinary behavior |
| Endgame, MPC 74% | Main | Main | Ordinary behavior |
| Endgame, MPC 88/93/98/99/99.9% | Offset | Offset | Shared, no model tags |
| Exact endgame, MPC 100% | Decay | Decay | Isolated from ordinary bounds |

The root classification stays constant as recursive remaining depth decreases.
Parallel subtrees inherit it. Iterative deepening and lazy SMP workers select
the policy for their own starting depth and MPC level.

Additional auxiliary leaves are enabled only for ordering/MPC probes starting
from an endgame context at depth 1 through 5. Deeper probes and normal main
search leaves retain the main evaluator. Nested probes and passes inherit the
active auxiliary model. Auxiliary probes run serially. In isolated mode they
may still read TT moves for ordering, but never read or write ordinary score
bounds; this matches the experimental `shallow_isolated` condition.

## Resources and overrides

Keep these files together in the directory containing the main evaluation file:

- `eval.egev2`: existing main model.
- `offset0_16.egev2`: primary bank (Offset).
- `decay_half16.egev2`: secondary bank (Decay).

The engine initialization used by Console and GUI loads both additional models
by default. When Console uses `-eval <path>`, they are resolved beside that
file, regardless of the working directory or executable directory. Missing or
invalid additional models fail initialization; Console returns exit status 1.

Console `-evalmo-auto-off` keeps ordinary evaluation without requiring the two
additional files. Explicit `-evalmo <path>` also disables automatic selection
and preserves the manual experiment options (`-evalmo-shallow-depth`,
`-evalmo-aux`, `-evalmo-shallow-tt`, `-evalmo-shallow-end-only`). A build with
`USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION=0` requires no additional models.

## Validation

Build the tests with the same flags as Console, for example:

```text
clang++ -O2 -mtune=native -march=native -pthread -std=c++20 src/tools/search/test_move_ordering_policy.cpp -o test_policy.exe
test_policy.exe bin/resources
clang++ -O2 -mtune=native -march=native -pthread -std=c++20 src/tools/search/test_move_ordering_model_init.cpp -o test_init.exe
test_init.exe bin/resources ignored/tmp/new-model-init-test
```

The policy test covers depth 26/27, every endgame selectivity, the five-ply
probe limit, model-specific static/MPC hints, ordinary main leaves, TT sharing
and isolation, scope restoration, passes, and parallel inheritance. It also
compares automatic and manual endgame searches at 74%, 88%, and 100%, checking
serial results and node counts plus parallel scores. Both the ordinary TT and
the persistent local endgame cache are cleared before serial comparisons.
The initialization test requires a fresh scratch directory and leaves its
temporary resource copies there.
