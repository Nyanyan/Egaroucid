# 2026-07-16 Grouped/Shared FM Initial Implementation Report

## Goal

Add an experimental evaluator for the 7.7 beta + FM dim16 linearft model where the linear weights remain phase-specific for all 60 phases, while FM vectors are shared by a small number of phase groups or by all phases.

## Implementation

- Added a new EGEV version 10 grouped-FM file format.
- Version 10 stores 60 phase-specific linear tables, FM vector tables for `fm_group_count` groups, `phase_to_fm_group[60]`, group count, dim, and scales.
- Added `EVALUATE_EXPERIMENT_7_7_FM_GROUPED` and `EVALUATE_EXPERIMENT_7_7_FM_SHARED` routing in `evaluate.hpp`.
- Added thin hooks to the existing fast evaluator so the common loader/scorer can be swapped without duplicating the large feature-definition headers.
- Added `evaluate_experiment_7_7_fm_grouped_common.hpp`, which memory maps version 10 files and reuses the existing fast active-id enumeration path.
- Added `src/tools/evaluation/util/convert_eval77_fm_egev4_to_grouped_fm.cpp`.

## Generated Initial Models

Input:

- `bin/resources/eval.egev4`
- version 8
- phases 60
- params/phase 944849
- dim 16

Outputs:

- `model/eval77_fm_grouped7_init/eval_grouped7.egev10`
  - mode grouped7
  - fm_groups 7
  - output_bytes 219205360
- `model/eval77_fm_shared_init/eval_shared.egev10`
  - mode shared
  - fm_groups 1
  - output_bytes 128499856

The converter averages real-valued FM vectors inside each group after applying each phase vector scale, then requantizes the group vector table.

## Verification

Build checks:

- Converter: passed
- Default fast SIMD build: passed
- Grouped SIMD build: passed
- Grouped Generic build: passed
- Shared SIMD build: passed
- Shared Generic build: passed

Short runtime checks:

- grouped7 SIMD: `midgame_test.txt` level 1 solve passed
- grouped7 Generic: matched SIMD moves, scores, and node counts
- shared SIMD: `midgame_test.txt` level 1 solve passed
- shared Generic: matched SIMD moves, scores, and node counts

The compiler warnings were the existing unused `future.get()` return-value warnings. Runtime printed a missing `hash16.eghs` warning; the engine says this can be ignored.

## Not Done Yet

- Speed benchmark for grouped/shared.
- Match checks at level 5 and level 10 or higher.
- Retraining grouped/shared models.
- FM sum/squared-sum differential cache in `Eval_search`.

## Next Steps

1. Benchmark grouped7 against current EGEV4 fast on the same machine and settings.
2. Run grouped7 matches at level 1, 5, and 10.
3. Treat shared-fm as secondary until grouped7 strength is understood.
4. Add a small direct score-consistency utility for SIMD/Generic position-by-position checks.
5. If grouped7 is promising, implement the `Eval_search` FM differential cache.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 1

### Midgame Level 23 / 28 Threads

Settings:

- `bin/problem/midgame_test.txt`
- `-l 23 -nobook -thread 28 -hash 25`

Results:

| Evaluator | nodes | time | NPS | Note |
|---|---:|---:|---:|---|
| Current EGEV4 fast | 1,662,079,937 | 26.31s | 63.17M | baseline |
| grouped7 initial | 1,225,806,115 | 20.786s | 58.97M | about 6.6% lower NPS than baseline |
| shared initial | did not finish | timed out at 300s | - | reached 22/32 positions; deprioritized |
| grouped7 naive diff cache | 1,745,395,822 | 259.427s | 6.73M | cache update on every move was much too expensive |
| grouped7 2-pass no-prefetch | 1,680,727,519 | 35.365s | 47.53M | no search-NPS improvement |

Logs:

- `speed_logs/20260716_mid_l23_evalfast.txt`
- `speed_logs/20260716_mid_l23_grouped7.txt`
- `speed_logs/20260716_mid_l23_shared.txt`
- `speed_logs/20260716_mid_l23_grouped7_cache.txt`
- `speed_logs/20260716_mid_l23_grouped7_twopass_nopf.txt`

### Isolated Score Speed

Updated `eval77_fm_simdopt_score_speed_check.cpp` so it recognizes fast/grouped/shared macros and measures the same fast fused scoring path used by the engine.

Settings:

- iterations 5,000,000
- case_count 65,536
- phase 0-59

| Evaluator | selected eval/s | Note |
|---|---:|---|
| Current EGEV4 fast fused | 1.724M | `20260716_score_speed_fast_allphase_fused.txt` |
| grouped7 initial | 1.153M | `20260716_score_speed_grouped7_allphase_fused.txt` |
| grouped7 2-pass no-prefetch | 1.172M | `20260716_score_speed_grouped7_twopass_nopf_allphase.txt` |

The 2-pass no-prefetch scorer improved isolated grouped7 score speed by about 1.7%, but it did not produce a useful search-NPS gain.

### Differential Cache Experiment

Tried storing FM sum/sum_sq in `Eval_search` and updating it incrementally inside the same FM group.

Consistency:

- `games=2000`
- `checked_positions=120304`
- `mismatches=0`

The values were correct, but `eval_move()` runs at every search node, so updating the FM cache there paid a large cost on internal nodes. NPS dropped to 6.73M. The cache path is now isolated behind the explicit `EVALUATE_EXPERIMENT_7_7_FM_GROUPED_DIFF_CACHE` macro and is not enabled for normal grouped/shared builds.

### Judgment

- grouped7 without a better cache strategy is slower than current EGEV4 fast.
- the naive differential cache is correct but far too slow.
- shared-fm timed out in the level 23 speed check and is low priority for now.
- next work should either avoid heavy per-move cache updates or revisit the grouped file/layout strategy.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 2

### EGEV4 Duplicate Layout

Added a converter that keeps the EGEV10 grouped7 semantics but writes an EGEV4 version 8 file where each phase duplicates the FM vector table of its mapped group.

- New tool: `src/tools/evaluation/util/convert_eval77_grouped_fm_to_egev4.cpp`
- Generated model: `model/eval77_fm_grouped7_dup_egev4/eval_grouped7_dup.egev4`
- Output size: 1,020,437,434 bytes

The level 1 moves, scores, and node counts matched EGEV10 grouped7.

Isolated score speed:

| Evaluator | selected eval/s |
|---|---:|
| Current EGEV4 fast fused | 1.724M |
| grouped7 EGEV10 split | 1.172M |
| grouped7 EGEV4 duplicate | 1.661M |

This shows that the main grouped7 slowdown is likely caused by the split/de-duplicated layout, not by the grouped FM vectors themselves.

### Lazy Cache Experiment

Tried updating FM sum/sum_sq lazily in `mid_evaluate_diff()` instead of updating it at every `eval_move()`.

Consistency:

- `games=2000`
- `checked_positions=120304`
- `mismatches=0`

Midgame level 23 / 28 threads:

| Evaluator | nodes | time | NPS |
|---|---:|---:|---:|
| grouped7 lazy cache | 1,990,607,689 | 158.164s | 12.59M |

The values were correct, but many same-slot evaluations differed by many feature IDs, so the cost of 65 ID comparisons and vector add/sub work was still too high. Do not adopt this path.

### Materialized Interleaved Layout

Added an experiment that reads the compact EGEV10 grouped file, then materializes an EGEV4-like interleaved phase payload in memory at load time.

- Macro: `EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZED`
- Internally this enables `EVALUATE_EXPERIMENT_7_7_FM_GROUPED` and `EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE`.
- The on-disk file stays compact, around 219MB.
- Runtime memory additionally holds an EGEV4-duplicate-sized payload.

Consistency:

- `games=1000`
- `checked_positions=60147`
- `mismatches=0`
- Generic materialized level 1 output also matched SIMD.

Isolated score speed:

| Evaluator | selected eval/s |
|---|---:|
| grouped7 EGEV10 split | 1.172M |
| grouped7 EGEV10 materialized | 1.666M |
| grouped7 EGEV4 duplicate | 1.661M |
| Current EGEV4 fast fused | 1.724M |

Materialization preserves the compact grouped file while recovering EGEV4-duplicate-like scoring speed. This is the most promising improvement from this cycle.

Midgame level 23 / 28 threads:

| Evaluator | nodes | time | NPS |
|---|---:|---:|---:|
| grouped7 materialized | 1,193,178,984 | 25.46s | 46.86M |

Search NPS is noisy because the tree differs and the run is parallel. The isolated score-speed result is clearer. Next cycle should focus on materialized grouped7 and start level 5/10 match checks or small match runs.

### Next Judgment

- There is little reason to use the split grouped layout for speed.
- If compact files are preferred, the materialized loader is promising.
- If memory must stay low, split layout needs a different low-cost improvement, but it is currently slower.
- The next cycle should center on materialized grouped7 and strength checks.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 3

### Materialized Macro Coverage

Rebuilt the materialized consistency and score-speed tools from the current working tree and found that
`EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZED` was not recognized by the utility-level default macro guard.
That caused the utilities to also define `EVALUATE_EXPERIMENT_7_7_FM_SIMDOPT`, so `evaluate.hpp` selected the
normal EGEV4 SIMDOPT loader before the grouped loader.

Fixed both utility guards:

- `eval77_fm_simdopt_consistency_check.cpp`
- `eval77_fm_simdopt_score_speed_check.cpp`

The materialized alias now selects the grouped loader in the tools as well as in the console engine.

### Rechecks After The Fix

Consistency:

- `games=500`
- `checked_positions=30084`
- `mismatches=0`
- loader log showed `layout grouped-materialized-interleaved`

Isolated score speed:

| Evaluator | selected eval/s |
|---|---:|
| grouped7 EGEV10 materialized | 1.679M |

This is in the same range as the previous 1.666M measurement.

### Midgame Level Checks

Settings:

- `bin/problem/midgame_test.txt`
- `-nobook -hash 25`

Current fast and materialized grouped7 are different evaluators, so their trees differ. These runs are only
sanity and throughput checks, not strength conclusions.

| Evaluator | level | threads | nodes | time | NPS |
|---|---:|---:|---:|---:|---:|
| grouped7 materialized | 5 | 28 | 82,062 | 0.023s | 3.57M |
| Current EGEV4 fast | 5 | 28 | 85,291 | 0.048s | 1.78M |
| grouped7 materialized | 10 | 28 | 15,348,108 | 0.620s | 24.76M |
| Current EGEV4 fast | 10 | 28 | 17,916,428 | 0.722s | 24.81M |

For a cleaner same-evaluator comparison, rebuilt the normal grouped split engine and compared it against
materialized grouped7.

| Evaluator | level | threads | nodes | time | NPS |
|---|---:|---:|---:|---:|---:|
| grouped7 materialized | 10 | 1 | 13,376,940 | 2.485s | 5.38M |
| grouped7 split | 10 | 1 | 13,376,940 | 2.519s | 5.31M |
| grouped7 materialized | 10 | 28 | 15,348,108 | 0.620s | 24.76M |
| grouped7 split | 10 | 28 | 14,948,376 | 0.633s | 23.62M |

The one-thread run confirms identical moves, scores, and node counts for split vs materialized. Search-level
speed gain is modest at level 10 because evaluation is only part of total cost, but the materialized path is
correct and does not regress the same-tree search.

### Current Judgment

- Keep `EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZED` as the leading compact-file path.
- Do not use lazy cache or per-move diff cache for adoption.
- Next useful work is a small GTP match or level 10+ strength check between current fast and grouped7 materialized.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 4

### Small Strength Check For grouped7

Ran `battle_parallel_nonstop_gtp.py` with current EGEV4 fast against grouped7 materialized.

Settings:

- level 10
- 1 thread per engine
- XOT openings from `problem/xot/openingslarge.txt`
- 30 paired matches

Result:

| Player | Win rate | Average disc difference |
|---|---:|---:|
| Current EGEV4 fast | 0.7000 | +2.23 |
| grouped7 materialized | 0.3000 | -2.23 |

This is a small sample, but it confirms the grouped7 averaged model is still weaker even after the materialized
speed fix.

### groupedN Converter

Extended `convert_eval77_fm_egev4_to_grouped_fm.cpp` to accept generic `groupedN` modes with contiguous,
approximately equal phase groups, while preserving the previous special `grouped7` layout.

Generated and tested two additional compact models:

| Model | FM groups | Output bytes | Materialized selected eval/s |
|---|---:|---:|---:|
| grouped15 | 15 | 340,146,096 | 1.675M |
| grouped30 | 30 | 566,909,920 | 1.664M |

Both passed materialized consistency checks:

- `games=500`
- `checked_positions=30084`
- `mismatches=0`

Midgame level 10 / 28 threads:

| Evaluator | nodes | time | NPS |
|---|---:|---:|---:|
| grouped15 materialized | 15,515,764 | 0.669s | 23.19M |
| grouped30 materialized | 24,689,450 | 0.918s | 26.89M |

Small level 10 match checks:

| Player | Win rate vs current fast | Average disc difference vs current fast |
|---|---:|---:|
| grouped15 materialized | 0.3167 | -3.10 |
| grouped30 materialized | 0.1333 | -4.50 |

The simple contiguous `groupedN` averaging did not improve strength in these small samples. More groups make the
aggregate score-speed checksum closer to current fast, but that did not translate into better play at level 10.

### Current Judgment

- The main runtime improvement is solved well enough by materialization.
- Strength loss is now the dominant problem.
- Naive post-hoc vector averaging is probably not enough; the next meaningful quality work should compare score
  error distributions directly and/or retrain grouped vectors with fixed linear weights.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 5

### Direct Score-Error Measurement

Added `src/tools/evaluation/util/eval77_fm_grouped_error_check.cpp`.

The tool loads a baseline EGEV4 model and a grouped EGEV10 model independently, samples random-play positions,
uses the engine's existing 7.7-FM feature ID extraction, and reports score error distributions by phase.
It also validates the custom EGEV4 scorer against the existing fast evaluator; all reported runs had
`engine_mismatches=0`.

Settings:

- baseline: `bin/resources/eval.egev4`
- games: 2000
- seed: 20260716
- positions: 120,304

Overall score error:

| Model | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| grouped7 average | 0.1260 | 0.0355 | +2.37 | 12.70 | 16.02 |
| grouped15 average | 0.1016 | 0.0393 | +2.54 | 11.66 | 14.72 |
| grouped30 average | 0.1350 | 0.0526 | +1.13 | 9.18 | 12.11 |

The grouped30 average model improved aggregate MAE/RMSE but got worse in the level 10 match sample. Phase-level
inspection showed large localized failures, especially around phases 14-17:

| Model | phase | bias | MAE | RMSE |
|---|---:|---:|---:|---:|
| grouped30 average | 14 | -16.49 | 18.39 | 21.59 |
| grouped30 average | 16 | -17.65 | 18.84 | 21.87 |

### Representative-Copy Grouping

Extended `convert_eval77_fm_egev4_to_grouped_fm.cpp` with optional `copyfirst` and `copylast` suffixes.
For example, `grouped30copyfirst` keeps the first phase's FM vector table in each 2-phase group instead of
averaging vectors.

Generated models:

- `model/eval77_fm_grouped30_copyfirst_init/eval_grouped30_copyfirst.egev10`
- `model/eval77_fm_grouped30_copylast_init/eval_grouped30_copylast.egev10`

Score error against current fast:

| Model | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| grouped30 average | 0.1350 | 0.0526 | +1.13 | 9.18 | 12.11 |
| grouped30 copyfirst | 0.5820 | 0.0285 | +1.47 | 2.29 | 4.47 |
| grouped30 copylast | 0.5813 | 0.0296 | -1.24 | 2.15 | 4.11 |

Materialized consistency/speed:

| Model | consistency | selected eval/s |
|---|---|---:|
| grouped30 copyfirst | `mismatches=0`, 30,084 positions | 1.691M |
| grouped30 copylast | `mismatches=0`, 30,084 positions | 1.703M |

Level 10 match checks, current fast vs candidate:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| grouped7 average | 100 paired matches | 0.2950 | -3.04 |
| grouped30 copyfirst | 100 paired matches | 0.4150 | -1.21 |
| grouped30 copylast | 30 paired matches | 0.2500 | -2.50 |

`grouped30copyfirst` is the best post-hoc grouped model so far: it keeps the compact EGEV10 file at about
567MB, keeps materialized scoring near current fast, and improves the level 10 match result substantially
over grouped7. It is still weaker than current fast, so the goal is not complete.

### Current Judgment

- Averaging grouped vectors is harmful for strength even when aggregate RMSE improves.
- Representative-copy grouping is a useful baseline and should replace simple averaging as the next candidate path.
- Next work should either tune representative choice per group using error/match feedback, or retrain grouped
  vectors rather than averaging them.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 6

### Custom Representative Selection

Extended `convert_eval77_fm_egev4_to_grouped_fm.cpp` with a `copycustom` suffix. The converter now accepts one
representative phase per FM group as a CSV argument, validates that each representative belongs to its group, and
copies that phase's vector table and vector scale.

Generated two `grouped30copycustom` hybrid models from the phase-level copyfirst/copylast error logs:

- RMSE hybrid representatives:
  `0,2,4,7,8,11,13,15,17,19,21,23,25,26,29,31,33,34,37,38,40,43,45,47,48,50,53,54,56,59`
- MAE hybrid representatives:
  `0,2,4,7,8,11,13,15,17,19,21,23,25,26,29,31,32,34,37,38,40,43,44,47,48,50,53,54,56,59`

Score error improved slightly over both fixed representative models:

| Model | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| grouped30 copyfirst | 0.5820 | 0.0285 | +1.47 | 2.29 | 4.47 |
| grouped30 copylast | 0.5813 | 0.0296 | -1.24 | 2.15 | 4.11 |
| grouped30 hybrid RMSE | 0.5818 | 0.0284 | -0.39 | 2.12 | 4.06 |
| grouped30 hybrid MAE | 0.5815 | 0.0283 | -0.22 | 2.12 | 4.06 |

Both custom models passed the materialized consistency check:

- `games=500`
- `checked_positions=30084`
- `mismatches=0`

However, level 10 match checks got worse:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| grouped30 hybrid RMSE | 30 paired matches | 0.2000 | -2.90 |
| grouped30 hybrid MAE | 30 paired matches | 0.2500 | -3.00 |

This is a useful negative result: minimizing random-play score error per phase did not improve level 10 play.

### More Copyfirst Groups

Since `copyfirst` was the best grouped30 match baseline, tested larger contiguous `groupedNcopyfirst` layouts.

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| grouped30 copyfirst | 30 | 566,909,920 | 0.5820 | 0.0285 | 2.29 | 4.47 | 1.691M |
| grouped40 copyfirst | 40 | 718,085,760 | 0.7219 | 0.0201 | 1.54 | 3.73 | 1.678M |
| grouped50 copyfirst | 50 | 869,261,664 | 0.8605 | 0.0112 | 0.84 | 2.89 | 1.687M |

All passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| grouped30 copyfirst | 100 paired matches | 0.4150 | -1.21 |
| grouped40 copyfirst | 30 paired matches | 0.4333 | -1.00 |
| grouped50 copyfirst | 30 paired matches | 0.4667 | -1.03 |
| grouped50 copyfirst | 100 paired matches | 0.5450 | -0.42 |

`grouped50copyfirst` is now the strongest measured post-hoc grouped model. It gives up a much larger file than
grouped30, but it keeps materialized score speed essentially unchanged and gets close to current fast in level 10
play. The remaining question is whether the memory increase is acceptable, or whether a smaller model needs
training/fitting rather than representative copying.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 7

### Copylast And Custom Group Layouts

Extended `convert_eval77_fm_egev4_to_grouped_fm.cpp` again with a `groupsizes` base mode. This allows an explicit
CSV of contiguous group sizes instead of only the fixed `groupedN` equal partition. Existing `copyfirst`,
`copylast`, and `copycustom` suffixes work with this layout too.

Follow-up fixed-layout copylast checks:

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| grouped40 copylast | 40 | 718,085,760 | 0.7203 | 0.0189 | 1.48 | 3.48 | not retimed |
| grouped45 copylast | 45 | 793,673,744 | 0.7822 | 0.0156 | 1.13 | 3.03 | not retimed |
| grouped50 copylast | 50 | 869,261,664 | 0.8590 | 0.0099 | 0.77 | 2.58 | 1.657M |

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| grouped40 copylast | 30 paired matches | 0.4333 | -0.63 |
| grouped45 copylast | 30 paired matches | 0.3500 | -1.47 |
| grouped50 copylast | 30 paired matches | 0.5167 | -0.40 |
| grouped50 copylast | 100 paired matches | 0.4950 | -0.01 |

`grouped50copylast` is effectively equal to current fast in this 100-match XOT sample, but fixed `grouped45`
was unexpectedly poor. Equal partitioning is not monotonic in strength because the paired phases land in different
parts of the game.

### Low-Cost 50-Group Layout

Built a custom 50-group layout by choosing 10 non-overlapping adjacent phase pairs with low observed RMSE cost from
the earlier copyfirst/copylast phase logs. Selected pairs:

`0-1, 2-3, 4-5, 12-13, 14-15, 16-17, 18-19, 20-21, 24-25, 26-27`

Group size CSV:

`2,2,2,1,1,1,1,1,1,2,2,2,2,2,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1`

Representative phase CSV:

`0,2,4,6,7,8,9,10,11,13,15,17,19,21,22,23,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59`

Results:

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| groupsizes50 lowcost | 50 | 869,261,664 | 0.8936 | 0.0051 | 0.44 | 1.61 | 1.679M |

Consistency check:

- `games=500`
- `checked_positions=30084`
- `mismatches=0`

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes50 lowcost | 30 paired matches | 0.5333 | +0.20 |
| groupsizes50 lowcost | 100 paired matches | 0.5250 | +0.37 |
| groupsizes50 lowcost | 200 paired matches | 0.4900 | +0.02 |

This is the best measured post-hoc grouped model so far. It keeps the same 50-group file size as fixed grouped50,
keeps selected score speed near 1.68M eval/s, and is essentially equal to current fast in the 200-match XOT sample.
The result still needs broader validation, but the layout-selection direction is clearly better than plain equal
partitioning or random-play RMSE representative tuning.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 8

### Low-Cost Layout Size Curve

Generated additional low-cost `groupsizescopycustom` layouts around the 50-group result. The layouts use the same
DP pair-cost heuristic as cycle 7, choosing the lowest-cost non-overlapping adjacent phase pairs from the existing
copyfirst/copylast phase-error logs.

Score-error results:

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| groupsizes45 lowcost | 45 | 793,673,744 | 0.8166 | 0.0095 | 0.79 | 2.23 | 1.698M |
| groupsizes48 lowcost | 48 | 839,026,496 | 0.8628 | 0.0068 | 0.57 | 1.86 | not standalone-retimed |
| groupsizes50 lowcost | 50 | 869,261,664 | 0.8936 | 0.0051 | 0.44 | 1.61 | 1.679M |
| groupsizes52 lowcost | 52 | 899,496,832 | 0.9242 | 0.0035 | 0.30 | 1.33 | not standalone-retimed |

All three new models passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes45 lowcost | 30 paired matches | 0.6000 | -0.17 |
| groupsizes45 lowcost | 100 paired matches | 0.5200 | -0.06 |
| groupsizes45 lowcost | 200 paired matches | 0.4825 | -0.45 |
| groupsizes48 lowcost | 30 paired matches | 0.5333 | -0.27 |
| groupsizes48 lowcost | 100 paired matches | 0.5100 | +0.05 |
| groupsizes48 lowcost | 200 paired matches | 0.4850 | -0.12 |
| groupsizes52 lowcost | 30 paired matches | 0.5000 | +0.20 |
| groupsizes52 lowcost | 100 paired matches | 0.4850 | +0.17 |

The 45/48-group low-cost models are useful smaller tradeoff points, but their 200-match checks are slightly below
current fast. The 50-group low-cost model remains the best robust post-hoc grouped candidate so far: it is the
smallest layout that stayed essentially equal to current fast in a 200-match XOT sample. The 52-group model has the
best direct score error, but did not show a clear match-strength gain in the 100-match sample.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 9

### Phase-Level Sign Metrics

Extended `eval77_fm_grouped_error_check.cpp` so `ErrorStats` now tracks `exact_match`, `exact_rate`,
`sign_disagree`, and `sign_disagree_rate` for both overall and per-phase output. This keeps the existing MAE/RMSE
fields intact while making phase-level sign errors visible.

Extended `select_eval77_fm_lowcost_group_layout.py` with metric selection:

- `--metric rmse`: previous default, minimizing per-phase SSE from RMSE.
- `--metric mae`: minimizes summed absolute error.
- `--metric sign`: minimizes phase sign-disagreement count first and uses RMSE as a tie-breaker.

### Sign-Cost 50-Group Layout

Regenerated grouped30/40/45/50 candidate error logs with phase-level sign fields, then built a 50-group layout using
`--metric sign`.

Score-error comparison:

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| groupsizes50 lowcost RMSE | 50 | 869,261,664 | 0.8936 | 0.0051 | 0.44 | 1.61 |
| groupsizes50 signcost | 50 | 869,261,664 | 0.8937 | 0.0042 | 0.47 | 1.74 |

The sign-cost layout did reduce random-play sign disagreement, but level 10 play got worse:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes50 signcost | 30 paired matches | 0.3500 | -0.90 |

This is another useful negative result: optimizing sign disagreement alone is not a good selector for match
strength.

### 55-Group Low-Cost Layout

Generated a 55-group RMSE-lowcost layout using only five adjacent phase pairs:

`0-1, 2-3, 4-5, 12-13, 18-19`

Results:

| Model | FM groups | Output bytes | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| groupsizes55 lowcost | 55 | 944,849,584 | 0.9701 | 0.0013 | 0.12 | 0.81 | 1.667M |

Consistency check: `mismatches=0` over 30,084 positions.

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes55 lowcost | 30 paired matches | 0.5333 | +0.17 |
| groupsizes55 lowcost | 100 paired matches | 0.5000 | -0.10 |

The 55-group model has much better direct score error than 50 groups, but did not show a clear match-strength gain
in the 100-match XOT sample. It is larger than the current best 50-group candidate, so `groupsizes50 lowcost RMSE`
remains the best practical post-hoc grouped candidate measured so far.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 10

### Lossless Upper-Bound Layouts

Generated high-group-count layouts to find the practical upper bound of the post-hoc grouping approach.

Baseline reference:

| Model | bytes | selected eval/s |
|---|---:|---:|
| current EGEV4 fast | 1,020,437,434 | 1.677M |

Upper-bound grouped models:

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|
| groupsizes57 lowcost | 57 | 975,084,752 | -45,352,682 | 1.0000 | 0.0000 | 1.684M |
| groupsizes58 lowcost | 58 | 990,202,336 | -30,235,098 | 1.0000 | 0.0000 | 1.691M |
| grouped60 exact | 60 | 1,020,437,568 | +134 | 1.0000 | 0.0000 | 1.652M |

`groupsizes57 lowcost` uses the three zero-cost early phase pairs:

`0-1, 2-3, 4-5`

Evidence for the 57-group lossless candidate:

- direct score check: `games=10000`, `positions=601782`, `exact_match=601782`, `engine_mismatches=0`
- materialized consistency: `games=500`, `checked_positions=30084`, `mismatches=0`
- level 10 match check vs current fast: 30 paired matches, candidate win rate `0.5000`, average disc diff `+0.00`

`groupsizes58 lowcost` also checked as lossless over 601,782 sampled positions and got the same 30-match
`0.5000 / +0.00` result, but it saves less space than the 57-group layout. The 60-group exact layout is useful as a
format upper bound, but it is slightly larger than EGEV4 and not faster in this score-speed check.

Current practical split:

- `groupsizes57 lowcost`: best no-strength-loss / no-score-loss conservative model, saving about 45MB.
- `groupsizes50 lowcost RMSE`: best meaningful compression model so far, saving about 151MB and staying roughly
  equal to current fast in the 200-match XOT sample, but with nonzero score error.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 11

### Boundary Search Below The Lossless Layout

Measured additional low-cost layouts between the conservative 57-group lossless model and the more compressed
50-group model.

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groupsizes57 lowcost | 57 | 975,084,752 | -45,352,682 | 1.0000 | 0.0000 | 0.00 | 0.00 | 1.684M |
| groupsizes56 lowcost | 56 | 959,967,168 | -60,470,266 | 0.9848 | 0.0008 | 0.06 | 0.57 | 1.651M |
| groupsizes54 lowcost | 54 | 929,732,000 | -90,705,434 | 0.9546 | 0.0023 | 0.18 | 1.00 | 1.677M |
| groupsizes53 lowcost | 53 | 914,614,416 | -105,823,018 | 0.9397 | 0.0029 | 0.24 | 1.17 | 1.686M |

The 56/54/53 score-error rows used `games=10000`, `positions=601782`, and all had `engine_mismatches=0`.
All three passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes56 lowcost | 30 paired matches | 0.5000 | +0.00 |
| groupsizes56 lowcost | 100 paired matches | 0.5000 | +0.00 |
| groupsizes54 lowcost | 30 paired matches | 0.5333 | +0.17 |
| groupsizes54 lowcost | 100 paired matches | 0.5000 | -0.10 |
| groupsizes53 lowcost | 30 paired matches | 0.5667 | +0.37 |
| groupsizes53 lowcost | 100 paired matches | 0.5150 | +0.25 |
| groupsizes53 lowcost | 200 paired matches | 0.5125 | +0.21 |

Three-player level 10 cross-check, 50 paired matches per pairing:

| Player | vs fast | vs groupsizes50 | vs groupsizes53 | all win rate | all disc diff |
|---|---:|---:|---:|---:|---:|
| fast | - | 0.5400 / +0.32 | 0.5100 / -0.06 | 0.5250 | +0.13 |
| groupsizes50 lowcost | 0.4600 / -0.32 | - | 0.4900 / +0.10 | 0.4750 | -0.11 |
| groupsizes53 lowcost | 0.4900 / +0.06 | 0.5100 / -0.10 | - | 0.5000 | -0.02 |

Current interpretation:

- `groupsizes57 lowcost` is still the cleanest conservative candidate because score equality held over 601,782
  positions and paired match behavior was exactly identical in the checked XOT sample.
- `groupsizes56 lowcost` is an interesting near-lossless candidate: nonzero random-play score error, but exactly
  identical level 10 match behavior over 100 paired matches.
- `groupsizes53 lowcost` is the strongest compressed candidate in this latest XOT sample, saving about 106MB and
  scoring slightly better than current fast over 200 paired matches. The three-player check keeps it in the same
  practical strength band as current fast, while `groupsizes50 lowcost` remains the smaller but slightly riskier
  compression option.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 12

### Alternate 53-Group Layout Check

Extended `select_eval77_fm_lowcost_group_layout.py` with `--exclude-pair START|START-END` so nearby layouts can be
generated without manually editing pair lists. This preserves the default layout selection behavior.

Tested a 53-group alternate layout excluding the `26-27` pair from the cycle 11 best 53-group layout. The selector
then chose `20-21` instead.

Pair difference:

| Layout | extra seventh pair |
|---|---|
| groupsizes53 lowcost | `26-27`, representative `26` |
| groupsizes53 ex26 lowcost | `20-21`, representative `21` |

Score-error comparison over 10,000 random-play games / 601,782 positions:

| Model | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|
| groupsizes53 lowcost | 0.9397 | 0.0029 | -0.06 | 0.24 | 1.17 | 1.686M |
| groupsizes53 ex26 lowcost | 0.9392 | 0.0030 | -0.13 | 0.24 | 1.18 | 1.671M |

Both passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 match checks:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| groupsizes53 ex26 lowcost vs fast | 100 paired matches | 0.4650 | -0.17 |

Three-player level 10 cross-check, 40 paired matches per pairing:

| Player | vs fast | vs groupsizes53 | vs groupsizes53 ex26 | all win rate | all disc diff |
|---|---:|---:|---:|---:|---:|
| fast | - | 0.5000 / +0.03 | 0.5500 / +0.33 | 0.5250 | +0.17 |
| groupsizes53 lowcost | 0.5000 / -0.03 | - | 0.5500 / +0.42 | 0.5250 | +0.20 |
| groupsizes53 ex26 lowcost | 0.4500 / -0.33 | 0.4500 / -0.42 | - | 0.4500 | -0.38 |

The alternate layout was worse despite very similar random-play MAE/RMSE. The `26-27` pair should stay in the
current best 53-group layout; this reinforces that small pair-choice changes can matter more in search than their
aggregate random-play error suggests.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 13

### Direct Vector-Table Equivalence Check

Added `src/tools/evaluation/util/eval77_fm_phase_vector_compare.cpp` to inspect EGEV4 phase tables directly. The
tool compares vector scales and all FM vector bytes for candidate phase pairs. Linear-table differences are reported
too, but they do not break grouped-FM equivalence because the grouped EGEV10 layout keeps linear weights per phase.

Command:

`bin/eval77_fm_phase_vector_compare.exe bin/resources/eval.egev4 0-1,2-3,4-5,12-13,14-15,18-19,26-27`

Key results:

| Pair | vector scale equal | vector mismatch params | vector mismatch bytes | linear mismatch params | vector lossless |
|---|---:|---:|---:|---:|---:|
| 0-1 | yes | 0 | 0 | 0 | yes |
| 2-3 | yes | 0 | 0 | 195 | yes |
| 4-5 | yes | 0 | 0 | 1,173 | yes |
| 12-13 | no | 108,041 | 1,100,676 | 226,557 | no |
| 14-15 | no | 212,696 | 2,243,122 | 303,633 | no |
| 18-19 | no | 251,936 | 2,294,641 | 469,696 | no |
| 26-27 | no | 430,328 | 3,704,555 | 621,865 | no |

This turns the `groupsizes57 lowcost` result from sample evidence into direct model evidence: its three shared
pairs (`0-1`, `2-3`, `4-5`) have identical FM vector tables and vector scales in the source EGEV4 model, while the
linear tables remain phase-specific. That explains why `groupsizes57 lowcost` was exactly equal over 601,782 sampled
positions and in the paired match check.

It also explains the boundary behavior: `groupsizes56 lowcost` adds `12-13`, the first non-lossless pair, so it
introduces measurable score error even though the level 10 sample still played identically over 100 paired matches.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 14

### Automatic Lossless Group Layout

Extended `convert_eval77_fm_egev4_to_grouped_fm.cpp` with a `lossless` mode. The converter now scans adjacent
EGEV4 phases and keeps extending a vector group while the FM vector scale and all FM vector bytes are identical.
Linear weights remain per-phase in the grouped EGEV10 file, so linear-table differences do not affect score
equivalence.

Boundary check:

`bin/eval77_fm_phase_vector_compare.exe bin/resources/eval.egev4 0-1,1-2,2-3,3-4,4-5,5-6`

| Pair | vector scale equal | vector mismatch params | vector mismatch bytes | linear mismatch params | vector lossless |
|---|---:|---:|---:|---:|---:|
| 0-1 | yes | 0 | 0 | 0 | yes |
| 1-2 | yes | 0 | 0 | 83 | yes |
| 2-3 | yes | 0 | 0 | 195 | yes |
| 3-4 | yes | 0 | 0 | 470 | yes |
| 4-5 | yes | 0 | 0 | 1,173 | yes |
| 5-6 | no | 3,288 | 37,026 | 3,009 | no |

Generated model:

`model/eval77_fm_lossless_auto_init/eval_lossless_auto.egev10`

The automatic layout produced 55 FM groups. Group 0 contains phases `0 1 2 3 4 5`; phases `6..59` remain
singletons. Output size is `944,849,584` bytes, saving `75,587,850` bytes versus current EGEV4 and `30,235,168`
bytes versus the earlier manual `groupsizes57 lowcost` lossless layout.

Verification:

| Check | Result |
|---|---:|
| direct score error | `games=10000`, `positions=601782`, `exact_rate=1.000000`, `sign_disagree=0`, `MAE=0.0000`, `RMSE=0.0000` |
| engine mismatch count | `0` |
| materialized consistency | `games=500`, `checked_positions=30084`, `mismatches=0` |
| selected eval/s, current EGEV4 fast | `1.682M` |
| selected eval/s, lossless auto grouped | `1.684M` |

Level 10 paired match check used same-source generic buildcheck binaries:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| lossless auto grouped vs fast generic | 30 paired matches | 0.5000 | +0.00 |

Interpretation:

- `lossless auto grouped` supersedes the manual `groupsizes57 lowcost` conservative candidate: it is still exact in
  direct scoring, consistency, and the paired match check, while saving about 75.6MB instead of 45.4MB.
- The useful next compression cycle should start from this lossless base. A 53-group model built as `0..5` plus two
  additional low-cost non-lossless merges would have the same size as the previous `groupsizes53 lowcost` candidate
  but should have much lower random-play score error, because it spends only two lossy merges instead of four.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 15

### 53-Group Layout From The Lossless Base

Used the existing low-cost pair selector to choose two additional non-lossless merges after excluding the lossless
run-internal pairs `0-1,1-2,2-3,3-4,4-5`.

Selected additional pairs:

`12-13, 18-19`

Generated model:

`model/eval77_fm_losslessbase53_lowcost_init/eval_losslessbase53_lowcost.egev10`

Layout summary:

- group 0: phases `0 1 2 3 4 5`, representative `0`
- additional lossy groups: `12-13` representative `13`, `18-19` representative `19`
- FM groups: `53`
- output bytes: `914,614,416`
- delta vs current EGEV4: `-105,823,018` bytes

Score-error comparison over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groupsizes53 lowcost | 53 | 914,614,416 | 0.9397 | 0.0029 | -0.06 | 0.24 | 1.17 | 1.686M |
| losslessbase53 lowcost | 53 | 914,614,416 | 0.9699 | 0.0014 | -0.05 | 0.12 | 0.81 | 1.671M |
| groupsizes55 lowcost | 55 | 944,849,584 | 0.9701 | 0.0013 | - | 0.12 | 0.81 | 1.667M |

The new 53-group layout has the same file size as the previous `groupsizes53 lowcost` model but essentially the
same score-error profile as the larger 55-group low-cost model.

Other checks:

- direct score check: `engine_mismatches=0`
- materialized consistency: `games=500`, `checked_positions=30084`, `mismatches=0`
- level 10 paired match check vs fast generic: 100 paired matches, candidate win rate `0.5000`, average disc diff
  `-0.10`

Interpretation:

`losslessbase53 lowcost` supersedes the previous 53-group layout as the lower-risk 53-group candidate. The previous
53-group layout had a favorable 200-match sample, so it is not discarded as a strength data point, but it now has no
size advantage and much worse direct score error.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 16

### 50-Group Layout From The Lossless Base

Repeated the same idea at 50 FM groups. With the `0..5` lossless run fixed, the selector only needs five additional
lossy merges to reach 50 groups, instead of spending three separate pair slots on `0-1,2-3,4-5`.

Selected additional pairs:

`12-13, 14-15, 18-19, 20-21, 26-27`

Generated model:

`model/eval77_fm_losslessbase50_lowcost_init/eval_losslessbase50_lowcost.egev10`

Layout summary:

- group 0: phases `0 1 2 3 4 5`, representative `0`
- additional lossy groups: `12-13` representative `13`, `14-15` representative `15`, `18-19` representative `19`,
  `20-21` representative `21`, `26-27` representative `26`
- FM groups: `50`
- output bytes: `869,261,664`
- delta vs current EGEV4: `-151,175,770` bytes

Score-error comparison over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groupsizes50 lowcost | 50 | 869,261,664 | 0.8936 | 0.0051 | - | 0.44 | 1.61 | 1.679M |
| losslessbase50 lowcost | 50 | 869,261,664 | 0.9244 | 0.0036 | -0.10 | 0.30 | 1.33 | 1.687M |
| groupsizes52 lowcost | 52 | 899,496,832 | 0.9242 | 0.0035 | - | 0.30 | 1.33 | not standalone-retimed |

The new 50-group layout has the same file size as the previous 50-group low-cost model but the direct error profile
of the larger 52-group model.

Other checks:

- direct score check: `engine_mismatches=0`
- materialized consistency: `games=500`, `checked_positions=30084`, `mismatches=0`
- level 10 paired match check vs fast generic: 100 paired matches, candidate win rate `0.4850`, average disc diff
  `+0.17`

Interpretation:

`losslessbase50 lowcost` is now the best measured 50-group post-hoc layout by direct error and keeps the same
151MB file-size saving. Its 100-match XOT result remains in the same practical band as current fast, but a 200-match
extension is the next useful validation before replacing the earlier `groupsizes50 lowcost` strength interpretation.

## Addendum: 2026-07-16 Measurement And Improvement Cycle 17

### Fixed-Run Layout Selector

Extended `select_eval77_fm_lowcost_group_layout.py` with `--fixed-run START-END[:REP]`. This lets the selector keep
the directly proven lossless run `0-5:0` fixed, subtract those five free merges from the target group count, and then
choose only the remaining low-cost lossy adjacent pairs with the existing DP.

Reproduction checks:

- `target_groups=53 --fixed-run 0-5` selects `12-13,18-19` and reproduces `losslessbase53 lowcost`.
- `target_groups=50 --fixed-run 0-5` selects `12-13,14-15,18-19,20-21,26-27` and reproduces
  `losslessbase50 lowcost`.
- Running the old selector command without `--fixed-run` still reproduces the previous `groupsizes50 lowcost`
  pair list.

### 200-Match Check For losslessbase50

Extended the level 10 XOT check for `losslessbase50 lowcost` from 100 to 200 paired matches, using the same-source
generic buildcheck binaries.

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| losslessbase50 lowcost vs fast generic | 100 paired matches | 0.4850 | +0.17 |
| losslessbase50 lowcost vs fast generic | 200 paired matches | 0.4900 | +0.09 |

This keeps `losslessbase50 lowcost` in the same practical strength band as current fast, while preserving the
151MB file-size saving and improving direct error substantially over the old 50-group layout.

### Smaller Fixed-Run Layouts

Generated 48- and 45-group layouts from the same fixed lossless base.

Selected additional pairs:

- `losslessbase48`: `12-13,14-15,16-17,18-19,20-21,24-25,26-27`
- `losslessbase45`: `12-13,14-15,16-17,18-19,20-21,22-23,24-25,26-27,28-29,54-55`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| groupsizes50 lowcost | 50 | 869,261,664 | -151,175,770 | 0.8939 | 0.0052 | -0.18 | 0.44 | 1.62 | 1.679M |
| losslessbase48 lowcost | 48 | 839,026,496 | -181,410,938 | 0.8939 | 0.0052 | -0.18 | 0.44 | 1.62 | 1.629M |
| groupsizes45 lowcost | 45 | 793,673,744 | -226,763,690 | 0.8166 | 0.0095 | - | 0.79 | 2.23 | 1.698M |
| losslessbase45 lowcost | 45 | 793,673,744 | -226,763,690 | 0.8480 | 0.0077 | -0.23 | 0.64 | 1.99 | 1.634M |

Both new models passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Interpretation:

- `losslessbase48 lowcost` is the cleanest replacement for the previous `groupsizes50 lowcost`: it has the same
  sampled direct-error profile and, by construction, the same lossy pair set, but saves another 30.2MB because the
  lossless early phases are one 6-phase run instead of three separate pairs.
- `losslessbase45 lowcost` improves direct error over the old 45-group layout at the same file size, but needs a
  level 10 match check before it can be treated as a practical candidate.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 18

### Semantic Equivalence Check For losslessbase48

Added `src/tools/evaluation/util/eval77_grouped_semantic_compare.cpp`. The tool compares two grouped EGEV10 files by
the per-phase evaluation semantics instead of raw bytes: linear scales, per-phase linear tables, mapped vector scales,
and mapped vector bytes.

Comparison:

`bin/eval77_grouped_semantic_compare.exe model/eval77_fm_groupsizes50_lowcost_init/eval_groupsizes50_lowcost.egev10 model/eval77_fm_losslessbase48_lowcost_init/eval_losslessbase48_lowcost.egev10`

Result:

`linear_scale_mismatches=0 vector_scale_mismatches=0 linear_mismatch_params=0 vector_mismatch_params=0 vector_mismatch_bytes=0 semantic_equal=1`

This proves `losslessbase48 lowcost` is semantically identical to the previous `groupsizes50 lowcost` evaluator,
despite using 48 FM groups instead of 50.

### 200-Match Check For losslessbase48

Measured `losslessbase48 lowcost` against current fast with the same-source generic buildcheck binaries.

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| losslessbase48 lowcost vs fast generic | 200 paired matches | 0.4875 | +0.01 |

Current interpretation:

- `losslessbase48 lowcost` supersedes `groupsizes50 lowcost`: it has identical evaluation semantics and similar
  measured level 10 behavior, while saving `30,235,168` additional bytes.
- The current practical candidates are now:
  - conservative exact: `lossless auto grouped`, 55 groups, `944,849,584` bytes, exact score equality.
  - balanced compression: `losslessbase48 lowcost`, 48 groups, `839,026,496` bytes, semantically identical to the
    old 50-group practical candidate.
  - higher compression probe: `losslessbase45 lowcost`, 45 groups, `793,673,744` bytes, better direct error than the
    old 45-group layout but still needs a level 10 match check.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 19

### Match Check For losslessbase45

Measured the higher-compression `losslessbase45 lowcost` candidate against current fast with the same-source generic
buildcheck binaries.

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| losslessbase45 lowcost vs fast generic | 100 paired matches | 0.5100 | -0.07 |
| losslessbase45 lowcost vs fast generic | 200 paired matches | 0.4725 | -0.41 |

Interpretation:

- The 100-match sample looked acceptable, but the 200-match extension moved clearly below current fast.
- `losslessbase45 lowcost` remains a useful compression data point, but it is not the current practical candidate.
- `losslessbase48 lowcost` is the better balanced candidate: it is semantically identical to the old 50-group
  practical model, passes a 200-match check at `0.4875 / +0.01`, and saves 181MB versus EGEV4.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 20

### Boundary Check Between 48 And 45 Groups

Generated and measured the intermediate fixed-run layouts between `losslessbase48` and `losslessbase45`.

Selected additional pairs:

- `losslessbase47`: `12-13,14-15,16-17,18-19,20-21,22-23,24-25,26-27`
- `losslessbase46`: `12-13,14-15,16-17,18-19,20-21,22-23,24-25,26-27,54-55`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase48 lowcost | 48 | 839,026,496 | -181,410,938 | 0.8939 | 0.0052 | -0.18 | 0.44 | 1.62 | 1.629M |
| losslessbase47 lowcost | 47 | 823,908,912 | -196,528,522 | 0.8786 | 0.0059 | -0.22 | 0.51 | 1.74 | 1.634M |
| losslessbase46 lowcost | 46 | 808,791,328 | -211,646,106 | 0.8632 | 0.0070 | -0.20 | 0.57 | 1.87 | 1.644M |
| losslessbase45 lowcost | 45 | 793,673,744 | -226,763,690 | 0.8480 | 0.0077 | -0.23 | 0.64 | 1.99 | 1.634M |

Both new models passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 XOT match checks against current fast, using same-source generic buildcheck binaries:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| losslessbase48 lowcost | 200 paired matches | 0.4875 | +0.01 |
| losslessbase47 lowcost | 200 paired matches | 0.4850 | -0.14 |
| losslessbase46 lowcost | 100 paired matches | 0.5100 | +0.05 |
| losslessbase46 lowcost | 200 paired matches | 0.4850 | -0.11 |
| losslessbase45 lowcost | 200 paired matches | 0.4725 | -0.41 |

Interpretation:

- `losslessbase48 lowcost` remains the best balanced replacement for the old `groupsizes50 lowcost`.
- `losslessbase46 lowcost` and `losslessbase47 lowcost` are viable extra-compression probes, saving another
  30-45MB over `losslessbase48`, but their 200-match results are slightly below the 48-group candidate.
- `losslessbase45 lowcost` appears to cross the practical-risk boundary in this measurement set.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 21

### Longer And Alternate-Seed Match Checks

Extended `battle_parallel_nonstop_gtp.py` with `--shuffle-seed`, defaulting to the previous fixed seed `57`. This
keeps old runs reproducible while allowing alternate opening subsets to be checked explicitly.

First, extended the main `losslessbase48 lowcost` check from 200 to 400 paired matches with the default seed:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase48 lowcost | 57 | 200 paired matches | 0.4875 | +0.01 |
| losslessbase48 lowcost | 57 | 400 paired matches | 0.4913 | -0.10 |

Then measured a different opening shuffle seed, `20260717`:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| lossless auto exact | 20260717 | 100 paired matches | 0.5000 | +0.00 |
| losslessbase48 lowcost | 20260717 | 200 paired matches | 0.4625 | -0.24 |
| losslessbase50 lowcost | 20260717 | 200 paired matches | 0.4575 | -0.23 |
| losslessbase53 lowcost | 20260717 | 200 paired matches | 0.4775 | -0.25 |

The exact lossless model stayed perfectly identical, so the alternate-seed drop is not caused by the fast-vs-grouped
binary comparison itself. It is caused by the lossy grouped layout interacting with that opening subset.

### Single-Pair Isolation

To isolate the issue, generated two 54-group models from the lossless base:

- `losslessbase54 pair12`: only adds `12-13`, representative `13`.
- `losslessbase54 pair18`: only adds `18-19`, representative `19`.

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair12 | 54 | 929,732,000 | 0.9848 | 0.0008 | -0.03 | 0.058 | 0.573 |
| losslessbase54 pair18 | 54 | 929,732,000 | 0.9850 | 0.0007 | -0.02 | 0.059 | 0.579 |

Both passed materialized consistency checks with `mismatches=0` over 30,084 positions.

Level 10 XOT match checks with `--shuffle-seed 20260717`:

| Candidate | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|
| losslessbase54 pair12 | 100 paired matches | 0.5000 | +0.00 |
| losslessbase54 pair12 | 200 paired matches | 0.4975 | -0.03 |
| losslessbase54 pair18 | 100 paired matches | 0.4650 | -0.30 |

Interpretation:

- `18-19` is the main culprit for the alternate-seed weakness, despite having random-play direct error similar to
  `12-13`.
- `12-13` looks much safer in this measurement set. A promising next layout family is therefore:
  `lossless base + 12-13 + other non-18-19 pairs`, rather than blindly following RMSE pair cost.
- The conservative exact candidate remains `lossless auto grouped` at 55 groups / 944.8MB.
- The strongest practical lossy candidate is no longer settled by direct RMSE alone; `losslessbase54 pair12` is the
  safest measured lossy step, while `losslessbase48` remains a compact candidate that needs broader opening-set
  validation.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 22

### Non-18-19 Layouts From The Lossless Base

The alternate-seed checks in cycle 21 isolated `18-19` as the main risky pair. Generated new layouts that keep the
lossless `0-5` run, force the safer `12-13`, and explicitly avoid `18-19`.

Selected layouts:

- `losslessbase53 no18`: `12-13,14-15`
- `losslessbase52 no18`: `12-13,14-15,26-27`
- `losslessbase52 no18 pair20`: `12-13,14-15,20-21`
- `losslessbase52 no18 pair24`: `12-13,14-15,24-25`
- `losslessbase52 no18 pair16`: `12-13,14-15,16-17`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase53 lowcost | 53 | 914,614,416 | -105,823,018 | 0.9699 | 0.0014 | -0.05 | 0.12 | 0.81 | 1.671M |
| losslessbase53 no18 | 53 | 914,614,416 | -105,823,018 | 0.9696 | 0.0016 | -0.07 | 0.12 | 0.83 | 1.646M |
| losslessbase52 no18 | 52 | 899,496,832 | -120,940,602 | 0.9546 | 0.0023 | -0.04 | 0.18 | 1.04 | 1.625M |
| losslessbase52 no18 pair20 | 52 | 899,496,832 | -120,940,602 | 0.9544 | 0.0023 | -0.11 | 0.18 | 1.04 | not standalone-retimed |
| losslessbase52 no18 pair24 | 52 | 899,496,832 | -120,940,602 | 0.9544 | 0.0024 | -0.09 | 0.19 | 1.06 | not standalone-retimed |
| losslessbase52 no18 pair16 | 52 | 899,496,832 | -120,940,602 | 0.9541 | 0.0024 | -0.12 | 0.19 | 1.04 | not standalone-retimed |

All new models had `engine_mismatches=0`; materialized consistency checks passed with `mismatches=0` over
30,105 positions.

Level 10 XOT match checks against current fast, using same-source generic buildcheck binaries:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase53 lowcost | 20260717 | 200 paired matches | 0.4775 | -0.25 |
| losslessbase53 no18 | 20260717 | 100 paired matches | 0.5000 | +0.00 |
| losslessbase53 no18 | 20260717 | 200 paired matches | 0.4950 | -0.03 |
| losslessbase53 no18 | 57 | 100 paired matches | 0.5050 | +0.03 |
| losslessbase52 no18 | 20260717 | 100 paired matches | 0.4500 | -0.04 |
| losslessbase52 no18 pair20 | 20260717 | 100 paired matches | 0.4900 | -0.18 |
| losslessbase52 no18 pair24 | 20260717 | 100 paired matches | 0.4800 | -0.09 |
| losslessbase52 no18 pair16 | 20260717 | 100 paired matches | 0.4900 | -0.03 |
| losslessbase52 no18 pair16 | 20260717 | 200 paired matches | 0.4850 | -0.07 |

Notes:

- `losslessbase53 no18` has slightly worse random-play direct error than the old `losslessbase53 lowcost`, but it
  removes the alternate-seed weakness almost completely at the same 914.6MB file size.
- The extra 52-group compression is still not settled. Of the tested third-pair choices, `16-17` was best on the
  alternate seed, but its 200-match result still stayed below `losslessbase53 no18`.
- For the battle command, current fast should use `Egaroucid_for_Console_buildcheck_generic.exe`. The
  `Egaroucid_for_Console_shared_generic_buildcheck.exe` binary expects grouped EGEV10 and fails on the current
  EGEV4 resource.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 23

### Broader Opening-Seed Validation

Extended `losslessbase53 no18` to two more shuffle seeds and lengthened the weaker one to 200 paired matches.

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase53 no18 | 20260717 | 200 paired matches | 0.4950 | -0.03 |
| losslessbase53 no18 | 20260718 | 100 paired matches | 0.4950 | -0.11 |
| losslessbase53 no18 | 20260718 | 200 paired matches | 0.4875 | -0.10 |
| losslessbase53 no18 | 20260719 | 100 paired matches | 0.5050 | +0.03 |

The old `losslessbase53 lowcost` result on seed `20260717` was `0.4775 / -0.25`, so excluding `18-19` is still a
clear improvement. However, seed `20260718` shows that the 53-group candidate still has some measurable opening-set
risk.

### Same-Size Pair Replacement Probe

Generated another 53-group candidate with `12-13,16-17`, replacing the current `14-15` second pair:

`model/eval77_fm_losslessbase53_no18_pair16_init/eval_losslessbase53_no18_pair16.egev10`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| losslessbase53 no18 | 53 | 914,614,416 | 0.9696 | 0.0016 | -0.07 | 0.12 | 0.83 |
| losslessbase53 no18 pair16 | 53 | 914,614,416 | 0.9694 | 0.0015 | -0.08 | 0.13 | 0.86 |

The pair16 variant passed materialized consistency with `mismatches=0`, but its level 10 check on seed `20260717`
was only `0.4900 / -0.03` over 100 paired matches. It does not improve on the current `14-15` version.

### Conservative 54-Group Comparison

Because `losslessbase53 no18` dipped on seed `20260718`, also measured the safer one-lossy-pair candidate,
`losslessbase54 pair12`, on the same seed:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase54 pair12 | 20260717 | 200 paired matches | 0.4975 | -0.03 |
| losslessbase54 pair12 | 20260718 | 200 paired matches | 0.4950 | -0.07 |
| losslessbase53 no18 | 20260718 | 200 paired matches | 0.4875 | -0.10 |

Interpretation:

- `losslessbase53 no18` remains the best compact lossy candidate found so far: it saves 105.8MB vs EGEV4 and is much
  better than the previous 53-group layout that included `18-19`.
- `losslessbase54 pair12` is the safer lossy candidate: it saves 90.7MB vs EGEV4, 15.1MB less than the 53-group
  candidate, but the two measured alternate seeds are closer to parity.
- The exact `lossless auto grouped` candidate remains the conservative baseline at 944.8MB / 55 groups. The current
  practical choice is now a tradeoff between exact safety, one-pair safety, and the more compact 53-group no18
  layout; more opening seeds or a more targeted opening-discrepancy analysis are needed before calling the compact
  candidate settled.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 24

### More 53-Group Second-Pair Replacements

Generated three more same-size 53-group candidates to check whether the second lossy pair in
`losslessbase53 no18` should be something other than `14-15`.

All candidates keep:

- lossless `0-5`
- `12-13` representative `13`
- no `18-19`

Second-pair variants:

- `losslessbase53 no18 pair20`: `20-21`, representative `21`
- `losslessbase53 no18 pair24`: `24-25`, representative `25`
- `losslessbase53 no18 pair26`: `26-27`, representative `26`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| losslessbase53 no18 | 53 | 914,614,416 | 0.9696 | 0.0016 | -0.07 | 0.12 | 0.83 |
| losslessbase53 no18 pair16 | 53 | 914,614,416 | 0.9694 | 0.0015 | -0.08 | 0.13 | 0.86 |
| losslessbase53 no18 pair20 | 53 | 914,614,416 | 0.9697 | 0.0014 | -0.07 | 0.12 | 0.86 |
| losslessbase53 no18 pair24 | 53 | 914,614,416 | 0.9697 | 0.0015 | -0.05 | 0.13 | 0.88 |
| losslessbase53 no18 pair26 | 53 | 914,614,416 | 0.9699 | 0.0014 | -0.00 | 0.12 | 0.86 |

All three new variants had `engine_mismatches=0` and passed materialized consistency checks with `mismatches=0`
over 30,105 positions.

Level 10 XOT match checks against current fast:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase53 no18 | 20260717 | 200 paired matches | 0.4950 | -0.03 |
| losslessbase53 no18 | 20260718 | 200 paired matches | 0.4875 | -0.10 |
| losslessbase53 no18 pair16 | 20260717 | 100 paired matches | 0.4900 | -0.03 |
| losslessbase53 no18 pair20 | 20260717 | 100 paired matches | 0.4900 | -0.20 |
| losslessbase53 no18 pair24 | 20260717 | 100 paired matches | 0.4800 | -0.09 |
| losslessbase53 no18 pair26 | 20260718 | 100 paired matches | 0.5150 | +0.28 |
| losslessbase53 no18 pair26 | 20260718 | 200 paired matches | 0.5025 | +0.18 |
| losslessbase53 no18 pair26 | 20260717 | 100 paired matches | 0.4500 | -0.04 |

Interpretation:

- `pair26` is a useful warning case. It has the cleanest random-play direct metrics and strongly fixes seed
  `20260718`, but it collapses on seed `20260717`.
- `pair20` and `pair24` also do not beat the current `14-15` version on seed `20260717`.
- Among the measured 53-group two-lossy-pair candidates, `losslessbase53 no18` (`12-13,14-15`) remains the best
  balanced compact candidate. The safer practical candidate is still `losslessbase54 pair12`; the exact candidate is
  still `lossless auto grouped`.
- Direct error, sign disagreement, and random-play bias are not sufficient selection criteria. The next improvement
  should use opening-aware evidence, either by saving kifu for bad seeds or by building an opening-level discrepancy
  summary.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 25

### Opening-Level Kifu Check On Seed 20260718

Re-ran the seed `20260718` 100-paired checks with `--save-kifu` for the compact 53-group candidate and the safer
54-group candidate:

- `bin/transcript/20260717_losslessbase53_no18_seed20260718_100.tsv`
- `bin/transcript/20260717_losslessbase54_pair12_seed20260718_100.tsv`

Opening-level aggregation showed that the result is driven by very few openings:

| Candidate | openings | W/D/L by paired opening | candidate win rate | candidate average disc diff | nonzero paired openings |
|---|---:|---:|---:|---:|---|
| losslessbase53 no18 | 100 | 1/97/2 | 0.4950 | -0.11 | `14:+6`, `16:-6`, `84:-22` |
| losslessbase54 pair12 | 100 | 0/99/1 | 0.4950 | -0.11 | `84:-22` |

The common bad opening was `opening_idx=84`. For both candidates, the paired result was:

- candidate black: `+8`
- candidate white: `-30`
- paired sum: `-22`

Since this bad opening exists even in the one-lossy-pair 54-group model, it is caused by the shared `12-13`
representative choice rather than by the second pair in the compact 53-group model.

### 12-13 Representative Swap

Generated a 54-group model that keeps only the `12-13` lossy pair but uses representative phase `12` instead of
representative phase `13`:

`model/eval77_fm_losslessbase54_pair12first_init/eval_losslessbase54_pair12first.egev10`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair12 | 54 | 929,732,000 | 0.9848 | 0.0008 | -0.03 | 0.058 | 0.573 | not retimed |
| losslessbase54 pair12first | 54 | 929,732,000 | 0.9848 | 0.0007 | +0.04 | 0.065 | 0.646 | 1.630M |

The representative-12 model passed materialized consistency with `mismatches=0` over 30,105 positions.

Level 10 XOT match checks:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase54 pair12 | 20260717 | 200 paired matches | 0.4975 | -0.03 |
| losslessbase54 pair12 | 20260718 | 200 paired matches | 0.4950 | -0.07 |
| losslessbase54 pair12first | 20260718 | 100 paired matches | 0.5050 | +0.04 |
| losslessbase54 pair12first | 20260718 | 200 paired matches | 0.4975 | -0.02 |
| losslessbase54 pair12first | 20260717 | 100 paired matches | 0.4950 | -0.07 |

The saved-kifu seed `20260718` 100-paired run for `pair12first`:

- `bin/transcript/20260717_losslessbase54_pair12first_seed20260718_100.tsv`
- W/D/L by paired opening: `1/99/0`
- candidate win rate: `0.5050`
- candidate average disc diff: `+0.04`
- only nonzero paired opening: `65:+8`
- `opening_idx=84` changed from `-22` to `0`

Interpretation:

- Opening-aware analysis found a real improvement that random-play RMSE would not select: representative `12`
  has worse RMSE but fixes the common bad opening from representative `13`.
- `losslessbase54 pair12first` is now the strongest conservative lossy candidate measured on seed `20260718`.
- It is still 54 groups / 929.7MB, so it saves 90.7MB versus EGEV4 but 15.1MB less than the compact 53-group no18
  candidate.
- A next promising experiment is to rebuild the compact 53-group family with `12-13` representative `12`, then test
  whether adding a second pair can keep the opening-aware gain while recovering the extra 15.1MB saving.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 26

### Compact 53-Group Layout With 12-13 Representative 12

Generated the compact 53-group version suggested by cycle 25:

`model/eval77_fm_losslessbase53_no18_pair12first_init/eval_losslessbase53_no18_pair12first.egev10`

Layout:

- lossless `0-5`, representative `0`
- `12-13`, representative `12`
- `14-15`, representative `15`
- no `18-19`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | delta vs EGEV4 | exact rate | sign disagree | bias | MAE | RMSE | selected eval/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase53 no18 | 53 | 914,614,416 | -105,823,018 | 0.9696 | 0.0016 | -0.07 | 0.12 | 0.83 | 1.646M |
| losslessbase53 no18 pair12first | 53 | 914,614,416 | -105,823,018 | 0.9695 | 0.0016 | +0.00 | 0.13 | 0.87 | 1.662M |
| losslessbase54 pair12first | 54 | 929,732,000 | -90,705,434 | 0.9848 | 0.0007 | +0.04 | 0.065 | 0.646 | 1.630M |

The compact representative-12 model had `engine_mismatches=0` and passed materialized consistency with
`mismatches=0` over 30,105 positions.

Level 10 XOT match checks:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| losslessbase53 no18 | 20260718 | 200 paired matches | 0.4875 | -0.10 |
| losslessbase53 no18 pair12first | 20260718 | 100 paired matches | 0.5000 | +0.00 |
| losslessbase53 no18 pair12first | 20260718 | 200 paired matches | 0.4925 | -0.03 |
| losslessbase53 no18 pair12first | 20260717 | 100 paired matches | 0.4950 | -0.07 |
| losslessbase54 pair12first | 20260718 | 200 paired matches | 0.4975 | -0.02 |

Saved-kifu opening aggregation for
`bin/transcript/20260717_losslessbase53_no18_pair12first_seed20260718_100.tsv`:

- W/D/L by paired opening: `1/98/1`
- candidate win rate: `0.5000`
- candidate average disc diff: `+0.00`
- nonzero paired openings: `14:+6`, `16:-6`
- `opening_idx=84` changed from `-22` in the representative-13 layout to `0`

Interpretation:

- The compact 53-group representative-12 model successfully carries over the opening84 fix while keeping the same
  914.6MB size as the previous compact candidate.
- It improves the weak seed `20260718` result from `0.4875 / -0.10` to `0.4925 / -0.03`, but it still does not
  beat the safer 54-group `pair12first` candidate on that seed.
- Current practical ranking from the measured evidence:
  exact safety: `lossless auto grouped` > conservative lossy: `losslessbase54 pair12first` > compact lossy:
  `losslessbase53 no18 pair12first` / `losslessbase53 no18`.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 27

### Second-Pair Search On The Representative-12 Base

Generated four more compact 53-group candidates using the opening-aware `12-13` representative `12` base, replacing
the second pair:

- `losslessbase53 no18 pair12first pair16`: `16-17`, representative `17`
- `losslessbase53 no18 pair12first pair20`: `20-21`, representative `21`
- `losslessbase53 no18 pair12first pair24`: `24-25`, representative `25`
- `losslessbase53 no18 pair12first pair26`: `26-27`, representative `26`

All keep:

- lossless `0-5`
- `12-13`, representative `12`
- no `18-19`

Direct score check over 10,000 random-play games / 601,782 positions:

| Model | FM groups | bytes | exact rate | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair12first + 14-15 | 53 | 914,614,416 | 0.9695 | 0.0016 | +0.00 | 0.13 | 0.87 |
| pair12first + 16-17 | 53 | 914,614,416 | 0.9693 | 0.0015 | -0.01 | 0.13 | 0.91 |
| pair12first + 20-21 | 53 | 914,614,416 | 0.9695 | 0.0014 | +0.01 | 0.13 | 0.90 |
| pair12first + 24-25 | 53 | 914,614,416 | 0.9696 | 0.0015 | +0.02 | 0.13 | 0.93 |
| pair12first + 26-27 | 53 | 914,614,416 | 0.9698 | 0.0013 | +0.07 | 0.13 | 0.90 |

All four new variants had `engine_mismatches=0` and passed materialized consistency checks with `mismatches=0`
over 30,105 positions.

Level 10 XOT match checks:

| Candidate | shuffle seed | sample | candidate win rate | candidate average disc diff |
|---|---:|---:|---:|---:|
| pair12first + 14-15 | 20260717 | 100 paired matches | 0.4950 | -0.07 |
| pair12first + 14-15 | 20260718 | 200 paired matches | 0.4925 | -0.03 |
| pair12first + 14-15 | 20260719 | 100 paired matches | 0.5000 | +0.00 |
| pair12first + 16-17 | 20260717 | 100 paired matches | 0.4850 | -0.04 |
| pair12first + 20-21 | 20260717 | 100 paired matches | 0.4850 | -0.23 |
| pair12first + 24-25 | 20260717 | 100 paired matches | 0.4750 | -0.16 |
| pair12first + 26-27 | 20260717 | 100 paired matches | 0.4450 | -0.11 |
| losslessbase54 pair12first | 20260719 | 100 paired matches | 0.5000 | +0.00 |

Interpretation:

- The representative-12 base did not make the second-pair search monotonic. `26-27` again has the cleanest direct
  metrics but collapses on seed `20260717`.
- `16-17`, `20-21`, and `24-25` also do not beat the current `14-15` second pair.
- The best compact measured candidate remains `losslessbase53 no18 pair12first` (`12-13` representative `12`,
  `14-15` representative `15`), at 914.6MB and 53 FM groups.
- The best safer lossy candidate remains `losslessbase54 pair12first`, at 929.7MB and 54 FM groups.
- The next useful work is no longer blind adjacent-pair selection. It should either:
  - accumulate more saved-kifu opening-level evidence for the two surviving candidates, or
  - build a small opening-level evaluator that compares candidate and fast decisions/scores on the exact XOT
    openings that drive nonzero paired outcomes.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 28

### Reproducible Opening-Level Kifu Analyzer

Added a reusable TSV analyzer:

`src/tools/evaluation/analyze_battle_kifu_by_opening.py`

It reads `battle_parallel_nonstop_gtp.py --save-kifu` TSVs and reports:

- paired-battle W/D/L, win rate, and average disc difference
- opening-aggregated W/D/L, win rate, and average disc difference
- nonzero openings sorted by absolute impact
- multi-run opening comparison by average disc difference

Validation command:

`python src/tools/evaluation/analyze_battle_kifu_by_opening.py --top 20 --run compact_old=bin/transcript/20260717_losslessbase53_no18_seed20260718_100.tsv --run safe_old=bin/transcript/20260717_losslessbase54_pair12_seed20260718_100.tsv --run safe_first=bin/transcript/20260717_losslessbase54_pair12first_seed20260718_100.tsv --run compact_first=bin/transcript/20260717_losslessbase53_no18_pair12first_seed20260718_100.tsv`

The tool reproduced the manual seed `20260718` opening analysis:

| Run | paired battles | openings | battle W/D/L | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---:|---|
| compact old | 100 | 100 | 1/97/2 | 0.4950 | -0.11 | `84:-11`, `14:+3`, `16:-3` |
| safe old | 100 | 100 | 0/99/1 | 0.4950 | -0.11 | `84:-11` |
| safe first | 100 | 100 | 1/99/0 | 0.5050 | +0.04 | `65:+4` |
| compact first | 100 | 100 | 1/98/1 | 0.5000 | +0.00 | `14:+3`, `16:-3` |

The comparison table shows the key improvement directly:

- `opening_idx=84`: representative-13 layouts `-11`; representative-12 layouts `0`
- `opening_idx=65`: only `safe first` gains `+4`
- `opening_idx=14/16`: only compact layouts move, and the representative-12 compact layout balances `+3` and `-3`

### Saved-Kifu Check On Seed 20260717

Also saved opening-level TSVs for the two surviving representative-12 candidates on seed `20260717`:

- `bin/transcript/20260717_losslessbase53_no18_pair12first_seed20260717_100.tsv`
- `bin/transcript/20260717_losslessbase54_pair12first_seed20260717_100.tsv`

Analyzer result:

| Run | paired battles | openings | battle W/D/L | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---:|---|
| compact first seed 20260717 | 100 | 100 | 0/99/1 | 0.4950 | -0.07 | `40:-7` |
| safe first seed 20260717 | 100 | 100 | 0/99/1 | 0.4950 | -0.07 | `40:-7` |
| compact first seed 20260718 | 100 | 100 | 1/98/1 | 0.5000 | +0.00 | `14:+3`, `16:-3` |
| safe first seed 20260718 | 100 | 100 | 1/99/0 | 0.5050 | +0.04 | `65:+4` |

Interpretation:

- Seed `20260717` is driven by a single common bad opening, `opening_idx=40`, and both surviving candidates have the
  same `-7` average disc impact there.
- This means the remaining seed `20260717` weakness is not caused by the compact candidate's second pair. It exists
  even in the safer one-lossy-pair representative-12 model.
- The next improvement target is therefore the shared `12-13` representative-12 approximation itself on
  `opening_idx=40`, or accepting the safer candidate as the best lossy tradeoff found so far.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 29

### Mapping Opening Indices Back To XOT Lines

Extended `src/tools/evaluation/analyze_battle_kifu_by_opening.py` with:

- `--problem-file`
- `--shuffle-seed`

When both are supplied, the analyzer reproduces the same opening shuffle as `battle_parallel_nonstop_gtp.py` and
prints the actual XOT line for each nonzero `opening_idx`.

Seed `20260717` surviving-candidate analysis:

`python src/tools/evaluation/analyze_battle_kifu_by_opening.py --top 10 --problem-file bin/problem/xot/openingslarge.txt --shuffle-seed 20260717 --run compact17=bin/transcript/20260717_losslessbase53_no18_pair12first_seed20260717_100.tsv --run safe17=bin/transcript/20260717_losslessbase54_pair12first_seed20260717_100.tsv`

Key line:

- `opening_idx=40`: `f5f4f3g4e3d6c4e6`, compact first `-7`, safe first `-7`

Seed `20260718` key lines:

- `opening_idx=84`: `f5d6c3f4f6f3c5c6`, representative-13 layouts `-11`, representative-12 layouts `0`
- `opening_idx=65`: `f5f6c4g5e6c5b6d3`, safe first `+4`
- `opening_idx=14`: `f5f4g3g6e3f2d3g5`, compact layouts `+3`
- `opening_idx=16`: `f5f6d3c5d6c7d7g5`, compact layouts `-3`

### Targeted One-Opening Checks

Created temporary one-line problem files under `bin/transcript/` for targeted checks:

- `target_opening40_seed20260717.txt`: `f5f4f3g4e3d6c4e6`
- `target_opening84_seed20260718.txt`: `f5d6c3f4f6f3c5c6`

Measured one paired match for the 54-group one-pair representative variants:

| Opening | representative 13 | representative 12 |
|---|---:|---:|
| `f5f4f3g4e3d6c4e6` | 0.5000 / +0.00 | 0.0000 / -7.00 |
| `f5d6c3f4f6f3c5c6` | 0.0000 / -11.00 | 0.5000 / +0.00 |

Interpretation:

- Representative 12 and representative 13 each fix a different sharp opening.
- The seed-level tradeoff is now concrete:
  - representative 13 loses `f5d6c3f4f6f3c5c6`
  - representative 12 loses `f5f4f3g4e3d6c4e6`
- This explains why neither single-representative `12-13` lossy merge is uniformly safe. The exact 55-group lossless
  candidate avoids both failures; the 54-group representative-12 candidate is the best lossy tradeoff found so far,
  but it still has a known targeted loss.
- A true next improvement would need something more expressive than one representative for phases `12-13` (for
  example keeping them separate, training a new shared table, or an opening/phase-aware exception). Blind adjacent
  pair selection has reached diminishing returns.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 30

### Keep 12 And 13 Separate, Merge 14-15 Instead

The targeted opening checks showed that merging `12-13` with either representative creates a sharp failure:

- representative `13` loses `f5d6c3f4f6f3c5c6`
- representative `12` loses `f5f4f3g4e3d6c4e6`

Generated a new 54-group layout with the same size as the previous conservative lossy candidates, but with phases
`12` and `13` kept separate and only `14-15` merged:

- `model/eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`
- FM groups: `54`
- size: `929,732,000` bytes
- saved vs EGEV4 baseline: `90,705,434` bytes
- lossy group: `14-15`, representative `15`

Direct and implementation checks:

| Candidate | groups | bytes | exact | sign disagree | bias | MAE | RMSE | selected nps | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair14 | 54 | 929,732,000 | 0.9847 | 0.0009 | -0.04 | 0.061 | 0.588 | 1.644M | 0 / 30,105 |
| losslessbase54 pair14first | 54 | 929,732,000 | 0.9844 | 0.0008 | +0.06 | 0.073 | 0.685 | 1.692M | 0 / 30,105 |

Targeted one-opening checks for `losslessbase54 pair14`:

| Opening | result |
|---|---:|
| `f5f4f3g4e3d6c4e6` | 0.5000 / +0.00 |
| `f5d6c3f4f6f3c5c6` | 0.5000 / +0.00 |

This confirms that keeping `12` and `13` separate removes both sharp failures found in Cycle 29.

### XOT Battle Results

Measured against current fast (`Egaroucid_for_Console_buildcheck_generic.exe`) at level 10:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---|
| losslessbase54 pair14 | 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14 | 20260717 | 200 | 0.5025 | +0.01 | `152:+1` |
| losslessbase54 pair14 | 20260718 | 100 | 0.5000 | +0.00 | `14:+3`, `16:-3` |
| losslessbase54 pair14 | 20260718 | 200 | 0.4975 | +0.00 | `14:+3`, `16:-3`, `187:+2`, `171:-1`, `175:-1` |
| losslessbase54 pair14 | 20260719 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14first | 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14first | 20260717 | 200 | 0.4925 | -0.04 | `149:-5`, `112:-1`, `138:-1` |
| losslessbase54 pair14first | 20260718 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14first | 20260718 | 200 | 0.5025 | +0.03 | `180:+5` |
| losslessbase54 pair14first | 20260719 | 100 | 0.5000 | +0.04 | `72:+5`, `57:-1` |

The representative-14 variant looked quieter in the first 100 games, but the 200-game seed `20260717` extension
exposed three bad openings. The representative-15 variant has better direct RMSE and the better measured 200-game
seed `20260717` result.

Current ranking:

- exact conservative: `eval77_fm_lossless_auto_init/eval_lossless_auto.egev10` (55 groups, exact, 944.8MB)
- best measured lossy conservative: `eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`
  (54 groups, 929.7MB, keeps `12` and `13` separate, avoids both known targeted failures)
- prior lossy candidates that merge `12-13` are now superseded for practical safety, because each representative has
  a known sharp opening loss.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 31

### Selector-Guided Single-Pair Alternatives

Ran the low-cost layout selector with fixed lossless run `0-5:0`, target `54` groups, and excluded `12-13`:

| Metric | selected pair | representative | note |
|---|---|---:|---|
| RMSE | `18-19` | 19 | already known bad from battle (`0.4650 / -0.30` on seed `20260717` 100) |
| sign | `13-14` | 13 | new candidate |
| MAE | `13-14` | 14 | new candidate |

Generated and checked both `13-14` representatives:

- `model/eval77_fm_losslessbase54_pair13_init/eval_losslessbase54_pair13.egev10`
- `model/eval77_fm_losslessbase54_pair13last_init/eval_losslessbase54_pair13last.egev10`

Direct and implementation checks:

| Candidate | groups | bytes | exact | sign disagree | bias | MAE | RMSE | selected nps | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair13 | 54 | 929,732,000 | 0.9849 | 0.0007 | -0.03 | 0.063 | 0.638 | 1.651M | 0 / 30,105 |
| losslessbase54 pair13last | 54 | 929,732,000 | 0.9851 | 0.0009 | +0.03 | 0.060 | 0.603 | not retimed | 0 / 30,105 |
| losslessbase54 pair14 | 54 | 929,732,000 | 0.9847 | 0.0009 | -0.04 | 0.061 | 0.588 | 1.644M | 0 / 30,105 |

Both `13-14` variants drew the four targeted openings checked:

- `f5f4f3g4e3d6c4e6`
- `f5d6c3f4f6f3c5c6`
- `f5f4g3g6e3f2d3g5`
- `f5f6d3c5d6c7d7g5`

### XOT Battle Results

Measured the representative-13 `13-14` model against current fast:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---|
| losslessbase54 pair13 | 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair13 | 20260717 | 200 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair13 | 20260718 | 100 | 0.5050 | +0.04 | `65:+4` |
| losslessbase54 pair13 | 20260718 | 200 | 0.5025 | +0.04 | `180:+5`, `65:+4`, `175:-1` |
| losslessbase54 pair13 | 20260719 | 100 | 0.4950 | -0.05 | `98:-5` |
| losslessbase54 pair13 | 20260719 | 200 | 0.4975 | -0.03 | `98:-5` |

The representative-14 `13-14` model was discarded after the first 100-game pass:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---|
| losslessbase54 pair13last | 20260717 | 100 | 0.5050 | +0.01 | `30:+1` |
| losslessbase54 pair13last | 20260718 | 100 | 0.4950 | -0.04 | `65:-4` |

Direct comparison of the two surviving 54-group lossy candidates over 200-game seeds:

| Candidate | seed 20260717 | seed 20260718 | seed 20260719 |
|---|---:|---:|---:|
| losslessbase54 pair13 | 0.5000 / +0.00 | 0.5025 / +0.04 | 0.4975 / -0.03 |
| losslessbase54 pair14 | 0.5025 / +0.01 | 0.4975 / +0.00 | 0.5025 / +0.02 |

Interpretation:

- `pair13` is a real contender and has the best direct sign-disagreement rate among the tested 54-group lossy
  candidates, but seed `20260719` exposes a single sharp `-5` opening: `f5f4g3e6d6c4e7d7`.
- `pair14` has slightly worse sign disagreement but better RMSE, better seed `20260717` and `20260719` battle
  results, and no measured negative opening larger than `-3` in the 200-game XOT passes.
- Current best measured lossy conservative candidate remains
  `model/eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 32

### Next Selector Candidates After Excluding Known Pairs

Ran the selector again with fixed lossless run `0-5:0`, target `54` groups, and excluded:

- `12-13`: sharp tradeoff from Cycle 29
- `18-19`: known bad battle result
- `13-14`: measured in Cycle 31
- `14-15`: current best `pair14` family

Results:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `26-27` | 26 |
| MAE | `26-27` | 26 |
| sign | `16-17` | 16 |

Generated both 54-group layouts:

- `model/eval77_fm_losslessbase54_pair26_init/eval_losslessbase54_pair26.egev10`
- `model/eval77_fm_losslessbase54_pair16first_init/eval_losslessbase54_pair16first.egev10`

Direct and implementation checks:

| Candidate | groups | bytes | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair26 | 54 | 929,732,000 | 0.9850 | 0.0006 | +0.03 | 0.064 | 0.632 | 0 / 30,105 |
| losslessbase54 pair16first | 54 | 929,732,000 | 0.9845 | 0.0006 | +0.06 | 0.073 | 0.686 | 0 / 30,105 |
| losslessbase54 pair14 | 54 | 929,732,000 | 0.9847 | 0.0009 | -0.04 | 0.061 | 0.588 | 0 / 30,105 |

The direct sign metric prefers `pair26` / `pair16first`, but this again did not predict practical opening stability.

### Targeted Checks

`losslessbase54 pair26` drew the old sharp openings and improved two newer targeted lines:

| Opening | result |
|---|---:|
| `f5f4f3g4e3d6c4e6` | 0.5000 / +0.00 |
| `f5d6c3f4f6f3c5c6` | 0.5000 / +0.00 |
| `f5f4g3e6d6c4e7d7` | 1.0000 / +7.00 |
| `f5f6c4g5e6c5b6d3` | 1.0000 / +3.00 |

`losslessbase54 pair16first` also drew the first three and improved `f5f6c4g5e6c5b6d3`:

| Opening | result |
|---|---:|
| `f5f4f3g4e3d6c4e6` | 0.5000 / +0.00 |
| `f5d6c3f4f6f3c5c6` | 0.5000 / +0.00 |
| `f5f4g3e6d6c4e7d7` | 0.5000 / +0.00 |
| `f5f6c4g5e6c5b6d3` | 1.0000 / +5.00 |

### XOT Battle Results

`losslessbase54 pair26` looked good in targeted checks but was too unstable in broader XOT samples:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | largest negative opening |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair26 | 20260717 | 100 | 0.4500 | -0.04 | 22 | `26:-5` |
| losslessbase54 pair26 | 20260718 | 100 | 0.5000 | +0.22 | 24 | `53:-8` |
| losslessbase54 pair26 | 20260719 | 100 | 0.5000 | +0.05 | 18 | `66:-11` |

`losslessbase54 pair16first` was better than `pair26` on seeds `20260717` and `20260718`, but seed `20260719`
exposed a sharper loss than current `pair14`:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | largest negative opening |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair16first | 20260717 | 100 | 0.5050 | +0.03 | 7 | `80:-6` |
| losslessbase54 pair16first | 20260718 | 100 | 0.5050 | +0.06 | 5 | `39:-4` |
| losslessbase54 pair16first | 20260719 | 100 | 0.4800 | -0.10 | 6 | `33:-8` |

Interpretation:

- Direct sign-disagreement is not sufficient as a selector for practical strength. `pair26` had the best direct sign
  rate in this batch but produced many nonzero XOT openings and a poor seed `20260717` win rate.
- `pair16first` is less noisy than `pair26`, but the seed `20260719` result is worse than the current `pair14`
  candidate and includes a `-8` opening.
- Current best measured lossy conservative candidate remains
  `model/eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 33

### Additional Seed Validation For Current Best Candidate

After rejecting `pair26` and `pair16first`, ran two new unseen shuffle seeds for the current best `pair14` candidate:

- `model/eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`
- current fast: `Egaroucid_for_Console_buildcheck_generic.exe`
- level: `10`
- saved kifu:
  - `bin/transcript/20260717_losslessbase54_pair14_seed20260720_100.tsv`
  - `bin/transcript/20260717_losslessbase54_pair14_seed20260721_100.tsv`

Results:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | largest negative opening |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair14 | 20260720 | 100 | 0.5000 | -0.02 | 2 | `2:-3` |
| losslessbase54 pair14 | 20260721 | 100 | 0.5000 | +0.00 | 0 | none |

The new seeds did not expose a sharp failure. Across the five measured seeds so far:

| Candidate | seed | paired matches | win rate | average disc diff |
|---|---:|---:|---:|---:|
| losslessbase54 pair14 | 20260717 | 200 | 0.5025 | +0.01 |
| losslessbase54 pair14 | 20260718 | 200 | 0.4975 | +0.00 |
| losslessbase54 pair14 | 20260719 | 200 | 0.5025 | +0.02 |
| losslessbase54 pair14 | 20260720 | 100 | 0.5000 | -0.02 |
| losslessbase54 pair14 | 20260721 | 100 | 0.5000 | +0.00 |

Interpretation:

- `pair14` remains the best measured lossy conservative candidate.
- Its observed negative openings are small compared with rejected candidates:
  - `pair13`: `-5`
  - `pair16first`: `-8`
  - `pair26`: `-11`
  - `pair14`: no measured negative opening below `-3` so far
- The exact conservative candidate remains `eval77_fm_lossless_auto_init/eval_lossless_auto.egev10`; `pair14` is the
  current best lossy candidate when saving an additional 15.1MB matters.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 34

### Rechecking 53-Group Candidates With 12-13 Kept Separate

Older 53-group candidates mostly merged `12-13`, which Cycle 29 showed is structurally unsafe because each
representative loses a different sharp opening. Re-ran the selector for target `53` groups with:

- fixed lossless run: `0-5:0`
- excluded pair: `12-13`

Raw selector results:

| Metric | selected pairs | representatives | note |
|---|---|---|---|
| RMSE | `13-14`, `18-19` | `14`, `19` | includes known bad `18-19` |
| MAE | `13-14`, `18-19` | `14`, `19` | includes known bad `18-19` |
| sign | `13-14`, `16-17` | `13`, `16` | includes `16-17`, which later showed a `-8` opening |

After also excluding known bad `18-19`, RMSE/MAE selected `13-14` + `26-27`, while sign still selected
`13-14` + `16-17`.

Because the current best 54-group lossy candidate is `14-15` representative `15`, also ran selector with
`14-15:15` fixed. The next selected pairs were:

| Metric | additional pair | representative |
|---|---|---:|
| RMSE | `26-27` | 26 |
| MAE | `26-27` | 26 |
| sign | `16-17` | 16 |

Generated two 53-group candidates that preserve the current best `14-15:15` merge and add one more pair:

- `model/eval77_fm_losslessbase53_pair14_pair26_init/eval_losslessbase53_pair14_pair26.egev10`
- `model/eval77_fm_losslessbase53_pair14_pair16first_init/eval_losslessbase53_pair14_pair16first.egev10`

Direct and implementation checks:

| Candidate | groups | bytes | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| losslessbase53 pair14+pair26 | 53 | 914,614,416 | 0.9697 | 0.0015 | -0.01 | 0.125 | 0.863 | 0 / 30,105 |
| losslessbase53 pair14+pair16first | 53 | 914,614,416 | 0.9692 | 0.0015 | +0.02 | 0.134 | 0.904 | 0 / 30,105 |

Targeted checks:

| Candidate | targeted opening | result | interpretation |
|---|---|---:|---|
| losslessbase53 pair14+pair26 | `f5f6c4e3f4c5b5g5` | 0.0000 / -11.00 | same sharp failure as `pair26` |
| losslessbase53 pair14+pair16first | `f5f4c3g6g3c4e3f2` | 0.0000 / -8.00 | same sharp failure as `pair16first` |
| losslessbase53 pair14+pair26 | `f5f4f3g4e3d6c4e6` | 0.5000 / +0.00 | old `12-13` failure avoided |
| losslessbase53 pair14+pair26 | `f5d6c3f4f6f3c5c6` | 0.5000 / +0.00 | old `12-13` failure avoided |

Interpretation:

- Keeping `12` and `13` separate can preserve the Cycle 29 targeted fixes at 53 groups, but the additional pair
  needed to reach 53 groups introduces new sharp failures.
- The best-looking 53-group candidate by direct RMSE (`pair14+pair26`) immediately inherits the `-11` opening from
  standalone `pair26`.
- The sign-guided 53-group route immediately inherits the `-8` opening from standalone `pair16first`.
- Current compact 53-group candidates are therefore still not better than the 54-group `pair14` candidate. The
  current ranking remains:
  - exact conservative: `eval77_fm_lossless_auto_init/eval_lossless_auto.egev10`
  - best measured lossy conservative: `eval77_fm_losslessbase54_pair14_init/eval_losslessbase54_pair14.egev10`
  - 53-group layouts: smaller, but currently fail targeted safety checks

## Addendum: 2026-07-17 Measurement And Improvement Cycle 35

### GGS Random Board Validation

Broadened validation beyond XOT move-sequence openings by using the board-problem set:

- `bin/problem/ggs_random_openings/14_random_setup2/0000000.txt`
- 1,000 board positions
- `battle_parallel_nonstop.py` with console protocol engines
- default shuffle seed `57`

First measured the current best XOT/direct candidate:

| Candidate | set | paired matches | win rate | average disc diff | nonzero boards |
|---|---|---:|---:|---:|---|
| losslessbase54 pair14 | GGS board seed 57 | 100 | 0.4950 | -0.08 | `83:-8` |
| losslessbase54 pair14 | GGS board seed 57 | 200 | 0.4950 | -0.07 | `83:-8`, `195:-5` |

The first bad board was:

- board index `83`
- line: `-------------X---OO-X----OXXXXX--O-OO-----O--------------------- X`
- `losslessbase54 pair14`: `0.0000 / -8.00`

Targeted checks on that same board:

| Candidate | result |
|---|---:|
| losslessbase54 pair14first | 0.5000 / +0.00 |
| losslessbase54 pair13 | 0.5000 / +0.00 |
| losslessbase54 pair13last | 0.0000 / -8.00 |
| losslessbase54 pair26 | 0.5000 / +0.00 |

This means the `14-15` representative choice matters outside XOT: representative `15` is better on previous XOT
seeds, but representative `14` avoids this GGS board failure.

### Rechecking Pair14first As A Cross-Set Candidate

Because `pair14first` fixed the GGS board `83`, remeasured it on additional XOT seeds and on the same GGS board set.

XOT:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---:|---|
| losslessbase54 pair14first | 20260719 | 200 | 0.4975 | +0.01 | `72:+5`, `113:+4`, `122:-4`, `175:-3`, `57:-1` |
| losslessbase54 pair14first | 20260720 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14first | 20260721 | 100 | 0.5050 | +0.10 | `10:+10` |

GGS board set:

| Candidate | set | paired matches | win rate | average disc diff | nonzero boards |
|---|---|---:|---:|---:|---|
| losslessbase54 pair13 | GGS board seed 57 | 100 | 0.4950 | -0.01 | `15:-1` |
| losslessbase54 pair14first | GGS board seed 57 | 100 | 0.5050 | +0.03 | `78:+3` |
| losslessbase54 pair14first | GGS board seed 57 | 200 | 0.5025 | +0.01 | `78:+3` |

Direct comparison of the two `14-15` representatives over the measured mixed suite:

| Candidate | XOT measured seeds | GGS board seed 57 | observed worst board/opening |
|---|---:|---:|---|
| losslessbase54 pair14 | roughly balanced, best XOT seed `20260717/19` | 0.4950 / -0.07 over 200 | `GGS 83:-8` |
| losslessbase54 pair14first | weaker on XOT seed `20260717`, better on `20260721` | 0.5025 / +0.01 over 200 | `XOT 149:-5`, `XOT 122:-4` |

Interpretation:

- `pair14` remains the best direct-RMSE and XOT-first candidate.
- `pair14first` is now a serious mixed-suite candidate: it has worse direct RMSE, but it fixes the measured GGS
  failures and has no observed negative board/opening below `-5`.
- The ranking is now evidence-dependent rather than one-dimensional:
  - choose exact 55-group `lossless_auto` when score equality matters most
  - choose `pair14` when prioritizing direct RMSE and the original XOT-centered validation
  - choose `pair14first` for the best measured XOT+GGS mixed-suite behavior so far
- More validation should compare `pair14` and `pair14first` on additional non-XOT board sets or longer GGS/XOT mixed
  runs before making either lossy model the final recommendation.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 36

### Added Shuffle Seed Control To Board-Problem Battles

The GTP battle driver already had `--shuffle-seed`, but `battle_parallel_nonstop.py` used by board-problem files was
hardcoded to seed `57`. Added the same option to `battle_parallel_nonstop.py`:

- `--shuffle-seed`
- default remains `57`
- the selected seed is printed in the battle log

Smoke-tested the new option on the GGS board set with one paired match:

`python battle_parallel_nonstop.py 10 1 1 2 1 --shuffle-seed 20260722 --problem-file problem/ggs_random_openings/14_random_setup2/0000000.txt ...`

The run completed and printed `shuffle seed: 20260722`.

### Additional GGS Board Seeds

Used the new seed option to compare the two `14-15` representative choices on additional GGS board orderings.

GGS seed `20260722`:

| Candidate | paired matches | win rate | average disc diff | nonzero boards |
|---|---:|---:|---:|---|
| losslessbase54 pair14 | 100 | 0.5050 | +0.03 | `60:+3` |
| losslessbase54 pair14first | 100 | 0.5000 | +0.00 | 0 |

GGS seed `20260723`:

| Candidate | paired matches | win rate | average disc diff | nonzero boards |
|---|---:|---:|---:|---|
| losslessbase54 pair14 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair14first | 100 | 0.5000 | +0.00 | 0 |

Updated GGS board summary:

| Candidate | seed 57 | seed 20260722 | seed 20260723 | worst observed GGS board |
|---|---:|---:|---:|---|
| losslessbase54 pair14 | 0.4950 / -0.07 over 200 | 0.5050 / +0.03 over 100 | 0.5000 / +0.00 over 100 | `83:-8` |
| losslessbase54 pair14first | 0.5025 / +0.01 over 200 | 0.5000 / +0.00 over 100 | 0.5000 / +0.00 over 100 | none negative |

Interpretation:

- `pair14`'s GGS seed `57` weakness did not repeat on the two new GGS shuffles, but the original seed still exposes
  real board-specific losses (`83:-8`, `195:-5`).
- `pair14first` remains quieter on the measured GGS board suite: no negative nonzero GGS board has been observed
  across seed `57`, `20260722`, and `20260723`.
- The mixed-suite tradeoff is now clearer:
  - `pair14`: better direct RMSE and stronger XOT-first evidence, but known GGS board losses
  - `pair14first`: worse direct RMSE, but better measured GGS behavior and acceptable newer XOT seeds
- More evidence is still needed before selecting a final lossy default, but `pair14first` is the best measured
  cross-set lossy candidate so far.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 37

### Rechecking Pair13 As A Mixed-Suite Candidate

`losslessbase54 pair13` also fixed the GGS board `83` targeted failure from `pair14`, and it has strong direct
sign-disagreement metrics. Extended its GGS board validation to match the newer `pair14` / `pair14first` comparisons.

GGS board results:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero boards |
|---|---:|---:|---:|---:|---|
| losslessbase54 pair13 | 57 | 100 | 0.4950 | -0.01 | `15:-1` |
| losslessbase54 pair13 | 57 | 200 | 0.4950 | -0.03 | `195:-5`, `15:-1` |
| losslessbase54 pair13 | 20260722 | 100 | 0.5100 | +0.17 | `15:+14`, `60:+3` |
| losslessbase54 pair13 | 20260723 | 100 | 0.4900 | -0.13 | `44:-12`, `20:-1` |

The seed `20260723` result is the important discriminator:

- board index `44`
- line: `-------------O------OO----XOO------XOX----O-OX------OX---------- X`
- result: `0.0000 / -12.00`

Updated cross-set comparison of the three relevant 54-group lossy layouts:

| Candidate | Direct RMSE | XOT summary | GGS summary | worst observed nonzero |
|---|---:|---|---|---|
| losslessbase54 pair14 | 0.588 | strongest XOT/direct candidate | GGS seed 57 has `83:-8`, `195:-5`; seeds 22/23 clean or positive | `GGS 83:-8` |
| losslessbase54 pair14first | 0.685 | weaker on XOT seed 17, acceptable on newer XOT seeds | no negative GGS board across seeds 57/22/23 | `XOT 149:-5`, `XOT 122:-4` |
| losslessbase54 pair13 | 0.638 | good XOT except seed 19 `98:-5` | GGS seed 23 has `44:-12` | `GGS 44:-12` |

Interpretation:

- `pair13` is no longer the best mixed-suite alternative: the GGS seed `20260723` `-12` board is worse than the
  observed failures for both `pair14` and `pair14first`.
- `pair14` remains best if the objective is direct RMSE / XOT-first validation.
- `pair14first` remains the best measured XOT+GGS mixed-suite lossy candidate: it trades worse direct RMSE for
  quieter board-set behavior, with no negative GGS board observed so far.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 38

### Testing A 14-15 Averaged-Vector Variant

Generated a 54-group `14-15` model using the default averaged vector mode instead of either representative copy:

- `model/eval77_fm_losslessbase54_pair14avg_init/eval_losslessbase54_pair14avg.egev10`
- layout: fixed lossless run `0-5`, single phases `6-13`, averaged group `14-15`, single phases `16-59`
- size: `929,732,000` bytes

Direct check:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair14avg | 0.9674 | 0.0052 | -0.45 | 0.557 | 3.662 | 0 / 30,105 |

This is much worse than both representative-copy variants (`pair14` / `pair14first` are around
`0.0008` sign-disagreement and sub-`0.7` RMSE), so the averaged-vector variant was rejected without battle testing.

### Selector Candidates After Excluding Known Bad Pairs

Re-ran the selector with fixed lossless run `0-5:0`, target `54`, and excluded:

- `12-13`: known sharp representative tradeoff
- `13-14`: GGS seed `20260723` board `44:-12`
- `14-15`: current pair14 family
- `16-17`: XOT seed `20260719` opening `33:-8`
- `18-19`: known bad battle result
- `26-27`: XOT seed `20260719` opening `66:-11`

The next selected candidates were:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `20-21` | 21 |
| MAE | `20-21` | 21 |
| sign | `24-25` | 24 |

Generated:

- `model/eval77_fm_losslessbase54_pair20_init/eval_losslessbase54_pair20.egev10`
- `model/eval77_fm_losslessbase54_pair24first_init/eval_losslessbase54_pair24first.egev10`

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair20 | 0.9847 | 0.0007 | -0.04 | 0.065 | 0.630 | 0 / 30,105 |
| losslessbase54 pair24first | 0.9849 | 0.0006 | +0.02 | 0.067 | 0.659 | 0 / 30,105 |

Targeted XOT sharp suite:

| Candidate | paired matches | win rate | average disc diff | important nonzero openings |
|---|---:|---:|---:|---|
| losslessbase54 pair20 | 10 | 0.6000 | +0.50 | `f5f4g3g6e3f2d3g5:+3`, `f5f4f3g4e3d6c4e6:+2` |
| losslessbase54 pair24first | 10 | 0.5000 | -0.90 | `f5f6c4e3f4c5b5g5:-12`, `f5f6d3c5d6c7d7g5:+3` |

`pair24first` was rejected immediately because it reproduced a sharp `-12` on the same line that exposed `pair26`.

`pair20` also passed the small GGS sharp suite:

| Candidate | paired matches | win rate | average disc diff | nonzero boards |
|---|---:|---:|---:|---:|
| losslessbase54 pair20 | 3 | 0.5000 | +0.00 | 0 |

But broader XOT seed `20260717` rejected it:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | worst observed |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair20 | 20260717 | 100 | 0.4850 | -0.23 | 19 | `9:-8`, `81:-7`, `94:-7` |

Interpretation:

- `pair20` looked safe on known sharp lines but introduced wider XOT noise on unseen openings.
- `pair24first` failed a known sharp line immediately.
- Neither improves on the current `pair14` / `pair14first` tradeoff.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 39

### Testing The Next 22-23 Candidates

After excluding `20-21` and `24-25`, the selector chose `22-23`:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `22-23` | 23 |
| MAE | `22-23` | 23 |
| sign | `22-23` | 22 |

Generated:

- `model/eval77_fm_losslessbase54_pair22_init/eval_losslessbase54_pair22.egev10`
- `model/eval77_fm_losslessbase54_pair22first_init/eval_losslessbase54_pair22first.egev10`

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair22 | 0.9847 | 0.0007 | -0.05 | 0.067 | 0.645 | 0 / 30,105 |
| losslessbase54 pair22first | 0.9846 | 0.0007 | +0.05 | 0.068 | 0.649 | 0 / 30,105 |

Targeted XOT sharp suite:

| Candidate | paired matches | win rate | average disc diff | worst targeted nonzero |
|---|---:|---:|---:|---|
| losslessbase54 pair22 | 10 | 0.4500 | -0.40 | `f5f6d3c5d6c7d7g5:-3` |
| losslessbase54 pair22first | 10 | 0.5500 | +0.50 | `f5f6c4e3f4c5b5g5:-3` |

Broader XOT seed `20260717`:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | worst observed |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair22first | 20260717 | 100 | 0.4650 | -0.25 | 23 | `17:-8`, `29:-8` |
| losslessbase54 pair22 | 20260717 | 100 | 0.5150 | +0.13 | 27 | `28:-8`, `30:-7`, `19:-6` |

Interpretation:

- `pair22first` is rejected: broad XOT seed `20260717` is clearly worse than the current candidates.
- `pair22` has a positive aggregate result on the same seed, but it is too noisy for a mixed-safe replacement:
  worst observed XOT losses reached `-8` and `-7`, worse than the current `pair14first` XOT worst of `-5`.
- The representative choice matters strongly, but neither `22-23` candidate improves the current ranking.

### Next Selector Candidate

After also excluding `22-23`, the selector chose `19-20` representative `19` for all three metrics.

Generated:

- `model/eval77_fm_losslessbase54_pair19first_init/eval_losslessbase54_pair19first.egev10`

Direct and implementation check:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair19first | 0.9848 | 0.0007 | -0.04 | 0.067 | 0.663 | 0 / 30,105 |

Targeted XOT sharp suite rejected it:

| Candidate | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---|
| losslessbase54 pair19first | 10 | 0.4000 | -1.00 | `f5f4c3g6g3c4e3f2:-5`, `f5f4g3e6d6c4e7d7:-5` |

Updated interpretation:

- The new selector-guided candidates (`pair20`, `pair24first`, `pair22`, `pair22first`, `pair19first`) do not
  supersede the existing `pair14` / `pair14first` choice.
- `pair14` remains the best direct-RMSE and XOT-first lossy candidate.
- `pair14first` remains the best measured XOT+GGS mixed-suite lossy candidate so far because the new alternatives
  either fail known sharp lines or produce broader XOT worst losses of `-8` or worse.
- More work can continue by selecting the next excluded-pair candidate after `19-20`, but the evidence so far is
  increasingly favoring `pair14first` as the practical 54-group mixed-suite default and exact `lossless_auto` as the
  conservative no-score-change default.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 40

### Testing The Next Late-Phase Candidate

After excluding `19-20` as well, the selector chose two new candidates:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `54-55` | 54 |
| MAE | `54-55` | 54 |
| sign | `28-29` | 28 |

Generated:

- `model/eval77_fm_losslessbase54_pair54first_init/eval_losslessbase54_pair54first.egev10`
- `model/eval77_fm_losslessbase54_pair28first_init/eval_losslessbase54_pair28first.egev10`

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair54first | 0.9845 | 0.0010 | +0.02 | 0.068 | 0.661 | 0 / 30,105 |
| losslessbase54 pair28first | 0.9849 | 0.0007 | +0.04 | 0.072 | 0.706 | 0 / 30,105 |

Targeted XOT sharp suite:

| Candidate | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---|
| losslessbase54 pair54first | 10 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair28first | 10 | 0.6000 | +1.00 | `f5f6c4e3f4c5b5g5:+7`, `f5f4e3f6g5f2d2f3:+3` |

Broader XOT:

| Candidate | seed | paired matches | win rate | average disc diff | nonzero openings | worst observed |
|---|---:|---:|---:|---:|---:|---|
| losslessbase54 pair54first | 20260717 | 200 | 0.5000 | +0.00 | 0 | none |
| losslessbase54 pair54first | 20260718 | 100 | 0.5000 | +0.00 | 0 | none |
| losslessbase54 pair54first | 20260719 | 100 | 0.5000 | +0.00 | 0 | none |
| losslessbase54 pair28first | 20260717 | 100 | 0.4900 | -0.01 | 26 | `44:-9`, `28:-8`, `37:-5` |

`pair28first` is rejected: aggregate was close to even, but it produced sharper negative XOT openings than the
current mixed-suite candidate.

### GGS Validation For Pair54first

Because `pair54first` was completely quiet on XOT seed `20260717`, extended it to the GGS sharp suite and the same
GGS board seeds used for `pair14` / `pair14first` / `pair13`.

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero boards |
|---|---|---:|---:|---:|---:|
| losslessbase54 pair54first | GGS sharp suite | 3 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair54first | GGS seed 57 | 200 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair54first | GGS seed 20260722 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair54first | GGS seed 20260723 | 100 | 0.5000 | +0.00 | 0 |

Updated comparison of relevant 54-group lossy candidates:

| Candidate | Direct RMSE | XOT summary | GGS summary | worst observed nonzero |
|---|---:|---|---|---|
| losslessbase54 pair14 | 0.588 | strongest direct-RMSE candidate, good XOT | GGS seed 57 has `83:-8`, `195:-5` | `GGS 83:-8` |
| losslessbase54 pair14first | 0.685 | acceptable XOT, but seed 17 has `149:-5` | no negative GGS board across seeds 57/22/23 | `XOT 149:-5`, `XOT 122:-4` |
| losslessbase54 pair54first | 0.661 | no XOT difference observed across seed 17/18/19 samples | no GGS difference observed across seed 57/22/23 samples | none observed |

Interpretation:

- `pair54first` is now the best measured XOT+GGS mixed-suite 54-group lossy candidate, despite not having the best
  direct RMSE or sign-disagreement metric.
- This is a useful reminder that random-position direct metrics are a filter, not the final strength selector:
  `pair54first` has worse direct sign-disagreement than many rejected candidates, but its late-phase merge did not
  affect any measured XOT or GGS game path so far.
- `pair14` remains the best direct-RMSE candidate.
- `pair14first` is superseded by `pair54first` on the measured mixed suite, but `pair54first` should still receive
  additional validation on more late-game-heavy/random board sets before becoming the final lossy default.
- Exact `lossless_auto` remains the conservative no-score-change default.

After excluding `54-55` and `28-29`, the next selector candidates are:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `55-56` | 55 |
| sign | `34-35` | 35 |

These are the next natural candidates if continuing the single-merge sweep.

## Addendum: 2026-07-17 Measurement And Improvement Cycle 41

### Late-Game Validation For Pair54first

`pair54first` merges late phases `54-55`, so added a direct late-game board-problem validation using
`bin/problem/endgame_test_1000.txt`.

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero boards |
|---|---|---:|---:|---:|---:|
| losslessbase54 pair54first | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |

This strengthens the interpretation from Cycle 40: the late-phase merge has not changed any measured game path in
XOT, GGS, or the sampled endgame board set so far.

### Next 54-Group Candidates

Generated the selector next candidates after `54-55` / `28-29`:

- `model/eval77_fm_losslessbase54_pair55first_init/eval_losslessbase54_pair55first.egev10`
- `model/eval77_fm_losslessbase54_pair34last_init/eval_losslessbase54_pair34last.egev10`

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase54 pair55first | 0.9845 | 0.0010 | +0.01 | 0.068 | 0.667 | 0 / 30,105 |
| losslessbase54 pair34last | 0.9846 | 0.0009 | -0.04 | 0.081 | 0.799 | 0 / 30,105 |

Targeted XOT sharp suite:

| Candidate | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---|
| losslessbase54 pair55first | 10 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair34last | 10 | 0.5500 | -0.20 | `f5d6c7g5e6f4g4g3:-8`, `f5f4c3g6g3c4e3f2:-2`, positives elsewhere |

`pair34last` is rejected due the sharp `-8` targeted loss.

`pair55first` stayed quiet in the broader checks:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---:|
| losslessbase54 pair55first | XOT seed 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase54 pair55first | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |

Interpretation:

- `pair55first` is another quiet late-phase 54-group candidate, but it does not improve on `pair54first`:
  direct metrics are slightly worse and validation coverage is smaller.
- `pair54first` remains the best measured 54-group mixed-suite lossy candidate.

### 53-Group Tail Merge Experiment

Because both `54-55` and `55-56` single-pair models were quiet, tested a more aggressive 53-group tail merge:
one group for phases `54-56`.

Generated:

- `model/eval77_fm_losslessbase53_tail54_56_rep54_init/eval_losslessbase53_tail54_56_rep54.egev10`
- `model/eval77_fm_losslessbase53_tail54_56_rep55_init/eval_losslessbase53_tail54_56_rep55.egev10`
- `model/eval77_fm_losslessbase53_tail54_56_rep56_init/eval_losslessbase53_tail54_56_rep56.egev10`

All three are 53 groups, `914,614,416` bytes.

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase53 tail54-56 rep54 | 0.9690 | 0.0020 | +0.06 | 0.137 | 0.949 | 0 / 30,105 |
| losslessbase53 tail54-56 rep55 | 0.9689 | 0.0020 | -0.01 | 0.139 | 0.964 | 0 / 30,105 |
| losslessbase53 tail54-56 rep56 | 0.9688 | 0.0018 | -0.05 | 0.155 | 1.078 | 0 / 30,105 |

Representative `54` has the best direct RMSE, while representative `56` has the best sign rate. Both drew the XOT
sharp suite with zero nonzero openings. Continued with representative `54`.

Tail `54-56` representative `54` results:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---:|
| losslessbase53 tail54-56 rep54 | XOT sharp suite | 10 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep54 | XOT seed 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep54 | XOT seed 20260718 | 100 | 0.4950 | -0.04 | `65:-4` |
| losslessbase53 tail54-56 rep54 | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep54 | GGS sharp suite | 3 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep54 | GGS seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep54 | GGS seed 20260722 | 100 | 0.5000 | +0.00 | 0 |

Interpretation:

- `tail54-56 rep54` is the best new 53-group compressed candidate found in this late-phase sweep.
- It is not as clean as `pair54first`: XOT seed `20260718` exposed one `-4` opening
  (`f5f6c4g5e6c5b6d3`), and direct metrics are substantially worse.
- It is still much quieter than earlier 53-group candidates that inherited `pair26` or `16-17` failures.
- Current ranking:
  - exact/no-score-change: `lossless_auto`
  - best measured 54-group lossy: `pair54first`
  - best measured 53-group lossy so far: `tail54-56 rep54`, pending more validation

## Addendum: 2026-07-17 Measurement And Improvement Cycle 42

### Rechecking Tail 54-56 Representative Choice

Cycle 41 continued with `tail54-56 rep54` because it had the best direct RMSE, but XOT seed `20260718` exposed one
`-4` opening. Rechecked representative `56`, which had worse direct RMSE but the best direct sign-disagreement among
the three representatives.

Additional `tail54-56 rep56` results:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---|
| losslessbase53 tail54-56 rep56 | XOT sharp suite | 10 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep56 | XOT seed 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep56 | XOT seed 20260718 | 100 | 0.5050 | +0.04 | `65:+4` |
| losslessbase53 tail54-56 rep56 | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep56 | GGS seed 57 | 100 | 0.5000 | +0.00 | 0 |

Representative comparison:

| Candidate | Direct sign | Direct RMSE | measured XOT/GGS/endgame worst |
|---|---:|---:|---|
| tail54-56 rep54 | 0.0020 | 0.949 | `XOT 65:-4` |
| tail54-56 rep56 | 0.0018 | 1.078 | no negative observed |

Interpretation:

- `rep56` fixed the only observed negative opening from `rep54`, flipping XOT seed `20260718` opening `65` from
  `-4` to `+4`.
- `rep56` has substantially worse direct RMSE, so this is another case where direct random-position metrics and
  measured game-path stability disagree.
- Current 53-group measured ranking is now:
  - practical/measured stability: `tail54-56 rep56`
  - direct-RMSE preference: `tail54-56 rep54`
- More validation is needed before preferring either 53-group model over the safer 54-group `pair54first`.

### Pair54first Plus Another Pair

Also tested a different 53-group strategy: keep the clean `54-55:54` pair fixed, then add one more non-overlapping
selector-guided pair.

With fixed runs `0-5:0` and `54-55:54`, and excluding known bad pairs, selector results were:

| Metric | selected pair | representative |
|---|---|---:|
| RMSE | `49-50` | 50 |
| MAE | `50-51` | 50 |
| sign | `25-26` | 25 |

Generated the two late/mid-late variants:

- `model/eval77_fm_losslessbase53_pair54_pair49_init/eval_losslessbase53_pair54_pair49.egev10`
- `model/eval77_fm_losslessbase53_pair54_pair50_init/eval_losslessbase53_pair54_pair50.egev10`

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase53 pair54+pair49 | 0.9691 | 0.0019 | +0.05 | 0.139 | 0.965 | 0 / 30,105 |
| losslessbase53 pair54+pair50 | 0.9692 | 0.0020 | +0.06 | 0.140 | 0.969 | 0 / 30,105 |

Targeted XOT sharp suite:

| Candidate | paired matches | win rate | average disc diff | nonzero openings |
|---|---:|---:|---:|---|
| losslessbase53 pair54+pair49 | 10 | 0.5500 | +0.30 | `f5d6c7g5e6f4g4g3:+3` |
| losslessbase53 pair54+pair50 | 10 | 0.5000 | +0.00 | 0 |

Continued with `pair54+pair49` because it had slightly better direct metrics and passed the sharp suite.

Broader checks:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards | worst observed |
|---|---|---:|---:|---:|---:|---|
| losslessbase53 pair54+pair49 | XOT seed 20260718 | 100 | 0.4900 | -0.07 | 14 | `26:-4` |
| losslessbase53 pair54+pair49 | endgame seed 57 | 100 | 0.4750 | -0.08 | 21 | `55:-3` |

Interpretation:

- `pair54+pair49` is rejected as a best-53 candidate: it produced broad endgame noise and a worse aggregate result
  than the tail-only 53-group candidates.
- For 53 groups, the contiguous tail merge `54-56` remains better measured than adding a separate `49-50` or `50-51`
  pair to `pair54first`.
- Updated ranking:
  - exact/no-score-change: `lossless_auto`
  - best measured 54-group lossy: `pair54first`
  - best measured 53-group lossy by practical suite: `tail54-56 rep56`
  - best measured 53-group lossy by direct RMSE among tail models: `tail54-56 rep54`

## Addendum: 2026-07-17 Measurement And Improvement Cycle 43

### Additional Validation For Tail 54-56 Representative 56

Extended the current best measured 53-group candidate, `tail54-56 rep56`, to the same additional seeds used for
`pair54first`:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---:|
| losslessbase53 tail54-56 rep56 | XOT seed 20260719 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase53 tail54-56 rep56 | GGS seed 20260722 | 100 | 0.5000 | +0.00 | 0 |

Updated measured summary for `tail54-56 rep56`:

| Set | measured seeds | paired matches | nonzero openings/boards |
|---|---|---:|---:|
| XOT | 20260717, 20260718, 20260719 | 300 | 1 positive (`65:+4`) |
| GGS | 57, 20260722 | 200 | 0 |
| Endgame | 57 | 100 | 0 |

This strengthens `tail54-56 rep56` as the current best measured 53-group lossy candidate.

### 52-Group Tail 54-57 Experiment

Because `tail54-56 rep56` remained quiet, tested a more aggressive 52-group tail merge: one group for phases
`54-57`.

Generated:

- `model/eval77_fm_losslessbase52_tail54_57_rep56_init/eval_losslessbase52_tail54_57_rep56.egev10`
- `model/eval77_fm_losslessbase52_tail54_57_rep57_init/eval_losslessbase52_tail54_57_rep57.egev10`

Both are 52 groups, `899,496,832` bytes.

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase52 tail54-57 rep56 | 0.9525 | 0.0031 | +0.01 | 0.237 | 1.334 | 0 / 30,105 |
| losslessbase52 tail54-57 rep57 | 0.9522 | 0.0041 | -0.16 | 0.325 | 1.853 | 0 / 30,105 |

Representative `56` is clearly better directly, so continued with it.

Measured results for `tail54-57 rep56`:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---:|
| losslessbase52 tail54-57 rep56 | XOT sharp suite | 10 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | XOT seed 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | XOT seed 20260718 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | XOT seed 20260719 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | GGS seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | GGS seed 20260722 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase52 tail54-57 rep56 | GGS seed 20260723 | 100 | 0.5000 | +0.00 | 0 |

Interpretation:

- `tail54-57 rep56` is now the best measured 52-group lossy candidate.
- It has much worse direct metrics than `tail54-56 rep56`, but no measured game-path difference so far across the
  sampled XOT, GGS, endgame, and sharp-suite checks.
- This makes `tail54-57 rep56` an interesting compression candidate, but it needs broader validation before it can
  supersede the safer 53-group or 54-group candidates.

### 51-Group Tail 54-58 Limit Probe

Tested one more aggressive tail merge, phases `54-58`, as a limit probe.

Generated:

- `model/eval77_fm_losslessbase51_tail54_58_rep56_init/eval_losslessbase51_tail54_58_rep56.egev10`
- `model/eval77_fm_losslessbase51_tail54_58_rep58_init/eval_losslessbase51_tail54_58_rep58.egev10`

Both are 51 groups, `884,379,248` bytes.

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase51 tail54-58 rep56 | 0.9354 | 0.0049 | +0.08 | 0.336 | 1.623 | 0 / 30,105 |
| losslessbase51 tail54-58 rep58 | 0.9347 | 0.0059 | -0.32 | 0.647 | 3.199 | 0 / 30,105 |

`rep58` is rejected directly. `rep56` drew the XOT sharp suite with zero nonzero openings, but its direct metrics are
substantially worse than the 52-group candidate. It remains only a limit-probe candidate for now.

Updated ranking:

- exact/no-score-change: `lossless_auto`
- best measured 54-group lossy: `pair54first`
- best measured 53-group lossy: `tail54-56 rep56`
- best measured 52-group lossy: `tail54-57 rep56`, needs broader validation despite clean measured games so far
- 51-group tail merge: not promoted beyond limit probe because direct metrics degrade sharply

## Addendum: 2026-07-17 Measurement And Improvement Cycle 44

### Extended Validation For 52-Group Tail 54-57 Representative 56

Extended the current best measured 52-group candidate, `tail54-57 rep56`, on the same level-10 paired setup.

Additional results:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards | worst observed |
|---|---|---:|---:|---:|---:|---|
| losslessbase52 tail54-57 rep56 | XOT seed 20260717 | 200 | 0.5025 | +0.01 | 1 positive | `162:+4` |
| losslessbase52 tail54-57 rep56 | GGS seed 57 | 200 | 0.5000 | +0.00 | 0 | none |
| losslessbase52 tail54-57 rep56 | endgame seed 57 | 200 | 0.5000 | +0.00 | 0 | none |

The only new nonzero XOT opening was favorable to the grouped candidate:

- opening `162`, line `f5d6c5f4d7c7e7c6`, disc sum `+4` over the paired battle.

Updated measured summary for `tail54-57 rep56`:

| Set | measured seeds | paired matches | nonzero openings/boards |
|---|---|---:|---:|
| XOT sharp suite | targeted 10 openings | 10 | 0 |
| XOT broad | 20260717, 20260718, 20260719 | 400 | 1 positive (`162:+4`) |
| GGS broad | 57, 20260722, 20260723 | 400 | 0 |
| Endgame | 57 | 200 | 0 |

Interpretation:

- `tail54-57 rep56` is still clean in the sampled GGS and endgame board sets.
- XOT seed `20260717` now exposes one difference after extending to 200 paired matches, but it is favorable.
- The candidate remains an interesting 52-group compression point, but direct metrics are still much worse than the
  safer 53-group and 54-group candidates.

### Broader Probe For 51-Group Tail 54-58 Representative 56

Since the 52-group candidate stayed quiet, gave the 51-group limit probe a small broader check.

Additional measured results:

| Candidate | set / seed | paired matches | win rate | average disc diff | nonzero openings/boards |
|---|---|---:|---:|---:|---:|
| losslessbase51 tail54-58 rep56 | XOT seed 20260717 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase51 tail54-58 rep56 | GGS seed 57 | 100 | 0.5000 | +0.00 | 0 |
| losslessbase51 tail54-58 rep56 | endgame seed 57 | 100 | 0.5000 | +0.00 | 0 |

This candidate is now quiet on the targeted XOT sharp suite plus initial broad XOT, GGS, and endgame samples.
It is still not promoted beyond limit-probe status because its direct metrics are substantially worse than the
52-group candidate (`sign_disagree=0.0049`, `RMSE=1.623`).

### 50-Group Tail 54-59 Limit Probe

Generated two more aggressive 50-group variants with phases `54-59` merged into one FM group:

- `model/eval77_fm_losslessbase50_tail54_59_rep56_init/eval_losslessbase50_tail54_59_rep56.egev10`
- `model/eval77_fm_losslessbase50_tail54_59_rep59_init/eval_losslessbase50_tail54_59_rep59.egev10`

Both are 50 groups, `869,261,664` bytes.

Direct and implementation checks:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase50 tail54-59 rep56 | 0.9191 | 0.0094 | +0.31 | 0.565 | 2.707 | 0 / 30,105 |
| losslessbase50 tail54-59 rep59 | 0.9164 | 0.0068 | -0.23 | 0.958 | 4.095 | 0 / 30,105 |

`rep56` also drew the XOT sharp suite with zero nonzero openings, but the direct-error degradation is too large.
The 50-group tail merge is not worth broader battle validation at this point.

Updated ranking:

- exact/no-score-change: `lossless_auto`
- best measured 54-group lossy: `pair54first`
- best measured 53-group lossy: `tail54-56 rep56`
- best measured 52-group lossy: `tail54-57 rep56`, now broader-validated but still direct-metric risky
- 51-group tail merge: quiet in initial broad checks, still limit-probe only because direct metrics degrade sharply
- 50-group tail merge: rejected for now due severe direct-error degradation

## Addendum: 2026-07-17 Measurement And Improvement Cycle 45

### Representative Sweep For 51-Group And 50-Group Tail Merges

Cycle 44 only checked a subset of representatives for the aggressive tail merges. Completed the representative sweep
to make sure the limit-probe conclusions were not caused by a poor representative choice.

51-group `54-58` representative comparison:

| Candidate | exact | sign disagree | bias | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| losslessbase51 tail54-58 rep54 | 0.9350 | 0.0070 | +0.26 | 0.393 | 1.956 |
| losslessbase51 tail54-58 rep55 | 0.9354 | 0.0056 | +0.13 | 0.341 | 1.692 |
| losslessbase51 tail54-58 rep56 | 0.9354 | 0.0049 | +0.08 | 0.336 | 1.623 |
| losslessbase51 tail54-58 rep57 | 0.9355 | 0.0052 | -0.15 | 0.395 | 1.971 |
| losslessbase51 tail54-58 rep58 | 0.9347 | 0.0059 | -0.32 | 0.647 | 3.199 |

`rep56` remains the best 51-group representative by sign-disagreement, MAE, and RMSE.

50-group `54-59` representative comparison:

| Candidate | exact | sign disagree | bias | MAE | RMSE | materialized mismatches |
|---|---:|---:|---:|---:|---:|---:|
| losslessbase50 tail54-59 rep54 | 0.9187 | 0.0120 | +0.51 | 0.655 | 3.138 | not checked |
| losslessbase50 tail54-59 rep55 | 0.9191 | 0.0101 | +0.35 | 0.571 | 2.791 | not checked |
| losslessbase50 tail54-59 rep56 | 0.9191 | 0.0094 | +0.31 | 0.565 | 2.707 | 0 / 30,105 |
| losslessbase50 tail54-59 rep57 | 0.9199 | 0.0079 | -0.02 | 0.542 | 2.505 | 0 / 30,105 |
| losslessbase50 tail54-59 rep58 | 0.9183 | 0.0096 | -0.12 | 0.850 | 3.728 | not checked |
| losslessbase50 tail54-59 rep59 | 0.9164 | 0.0068 | -0.23 | 0.958 | 4.095 | 0 / 30,105 |

`rep57` is the best 50-group representative by exact rate, bias, MAE, and RMSE, while `rep59` has the best sign
rate but very poor magnitude error. `rep57` also drew the XOT sharp suite with zero nonzero openings.

Updated interpretation:

- The 51-group limit-probe conclusion is unchanged: `tail54-58 rep56` is the best representative, but direct metrics
  remain much worse than the 52-group candidate.
- The best 50-group representative is now `tail54-59 rep57`, not `rep56`.
- Even the best 50-group representative has severe direct-error degradation (`sign_disagree=0.0079`, `RMSE=2.505`),
  so the 50-group tail merge is still not worth broader battle validation.

## Addendum: 2026-07-17 Current Authoritative Result

This addendum supersedes earlier speed conclusions in this report.

### Current Evaluation Model

The current model is:

`model/eval77_fm_losslessbase54_pair54first_init/eval_losslessbase54_pair54first.egev10`

Its linear weights remain phase-specific. Phases 0 through 5 share one FM vector table, phases 54 and 55 share one
FM vector table, and every other phase retains its original FM vector table. The file therefore contains 54 distinct
FM vector tables and is 929,732,000 bytes.

At startup, the loader expands this compact file into 18-byte records containing one 2-byte linear weight followed
by one 16-byte FM vector. The feature generator converts eight feature identifiers at a time into byte offsets for
these records with AVX2 instructions. The evaluation loop then uses one phase base address and the precomputed byte
offsets, instead of calculating separate linear-weight and FM-vector addresses for each feature.

The current executable is:

`bin/Egaroucid_for_Console_fm_mat_direct_offset_linear_mo_simd_nws40_phase26.exe`

### Evaluation Correctness

The current implementation was checked on 300,754 positions from 5,000 randomly generated games with seed 20260750.
The incrementally maintained evaluation state agreed with evaluation from freshly generated features on every
position. The AVX2 implementation also agreed with the scalar implementation on every position. Both mismatch
counts were zero.

### Match Strength

One paired match consists of two games from the same opening with colors exchanged. A pair is a win, draw, or loss
according to the sum of the candidate's final disc differences in the two games. Each result below contains 100
paired matches.

Against the current `bin/resources/eval.egev4`, with seed 20260743:

| Level | Wins | Draws | Losses | `(wins + 0.5 * draws) / 100` | Mean paired disc difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 11 | 47 | 0.475 | -0.64 |
| 5 | 14 | 69 | 17 | 0.485 | -0.13 |
| 10 | 2 | 97 | 1 | 0.505 | +0.13 |

Against the packaged Egaroucid 7.8.1 executable, with seed 20260744:

| Level | Wins | Draws | Losses | `(wins + 0.5 * draws) / 100` | Mean paired disc difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 0 | 41 | 0.590 | +6.18 |
| 5 | 65 | 3 | 32 | 0.665 | +4.17 |
| 10 | 68 | 5 | 27 | 0.705 | +2.13 |

These are equal configured-level matches, not equal-time matches. The byte-offset optimization changes neither
evaluation values nor move ordering, so the results remain applicable to the current executable.

### Speed Measurements

The midgame suite used `bin/problem/midgame_test.txt`, level 23, 28 threads, and hash size 25. The values below are
independent medians from three runs.

| Executable | Nodes | Elapsed time | Nodes per second |
|---|---:|---:|---:|
| Current candidate | 1,277,141,441 | 20.839 s | 61,513,972 |
| Candidate before byte-offset precomputation | 1,358,263,831 | 29.254 s | 46,430,020 |
| Egaroucid 7.8.1 | 889,072,489 | 8.879 s | 99,776,000 |

The current candidate's median elapsed time is 28.77% shorter than the earlier candidate's time, but it remains
2.35 times the Egaroucid 7.8.1 time on the midgame suite.

The endgame suite used `bin/problem/ffo40-59.txt`, level 60, 28 threads, and hash size 25. The values below are
independent medians from three runs.

| Executable | Correct answers | Nodes | Elapsed time | Nodes per second |
|---|---:|---:|---:|---:|
| Current candidate | 20 / 20 | 13,224,145,849 | 26.723 s | 490,629,367 |
| Candidate before byte-offset precomputation | 20 / 20 | 13,320,088,689 | 29.744 s | 446,637,199 |
| Egaroucid 7.8.1 | 20 / 20 | 15,010,308,023 | 26.772 s | 560,672,000 |

The current candidate's median elapsed time is 10.16% shorter than the earlier candidate's time and 0.18% shorter
than the Egaroucid 7.8.1 time. All 20 answers were correct in every run.

For a controlled single-thread level-15 comparison, each executable was run five times in alternating order:

| Executable | Nodes | Median elapsed time | Median nodes per second |
|---|---:|---:|---:|
| Current candidate | 16,364,952 | 1.693 s | 9,666,244 |
| Egaroucid 7.8.1 | 17,695,940 | 0.907 s | 19,510,407 |

The candidate searches 7.52% fewer nodes, but processes 49.54% as many nodes per second and takes 1.87 times as long.
Further midgame improvement therefore requires both a lower evaluation cost per search node and fewer search nodes
at level 23.

### Rejected Exact Optimizations

The following changes preserved evaluation values but did not improve speed:

- Storing a precomputed FM-vector squared norm reduced isolated evaluation throughput by 6.51%.
- Splitting the FM-vector sum into two or four accumulators produced no measurable improvement.
- Using AVX-VNNI integer vector-dot-product instructions increased the median level-15 elapsed time by 0.40%.
- Compiling with `-O3` instead of `-O2` increased the median level-15 elapsed time by 0.37%.
- Profile-guided optimization increased the median level-15 elapsed time by 1.12%.
- Exact 4-bit FM-vector storage with an exception table increased the median level-15 elapsed time by 26.87% and
  increased one level-23 midgame-suite elapsed time by 17.31%.

### Further Rejected Exact Optimizations

The following tests also preserved evaluation values. A change that improved a short single-thread measurement was
accepted only if the 28-thread midgame or endgame suite also supported the improvement.

- Disabling prefetching in every phase increased the median midgame-suite node rate by 1.48%, from 61,513,972 to
  62,425,953 nodes per second. It reduced the median endgame-suite node rate by 0.93%, from 490,629,367 to
  486,060,479 nodes per second, and increased the median endgame elapsed time from 26.723 to 27.268 seconds.
- Splitting each 18-byte record into a 16-byte FM-vector array and a 2-byte linear-weight array increased the median
  single-thread level-15 elapsed time by 8.33%, from 1.656 to 1.794 seconds.
- Storing the squared norm in the exact 4-bit representation reproduced the original values on 300,967 positions,
  but increased the median single-thread level-15 elapsed time by 28.85%, from 1.733 to 2.233 seconds.
- Moving the phases without FM interactions to a separate function reduced the median single-thread level-15 time
  by only 0.12%. One midgame-suite run and one endgame-suite run showed no reproducible improvement.
- Processing the 64 pattern features separately from the one disc-count feature increased the median single-thread
  level-15 time by 0.99%, from 1.723 to 1.740 seconds.
- Pre-scaling each pattern type's start position into an 18-byte record offset reduced the median single-thread
  level-15 time by 0.86%, from 1.735 to 1.720 seconds. In two alternating endgame-suite runs, mean throughput fell
  by 2.15%, from 488,597,070 to 478,096,302 nodes per second, and mean elapsed time increased from 26.843 to
  28.148 seconds.
After removing all of these experiments and restoring the accepted implementation, a new correctness check used
5,000 random games with seed 20260756. Evaluation from freshly generated features, evaluation from the incremental
search state, and scalar evaluation agreed on all 300,924 checked positions.

### Remaining Blocker And Next Work

The specified training data directories are absent from the current machine, and the machine has no `D:` drive.
The required phase-sharing retraining cannot begin until the current location of the 234.66 GB data set is known.

The endgame speed target is met, but the overall goal is not met because the midgame suite still takes 2.35 times as
long as Egaroucid 7.8.1. The next implementation work is to reduce the per-node cost of the 16-element FM evaluation
and reduce the level-23 search-node count. After either behavior changes, levels 1, 5, and 10, the midgame suite, and
the endgame suite must all be measured again.

## Addendum: 2026-07-18 Move-Ordering Evaluation Experiment

### Removing One Pattern Type From The Final Evaluation

Each of the 16 board-pattern types was removed individually from the FM interaction term while retaining its linear
term. Every candidate played 100 paired matches against the current candidate at level 5. A paired match consists of
two games from the same opening with colors exchanged.

The best initial result removed source pattern type 4. It scored 48 wins, 1 draw, and 51 losses. Its
`(wins + 0.5 * draws) / pairs` value was 0.485, and its mean candidate disc difference per game was -0.59.

A second level-5 test used 1,000 paired matches and different openings. The candidate scored 400 wins, 35 draws,
and 565 losses. Its win-rate-equivalent value was 0.4175, and its mean candidate disc difference per game was -2.02.
The initial result did not reproduce, so removing one pattern type from the final evaluation was rejected.

### Partial FM Evaluation Only Inside Move-Ordering Searches

The final move-selection search continues to use all 16 FM pattern types. Only the shallow searches used internally
to order legal moves use a partial FM interaction term. Linear terms remain active for every pattern type.

A validation build that enabled all 16 FM pattern types searched 16,364,952 nodes on the level-15 midgame suite,
exactly matching the current candidate.

Screening every single-type removal found that omitting source pattern type 11 was best. Pattern type 11 consists of
the ten squares A1, B2, C3, D4, B1, C2, D3, A2, B3, and C4, plus its 90-degree, 180-degree, and 270-degree rotations.

The partial move-ordering evaluation is used only when the requested result is a midgame result. Searches whose
requested result is an exact endgame result use all 16 FM pattern types, including in their internal move-ordering
searches.

The executable is:

`bin/Egaroucid_for_Console_fm_mat_direct_offset_recursive_mo_maskf7ff_midgame_only_simd_nws40_phase26.exe`

Additional build settings are:

- `EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING_SEARCH`
- `EVALUATE_EXPERIMENT_7_7_FM_SUBSET_MIDGAME_SEARCH_ONLY`
- `EVAL77_FM_MOVE_ORDERING_PATTERN_MASK=0xF7FF`

`0xF7FF` is a 16-bit mask with only bit 11 cleared. Each bit controls whether the FM interaction term for the
corresponding source pattern type is used.

### Direct Match Results Against The Current Candidate

| Level | Pairs | Wins | Draws | Losses | `(wins + 0.5 * draws) / pairs` | Mean candidate disc difference per game |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0 | 100 | 0 | 0.500 | 0.00 |
| 5 | 100 | 3 | 94 | 3 | 0.500 | +0.03 |
| 10 | 100 | 1 | 96 | 3 | 0.490 | -0.03 |
| 5 | 1,000 | 15 | 961 | 24 | 0.4955 | -0.05 |

The 1,000-pair level-5 test used openings different from the 100-pair test. The match evidence shows no material
strength difference between the new candidate and the current candidate.

### Single-Thread Level-15 Measurement

The new candidate and current candidate were measured five times each in alternating order.

| Executable | Nodes | Median elapsed time | Median nodes per second |
|---|---:|---:|---:|
| New candidate | 14,464,998 | 1.529 s | 9,460,430 |
| Current candidate | 16,364,952 | 1.701 s | 9,620,783 |

The new candidate searched 11.61% fewer nodes and took 10.11% less time.

### Midgame Suite

The 32 positions in `bin/problem/midgame_test.txt` were measured three times at level 23 with 28 threads, hash
level 25, and no opening book. Every table entry is the independent median of the three runs.

| Executable | Nodes | Elapsed time | Nodes per second |
|---|---:|---:|---:|
| New candidate | 1,276,885,966 | 20.228 s | 63,124,676 |
| Current candidate, remeasured immediately afterward | 1,231,078,357 | 20.667 s | 59,885,627 |

The new candidate took 2.12% less time than the immediately remeasured current candidate. The improvement is small,
and the 28-thread node counts varied substantially between runs.

### Endgame Suite

The 20 positions in `bin/problem/ffo40-59.txt` were measured three times at level 60 with 28 threads, hash level 25,
and no opening book. The new candidate returned the known correct value for all 20 positions.

| Executable | Correct answers | Nodes | Elapsed time | Nodes per second |
|---|---:|---:|---:|---:|
| New candidate | 20 / 20 | 13,286,161,430 | 27.495 s | 488,317,286 |
| Current candidate, remeasured immediately afterward | 20 / 20 | 13,249,364,887 | 27.063 s | 489,574,876 |

The new candidate took 1.60% more time. It did not improve endgame speed.

### Correctness

A consistency check generated 5,000 random games with seed 20260790 and checked 300,874 positions. Scores computed
from freshly generated features, incrementally maintained search state, and the scalar implementation agreed on
every position.

A regression-check executable was also built with only the current candidate's original build settings. It and the
existing current executable both searched exactly 16,364,952 nodes on the single-thread level-15 midgame suite.

### Current Decision

The new candidate preserves direct-match strength and improves the single-thread level-15 test and the 28-thread
midgame suite. It increases the endgame-suite elapsed time by 1.60%. Its 20.228-second midgame-suite result remains
2.28 times the Egaroucid 7.8.1 result of 8.879 seconds.

The candidate remains a useful midgame-speed comparison, but the overall objective of retaining speed close to
Egaroucid 7.7 and 7.8 is not achieved. Until the specified training data is located, further work should avoid
weakening the final evaluation and should continue to target both evaluation cost per call and searched-node count.

## Addendum: 2026-07-18 Phase Sharing And Phase Replacement

### FM-Only Phase Sharing

FM means Factorization Machines in this section. The 50-group model
`model/eval77_fm_losslessbase50_lowcost_init/eval_losslessbase50_lowcost.egev10`
keeps phase-specific linear parameters and shares only FM parameters.

On the 32-position midgame suite at level 23, 28 threads, hash level 25, and no book, the materialized layout
searched 1,461,744,548 nodes in 22.480 seconds. The split layout, which reads linear and FM parameters from
separate memory regions, searched 3,139,379,188 nodes in 56.975 seconds. Neither layout improved elapsed time.

The 52-group model
`model/eval77_fm_losslessbase52_tail54_57_rep56_init/eval_losslessbase52_tail54_57_rep56.egev10`
uses phase 56 FM parameters in phases 54 through 57. One level-60 run on the 20-position endgame suite returned
every known final disc difference correctly and searched 13,388,582,641 nodes in 26.959 seconds. This did not
establish an improvement over the accepted model's three-run medians of 13,249,364,887 nodes and 27.063 seconds.

### Sharing Linear And FM Parameters Together

The model converter can now copy a representative phase's linear parameters, linear quantization scale, and FM
parameters together. A new loader verifies that every phase in a group has identical linear parameters and an
identical linear quantization scale. It rejects incompatible models. Compatible models are materialized into one
contiguous parameter table per group.

A 55-group model that shared five adjacent phase pairs scored 30 wins, 35 draws, and 35 losses in 100 paired
level-5 matches. Counting a draw as half a win gives 47.5 percent, and the mean candidate final disc difference was
-0.55 per game. Its midgame-suite result was 1,562,889,664 nodes in 21.780 seconds, so it was rejected.

The one-pair screening results were:

| Shared phases | Representative phase | Nodes | Elapsed time |
|---:|---:|---:|---:|
| 12 and 13 | 13 | 1,189,990,033 | 19.635 s |
| 14 and 15 | 15 | 1,306,329,320 | 20.634 s |
| 18 and 19 | 19 | 1,243,657,590 | 20.412 s |
| 20 and 21 | 21 | 1,466,455,502 | 21.790 s |
| 26 and 27 | 26 | 1,270,028,052 | 19.308 s |

The phase-13 representative model and executable were:

- `model/eval77_fm_wholephase59_pair12_rep13_init/eval_wholephase59_pair12_rep13.egev10`
- `bin/Egaroucid_for_Console_fm_wholephase_mat_recursive_mo_maskf7ff_midgame_only_simd_nws40_phase26.exe`

Direct results against the accepted evaluator were:

| Level | Pairs | Wins | Draws | Losses | Score rate | Mean final disc difference per game |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 16 | 65 | 19 | 48.5% | -0.63 |
| 5 | 1,000 | 40 | 929 | 31 | 50.45% | +0.138 |
| 10 | 100 | 4 | 95 | 1 | 51.5% | +0.07 |

In alternating three-run midgame measurements, the 59-group model had medians of 1,394,855,267 nodes,
21.301 seconds, and 65,446,264 nodes per second. The 54-group model with source pattern type 11 omitted only during
shallow move-ordering searches had medians of 1,391,976,800 nodes, 21.738 seconds, and 64,034,262 nodes per second.

In alternating three-run endgame measurements, all 20 answers were correct. The 59-group model had medians of
13,351,243,326 nodes, 27.076 seconds, and 488,034,320 nodes per second. The comparison had medians of
13,123,966,012 nodes, 26.702 seconds, and 489,791,130 nodes per second.

The phase-13 representative model retained match strength but was 1.40 percent slower in the endgame suite.
The original linear move-ordering implementation also had a fresh midgame-suite median of 20.417 seconds, which is
shorter than 21.301 seconds. The 59-group model was therefore rejected as a final candidate.

An alternative model averaged all linear and FM parameters of phases 12 and 13. In 1,000 paired level-5 matches, it
scored 184 wins, 625 draws, and 191 losses, for 49.65 percent and a mean final disc difference of -0.263 per game.
At single-thread level 15 it searched 15,422,117 nodes with a median of 1.609 seconds. The comparison searched
14,464,998 nodes with a median of 1.511 seconds. The averaged model was rejected because it searched 6.62 percent
more nodes and took 6.49 percent longer.

### Replacing Phases Only During Move Ordering

The final evaluation was left unchanged. Only the shallow searches used to order legal moves were allowed to read
an adjacent phase's parameters. Every adjacent replacement from phase 6 through phase 30 was screened at
single-thread level 15.

For the move-ordering evaluator that omits source pattern type 11, the best combined setting used phase 15
parameters in phase 14, phase 24 parameters in phase 25, and phase 25 parameters in phase 26. It searched
14,249,011 nodes with a median of 1.500 seconds. The same evaluator without phase replacements searched
14,464,998 nodes with a median of 1.531 seconds.

On the midgame suite, the three-replacement setting had medians of 1,434,365,887 nodes and 21.387 seconds. The same
evaluator without replacements had medians of 1,400,729,402 nodes and 22.022 seconds. However, the original linear
move-ordering implementation had a fresh median of 20.417 seconds, which was 4.75 percent shorter.

Direct results for the three-replacement setting against the accepted evaluator were:

| Level | Pairs | Wins | Draws | Losses | Score rate | Mean final disc difference per game |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0 | 100 | 0 | 50.0% | 0.00 |
| 5 | 1,000 | 24 | 945 | 31 | 49.65% | -0.067 |
| 10 | 100 | 5 | 93 | 2 | 51.5% | +0.12 |

All three endgame runs returned all 20 known final disc differences correctly. The medians were 13,255,288,011
nodes, 26.712 seconds, and 500,361,665 nodes per second. Strength and endgame speed were retained, but midgame speed
was worse than the original linear move-ordering implementation, so the setting was rejected.

Adjacent phase replacements were also screened in the lightweight linear move-ordering evaluator. The best
single replacement used phase 16 linear parameters in phase 15. A three-replacement setting used phase 9 in
phase 10, phase 16 in phase 15, and phase 19 in phase 18. At single-thread level 15, it searched 16,176,509 nodes
with a median of 1.665 seconds. The evaluator without replacements searched 16,364,952 nodes with a median of
1.700 seconds.

On the midgame suite, the three-replacement setting had medians of 1,368,821,035 nodes and 21.003 seconds, while the
evaluator without replacements had medians of 1,359,508,012 nodes and 20.669 seconds. The single phase-15
replacement had medians of 1,297,168,126 nodes and 20.501 seconds. A fresh comparison without replacements had
medians of 1,244,027,668 nodes and 20.417 seconds. Both replacement settings were rejected.

### Retraining A 14-Element FM Model

The available local distillation data contains about 12.63 million training records and 4.21 million holdout
records across phases 6 through 59. A 14-element FM model was trained in three stages covering phases 6 through 20,
21 through 40, and 41 through 59:

`model/eval77_fm_dim14_distilled312064/eval77_fm_dim14_distilled312064.egev4`

The evaluation file is 907,055,554 bytes. On the single-thread level-15 midgame suite, it searched 31,025,528 nodes
with medians of 5.484 seconds and 5,657,463 nodes per second. The accepted 16-element model searches 16,364,952
nodes in about 1.7 seconds.

In 100 paired level-1 matches, the 14-element model scored 30 wins, 4 draws, and 66 losses. Its score rate was
32.0 percent and its mean final disc difference was -8.40 per game. It failed both the strength and speed
requirements and was rejected.

### Result After These Experiments

These experiments produced phase-sharing and move-ordering phase-replacement models that retained direct match
strength. None consistently improved the original implementation's 20.417-second midgame-suite median.
Egaroucid 7.8.1 completes the same suite in 8.879 seconds, so the overall target remains unmet.

A substantial improvement now requires either fewer parameter references while retaining the 16-element FM model,
or a lower-dimensional model trained to retain direct match strength. The approximately 234.66 GB training data
specified by the handoff document is absent from this workspace. Its current location is required before the
planned large-data retraining can proceed.
