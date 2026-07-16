# Training Results

Related issue: #613

## English

### Terminology

In this experiment, `games` means complete game transcripts. The generated
training positions are called `position_samples`. Existing directory names such
as `records0` and `records1` are kept only as dataset paths.

### Dataset

- Source: `train_data/transcript_release/0002`
- Selection: 1,000,000 games, seed `613`
- Converted games: 1,000,000 valid, 0 invalid
- Generated `position_samples`: 26,384,206
- Conversion resource: 179.371 sec, peak RSS 14.238 MiB

The number of `position_samples` is larger than the number of games because one
game contributes many move positions after the random opening segment.

### Training Search

All runs used 5,000,000 train `position_samples`, 500,000 validation
`position_samples`, batch size 8192, 20 epochs, patience 5, TensorFlow/Keras.

| Config | Params | Val top-1 | Val top-3 | Val top-5 | Time sec | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 0.297304 | 0.584340 | 0.742064 | 33.480 | 7751.410 |
| 96x3 | 37,216 | 0.308512 | 0.604394 | 0.763330 | 33.417 | 7787.094 |
| 128x3 | 57,792 | 0.326106 | 0.626642 | 0.783058 | 33.446 | 7782.047 |
| 96x4 | 46,528 | 0.313866 | 0.610754 | 0.767646 | 34.501 | 7784.199 |
| 160x3 | 82,464 | 0.341052 | 0.649038 | 0.804222 | 34.443 | 7794.395 |
| 192x3 | 111,232 | 0.353422 | 0.664336 | 0.816872 | 35.499 | 7738.395 |
| 128x4 | 74,304 | 0.329930 | 0.632604 | 0.787496 | 35.519 | 7652.391 |

Best validation config: `192x3`.

### Correct WTHOR Agreement

The WTHOR evaluator now uses a policy-output-order legal mask. Earlier
intermediate outputs used the feature-order bit mask and should be ignored.

| Config | Positions | Top-1 exact | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192x3 | 8,035,282 | 0.338097 | 0.360323 | 0.707869 | 0.870466 |
| 160x3 | 8,035,282 | 0.328847 | 0.351074 | 0.683399 | 0.852199 |
| 128x4 | 8,035,282 | 0.313147 | 0.335372 | 0.661174 | 0.841202 |
| 128x3 | 8,035,282 | 0.307269 | 0.329494 | 0.663007 | 0.839235 |
| 96x3 | 8,035,282 | 0.298108 | 0.314827 | 0.648127 | 0.824891 |
| 96x4 | 8,035,282 | 0.292177 | 0.308892 | 0.652110 | 0.829590 |
| 64x3 | 8,035,282 | 0.289575 | 0.306292 | 0.631556 | 0.810074 |

Best WTHOR config: `192x3`.

`192x3` by 10-move bucket:

| Moves | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: |
| 01-10 | 0.377334 | 0.812311 | 0.964305 |
| 11-20 | 0.292862 | 0.612274 | 0.799897 |
| 21-30 | 0.262999 | 0.577686 | 0.775109 |
| 31-40 | 0.280449 | 0.614852 | 0.811226 |
| 41-50 | 0.354634 | 0.715394 | 0.887165 |
| 51-60 | 0.595946 | 0.916727 | 0.986230 |

### Blend Benchmark

`hint 100` at Egaroucid Console 7.8.1 level 21 is expensive per position. A
30-position random WTHOR sample took 86.157 sec and peak RSS 1283.957 MiB.
That extrapolates to roughly 267 days for all 8,035,282 WTHOR positions on one
process before caching, so full WTHOR blend evaluation is not practical in this
session as a direct run.

The WTHOR blend evaluator supports `--jobs`, `--range-start` / `--range-end`,
and `--hint-cache-db`. The sharded runner
`20_test_with_wthor/run_wthor_blend_shards.py` wraps those pieces for resumable
full runs, skips already-finished shards, and creates a shared SQLite
hint-score cache by default.

Small random sample, seed `613`, 30 positions:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

This sample is only for timing and a rough blend sanity check.

### WTHOR Hint-Cache Planning

`20_test_with_wthor/analyze_wthor_position_duplicates.py` counts exact
Egaroucid hint inputs `(black, white, side)`.

| Metric | Value |
| --- | ---: |
| WTHOR `position_samples` | 8,035,282 |
| Unique hint positions | 5,574,955 |
| Duplicate hint positions | 2,460,327 |
| Duplicate fraction | 0.306190 |
| Invalid side samples | 0 |
| Analysis elapsed | 6.745 sec |
| Analysis in-process peak RSS | 440.371 MiB |
| Resource-monitor elapsed | 7.104 sec |
| Resource-monitor peak RSS | 293.832 MiB |

The cache can remove about 30.6% of repeated hint calls within WTHOR and also
protects interrupted/resumed shard runs from recomputing finished hint scores.
However, the unique hint-position count is still 5.57 million, so a full blend
evaluation remains very large unless the Egaroucid hint cost is reduced or the
evaluation set is sampled.

Cache smoke test on the first two WTHOR position samples:

| Run | Elapsed sec | Peak RSS MiB | Lookups | Hits | Misses | Writes | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First run | 2.032 | 1325.613 | 2 | 0 | 2 | 2 | 2 |
| Second run, same DB | 1.030 | 8.523 | 2 | 2 | 0 | 0 | 2 |

A two-shard smoke run also passed and the merged output preserved cache stats:
2 lookups, 0 hits, 2 misses, 2 writes, 2 rows.

### Resumable Full-Run Controls

The WTHOR sharded runner now supports `--positions-per-shard`,
`--range-start` / `--range-end`, and `--time-limit-sec`. This lets the full
WTHOR blend run advance in smaller, resumable chunks instead of requiring one
very large shard to finish. `--merge-completed` updates `partial_merged` from
all finished shards even before the full WTHOR run is complete.

For very small shard sizes such as `--positions-per-shard 5`, the runner no
longer expands every scheduled shard into `manifest.json`. It writes the total
`shard_count` plus a first/last preview controlled by
`--manifest-shard-preview`, so continuing from `--range-start 30` avoids a
large manifest.

Smoke test with 5 WTHOR position samples, `positions_per_shard=2`,
`blend_param=1.0`, `top_n=1,3`:

- First invocation used `--time-limit-sec 0.001`: completed shard `0..2`, then
  stopped with `stop_reason=time_limit`.
- Second invocation reused the same output directory, skipped shard 0, ran
  shard `2..4`, then stopped with `stop_reason=max_shards_to_run`.
- Third invocation skipped shards 0 and 1, ran shard `4..5`, and merged all
  outputs with `stop_reason=finished`.

Merged smoke result:

| Blend param | Top-N | Exact accuracy | Symmetric accuracy | Positions |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1 | 0.200000 | 0.400000 | 5 |
| 1.0 | 3 | 0.800000 | 0.800000 | 5 |

Real full-run progress in `20_test_with_wthor/output/blend_wthor_full_chunked`:

- Chunk 001 completed 20 / 8,035,282 WTHOR position samples.
- Chunk 002 reused the same output directory, skipped the first two finished
  shards, completed one more shard, and updated `partial_merged`.
- Chunk 003 continued from `--range-start 30` with `--positions-per-shard 5`,
  completed shard `30..35`, and updated `partial_merged`.
- After the manifest/schedule fix, chunk 004 continued from `--range-start 35`
  and completed six 5-position-sample shards (`35..65`).
- Chunk 005 continued from `--range-start 65` and completed two more
  5-position-sample shards (`65..75`).
- Chunk 006 tested `--positions-per-shard 25`, completed shard `75..100`,
  and showed that larger shards make time-limit control coarser.
- Chunk 007 used `--resume-from-completed-prefix`, automatically started from
  `100`, and completed seven 5-position-sample shards (`100..135`).
- Chunk 008 used the same auto-resume p5 setup with a 60-second limit and
  completed two 5-position-sample shards (`135..145`).
- Chunk 009 tested `--jobs-per-shard 2` with `--positions-per-shard 20`,
  completed shard `145..165`, and used 2634.234 MiB peak RSS.
- Chunk 010 tested `--jobs-per-shard 4` with `--positions-per-shard 20`,
  completed three shards (`165..225`) with 5205.410 MiB peak RSS.
- Chunk 011 kept `jobs_per_shard=4`, used a 120-second limit, and completed
  six shards (`225..345`) with 5203.469 MiB peak RSS.
- Chunk 012 kept `jobs_per_shard=4`, used a 180-second limit, and completed
  nine shards (`345..525`) with 5207.145 MiB peak RSS.
- Chunk 013 tested `egaroucid_threads=2`, completed four shards (`525..605`)
  with 5204.996 MiB peak RSS.
- Chunk 014 tested `egaroucid_threads=4`, completed five shards (`605..705`)
  with 5208.156 MiB peak RSS.
- Chunk 015 tested `egaroucid_threads=8`, completed nine shards (`705..885`)
  with 5212.191 MiB peak RSS.
- Chunk 016 tested `egaroucid_threads=16`, completed nine shards (`885..1065`)
  with 5221.848 MiB peak RSS.
- Chunk 017 used the current best short-benchmark setting
  `jobs_per_shard=4, egaroucid_threads=8`, used a 300-second limit, and
  completed 39 shards (`1065..1845`) with 5226.602 MiB peak RSS.
- Chunk 018 used the same setting and a 300-second limit, and completed
  35 shards (`1845..2545`) with 5213.312 MiB peak RSS.
- Chunk 019 used the same setting and a 300-second limit, and completed
  33 shards (`2545..3205`). This was a direct run; `progress_summary.json`
  reports 303.996 sec elapsed, but peak RSS was not sampled.
- Chunk 020 used the same setting with the resource monitor, and completed
  36 shards (`3205..3925`) with 5272.352 MiB peak RSS.
- Chunk 021 used the same setting with the resource monitor, and completed
  36 shards (`3925..4645`) with 5216.648 MiB peak RSS.
- Chunk 022 tested `jobs_per_shard=8, egaroucid_threads=4`, completed
  12 shards (`4645..4885`) with 10349.910 MiB peak RSS.
- Chunk 023 returned to `jobs_per_shard=4, egaroucid_threads=8`, and completed
  33 shards (`4885..5545`) with 5265.848 MiB peak RSS.
- Chunk 024 completed 40 shards (`5545..6345`) with 5322.113 MiB peak RSS.
  The evaluation shards succeeded, but the old merge command hit Windows
  argument-length error 206 after passing too many shard paths. The runner now
  writes merge inputs to a text file and calls `merge_wthor_blend_results.py`
  with `--input-list`. A merge-only recovery then merged 332 completed shards.
- Chunk 025 validated the input-list merge path in a normal continuation run,
  completed 37 shards (`6345..7085`) with 5254.488 MiB peak RSS, and merged
  369 completed shards successfully.
- Chunk 026 completed 39 shards (`7085..7865`) with 5214.230 MiB peak RSS,
  and merged 408 completed shards successfully.
- Chunk 027 completed 33 shards (`7865..8525`) with 5209.961 MiB peak RSS,
  and merged 441 completed shards successfully.
- Chunk 028 completed 30 shards (`8525..9125`) with 5290.145 MiB peak RSS,
  and merged 471 completed shards successfully.
- Chunk 029 completed 38 shards (`9125..9885`) with 5211.504 MiB peak RSS,
  and merged 509 completed shards successfully.
- Chunk 030 completed 39 shards (`9885..10665`) with 5250.785 MiB peak RSS,
  and merged 548 completed shards successfully.
- Chunk 031 completed 45 shards (`10665..11565`) with 5282.750 MiB peak RSS,
  and merged 593 completed shards successfully.
- Chunk 032 completed 42 shards (`11565..12405`) with 5210.551 MiB peak RSS,
  and merged 635 completed shards successfully.
- Chunk 033 completed 42 shards (`12405..13245`) with 5278.707 MiB peak RSS,
  and merged 677 completed shards successfully.
- Current completed total: 13,245 / 8,035,282 position samples.
- Chunk 002 resource: 94.232 sec, peak RSS 2061.105 MiB.
- Chunk 003 resource: 91.211 sec, peak RSS 2803.953 MiB.
- A merge-only refresh after the runner change rewrote `manifest.json` to
  2,863 bytes and confirmed `all_completed_position_samples=35`.
- Chunk 004 resource: 21.289 sec, peak RSS 1325.035 MiB.
- Chunk 005 resource: 25.359 sec, peak RSS 1320.488 MiB.
- Chunk 006 resource: 97.261 sec, peak RSS 1319.250 MiB.
- Chunk 007 resource: 40.542 sec, peak RSS 1321.125 MiB.
- Chunk 008 resource: 65.883 sec, peak RSS 1318.891 MiB.
- Chunk 009 resource: 75.021 sec, peak RSS 2634.234 MiB.
- Chunk 010 resource: 75.996 sec, peak RSS 5205.410 MiB.
- Chunk 011 resource: 140.770 sec, peak RSS 5203.469 MiB.
- Chunk 012 resource: 203.605 sec, peak RSS 5207.145 MiB.
- Chunk 013 resource: 61.803 sec, peak RSS 5204.996 MiB.
- Chunk 014 resource: 60.910 sec, peak RSS 5208.156 MiB.
- Chunk 015 resource: 64.938 sec, peak RSS 5212.191 MiB.
- Chunk 016 resource: 71.052 sec, peak RSS 5221.848 MiB.
- Chunk 017 resource: 304.566 sec, peak RSS 5226.602 MiB.
- Chunk 018 resource: 303.506 sec, peak RSS 5213.312 MiB.
- Chunk 019 elapsed: 303.996 sec. Peak RSS was not sampled because this was
  a direct run.
- Chunk 020 resource: 306.519 sec, peak RSS 5272.352 MiB.
- Chunk 021 resource: 310.645 sec, peak RSS 5216.648 MiB.
- Chunk 022 resource: 126.995 sec, peak RSS 10349.910 MiB.
- Chunk 023 resource: 307.399 sec, peak RSS 5265.848 MiB.
- Chunk 024 resource: 309.350 sec, peak RSS 5322.113 MiB. Return code was
  1 because of the merge command-line length error after the shards completed.
- Chunk 024 merge-only recovery: 2.034 sec, peak RSS 33.066 MiB.
- Chunk 025 resource: 324.865 sec, peak RSS 5254.488 MiB.
- Chunk 026 resource: 304.312 sec, peak RSS 5214.230 MiB.
- Chunk 027 resource: 304.473 sec, peak RSS 5209.961 MiB.
- Chunk 028 resource: 318.616 sec, peak RSS 5290.145 MiB.
- Chunk 029 resource: 327.859 sec, peak RSS 5211.504 MiB.
- Chunk 030 resource: 303.320 sec, peak RSS 5250.785 MiB.
- Chunk 031 resource: 305.466 sec, peak RSS 5282.750 MiB.
- Chunk 032 resource: 303.380 sec, peak RSS 5210.551 MiB.
- Chunk 033 resource: 309.481 sec, peak RSS 5278.707 MiB.

`jobs_per_shard=4` is the current practical WTHOR continuation setting. It is
more memory-hungry, but it advanced 120 position samples in 140.770 sec in
chunk 011, while `jobs_per_shard=2` advanced 20 position samples in 75.021 sec
in chunk 009.
With `jobs_per_shard=4`, `egaroucid_threads=8` was the best short benchmark:
180 position samples in 64.938 sec. `egaroucid_threads=16` did not improve the
60-second chunk and used slightly more memory. Chunk 022 showed that
`jobs_per_shard=8, egaroucid_threads=4` was worse for this machine: it advanced
240 position samples in 126.995 sec, about 1.890 position samples/sec, while
chunk 023 with `jobs_per_shard=4, egaroucid_threads=8` advanced 660 position
samples in 307.399 sec, about 2.147 position samples/sec, and used about half
the peak RSS.

Current `partial_merged` top-1 symmetric accuracy on the first 13,245 position
samples:

| Blend param | Top-1 symmetric |
| ---: | ---: |
| 0.0 | 0.585051 |
| 0.1 | 0.510608 |
| 0.2 | 0.510608 |
| 0.3 | 0.510608 |
| 0.4 | 0.510155 |
| 0.5 | 0.510381 |
| 0.6 | 0.507663 |
| 0.7 | 0.502001 |
| 0.8 | 0.479275 |
| 0.9 | 0.435636 |
| 1.0 | 0.382861 |

### Strength Benchmark

The strength runner starts engine processes lazily instead of prestarting all
player slots. `blend_param=1.0` skips Egaroucid `hint 100`, and blended GTP
engines can cache hint output. Completed tasks are appended to
`strength_games.jsonl`; `--resume` reloads that file and runs only unfinished
tasks. `40_test_strength/run_strength_full.py` wraps the full schedule and
appends a raw run log to `run_strength_full.log`. The strength runner now also
supports `--time-limit-sec`, so the 120,000-game full schedule can be advanced
in bounded resumable chunks. It also supports `--task-retries` and writes
failed task details to `strength_failed_tasks.jsonl` so single engine timeouts
do not have to invalidate already saved games.

Smoke results:

- `egaroucid_l1` vs `blend_1.0`, 2 games: 2.034 sec after the policy-only skip.
- `egaroucid_l1` vs `blend_0.0`, 1 game: 37.499 sec, peak RSS 2567.539 MiB.
- Full-player short benchmark, `--max-games 2`: 152.064 sec, peak RSS 5109.445 MiB.
- Time-limit smoke, `egaroucid_l1` vs `blend_1.0`, target 4 games,
  `--time-limit-sec 1`: completed 2/4 games, stopped with
  `stop_reason=time_limit`, wrapper elapsed 3.039 sec, peak RSS 2608.930 MiB.
- Full schedule chunk 001 used the requested 32 parallel matches with
  `--resume --time-limit-sec 120`. It completed 92 / 120,000 games, stopped
  with `stop_reason=time_limit`, elapsed 729.340 sec including already-started
  game completion, and peaked at 112208.027 MiB RSS. No Egaroucid or Python
  child processes remained afterward.
- Full schedule chunk 002 kept 32 worker threads but used
  `--processes-per-player 8 --resume --time-limit-sec 60`. It advanced the
  run to 158 / 120,000 games, elapsed 771.611 sec, and peaked at 111064.254
  MiB RSS. Lowering `processes_per_player` did not materially reduce peak RSS,
  so the next optimization target is the per-process blended engine footprint.
- Full schedule chunk 003 kept 32 worker threads and
  `--processes-per-player 8`, but added `--no-blend-cache` to disable the
  per-process Egaroucid hint cache in blended engines. It advanced the run to
  232 / 120,000 games, elapsed 719.563 sec, and peaked at 117426.348 MiB RSS.
  The cache switch did not reduce memory, so the dominant cost is likely the
  number of simultaneously resident GTP/blended engine processes.
- Full schedule chunk 004 kept 32 worker threads,
  `--processes-per-player 8`, and `--no-blend-cache`, then added
  `--close-processes-after-game`. It advanced the run to 302 / 120,000 games,
  elapsed 735.136 sec, and peaked at 72669.684 MiB RSS. Closing engines after
  each game reduced peak memory by about 38% versus chunk 003, confirming that
  long-lived per-player process pools were the main memory pressure.
- Full schedule chunk 005 used the same close-after-game setting but still used
  a 300 sec Egaroucid hint timeout. It saved 50 additional games before one
  blended engine returned an Egaroucid prompt timeout and the pre-retry runner
  aborted with return code 1. Resource monitor elapsed 579.076 sec and peaked
  at 70188.805 MiB RSS. The next resume confirmed 352 saved games.
- Full schedule chunk 006 enabled `--task-retries 2` and raised
  `--egaroucid-timeout-sec` to 600 while keeping 32 worker threads and
  `--close-processes-after-game`. It advanced the run to 422 / 120,000 games,
  elapsed 1078.991 sec, peaked at 74207.070 MiB RSS, and completed without
  failed-task log entries.
- Full schedule chunk 007 used the same retry/timeout-600 close-after-game
  setting. It advanced the run to 490 / 120,000 games, elapsed 786.249 sec,
  peaked at 70102.605 MiB RSS, and completed without failed-task log entries.
- Full schedule chunk 008 again used the retry/timeout-600 close-after-game
  setting. It advanced the run to 558 / 120,000 games, elapsed 972.386 sec,
  peaked at 71333.969 MiB RSS, and completed without failed-task log entries.
- Full schedule chunk 009 used the same retry/timeout-600 close-after-game
  setting. It advanced the run to 622 / 120,000 games, elapsed 731.654 sec,
  peaked at 67624.945 MiB RSS, and completed without failed-task log entries.
- Full schedule chunk 010 used the same retry/timeout-600 close-after-game
  setting. It advanced the run to 690 / 120,000 games, elapsed 927.042 sec,
  peaked at 77386.398 MiB RSS, and completed without failed-task log entries.

The full requested schedule is 120,000 games. The short full-player benchmark
suggests a multi-day run even with 32 parallel matches, and `hint 100` required
raising the strength-runner timeout from 60 sec to 300 sec. The first full
schedule chunk also showed that `parallel_matches=32` with
`processes_per_player=32` is very memory-heavy on this machine. Chunk 002
showed that lowering only `processes_per_player` is not enough, and chunk 003
showed that disabling the blended-engine hint cache is not enough either. Chunk
004 showed that closing engines after each game is an effective memory-control
knob while preserving the requested 32 worker threads, though the wall-clock
time remains high because `hint 100` level-21 searches dominate the run. Chunk
005 showed that a 300 sec hint timeout is still too tight for some strength
positions; chunk 006 therefore uses 600 sec for continuation.

## 日本語

### 用語

この実験では、`games` は完全な棋譜、つまり対局数を意味します。学習用に生成された局面は
`position_samples` と呼びます。`records0` や `records1` という既存ディレクトリ名は、
データセットのパス名としてだけ残しています。

### データセット

- 入力元: `train_data/transcript_release/0002`
- 選択: 1,000,000 対局、seed `613`
- 変換成功: 1,000,000 対局
- 変換失敗: 0 対局
- 生成された `position_samples`: 26,384,206
- 変換リソース: 179.371 秒、peak RSS 14.238 MiB

1対局から、ランダム序盤後の複数の着手局面を書き出すため、1,000,000 対局から
26,384,206 局面サンプルが生成されています。

### 学習探索

すべての実行で、学習 5,000,000 `position_samples`、検証 500,000 `position_samples`、
batch size 8192、20 epochs、patience 5、TensorFlow/Keras を使いました。

| Config | Params | Val top-1 | Val top-3 | Val top-5 | Time sec | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x3 | 20,736 | 0.297304 | 0.584340 | 0.742064 | 33.480 | 7751.410 |
| 96x3 | 37,216 | 0.308512 | 0.604394 | 0.763330 | 33.417 | 7787.094 |
| 128x3 | 57,792 | 0.326106 | 0.626642 | 0.783058 | 33.446 | 7782.047 |
| 96x4 | 46,528 | 0.313866 | 0.610754 | 0.767646 | 34.501 | 7784.199 |
| 160x3 | 82,464 | 0.341052 | 0.649038 | 0.804222 | 34.443 | 7794.395 |
| 192x3 | 111,232 | 0.353422 | 0.664336 | 0.816872 | 35.499 | 7738.395 |
| 128x4 | 74,304 | 0.329930 | 0.632604 | 0.787496 | 35.519 | 7652.391 |

検証で最も良かった設定は `192x3` です。

### 修正後の WTHOR 一致率

WTHOR 評価では、policy 出力 index と同じ向きの合法手マスクを使うように修正しました。
途中で出力していた feature-order bit mask の値は破棄してください。

| Config | Positions | Top-1 exact | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192x3 | 8,035,282 | 0.338097 | 0.360323 | 0.707869 | 0.870466 |
| 160x3 | 8,035,282 | 0.328847 | 0.351074 | 0.683399 | 0.852199 |
| 128x4 | 8,035,282 | 0.313147 | 0.335372 | 0.661174 | 0.841202 |
| 128x3 | 8,035,282 | 0.307269 | 0.329494 | 0.663007 | 0.839235 |
| 96x3 | 8,035,282 | 0.298108 | 0.314827 | 0.648127 | 0.824891 |
| 96x4 | 8,035,282 | 0.292177 | 0.308892 | 0.652110 | 0.829590 |
| 64x3 | 8,035,282 | 0.289575 | 0.306292 | 0.631556 | 0.810074 |

WTHOR 一致率でも最良は `192x3` でした。

`192x3` の10手刻み:

| Moves | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| --- | ---: | ---: | ---: |
| 01-10 | 0.377334 | 0.812311 | 0.964305 |
| 11-20 | 0.292862 | 0.612274 | 0.799897 |
| 21-30 | 0.262999 | 0.577686 | 0.775109 |
| 31-40 | 0.280449 | 0.614852 | 0.811226 |
| 41-50 | 0.354634 | 0.715394 | 0.887165 |
| 51-60 | 0.595946 | 0.916727 | 0.986230 |

### ブレンド評価ベンチマーク

Egaroucid Console 7.8.1 level 21 の `hint 100` は1局面あたりかなり重いです。
WTHOR からランダムに30局面を選んだ評価では 86.157 秒、peak RSS 1283.957 MiB でした。
キャッシュなしで単純外挿すると、WTHOR 全 8,035,282 局面に対して1プロセスで約267日かかるため、
このセッション内で全局面のブレンド評価を直接実行するのは現実的ではありません。

WTHOR ブレンド評価は `--jobs`、`--range-start` / `--range-end`、`--hint-cache-db` に対応しています。
`20_test_with_wthor/run_wthor_blend_shards.py` はそれらをまとめ、完了済み shard を skip しながら
再開可能に実行し、既定で共有 SQLite hint-score cache を作成します。

seed `613`、30局面の小サンプル:

| Blend param | Top-1 symmetric | Top-3 symmetric | Top-5 symmetric |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.566667 | 0.866667 | 1.000000 |
| 0.25 | 0.466667 | 0.733333 | 0.900000 |
| 0.50 | 0.466667 | 0.733333 | 0.866667 |
| 0.75 | 0.400000 | 0.700000 | 0.900000 |
| 1.00 | 0.300000 | 0.666667 | 0.900000 |

この表は時間見積もりとブレンド処理の確認用で、最終的な係数決定にはもっと大きなサンプルか並列評価が必要です。

### WTHOR hint-cache 計画

`20_test_with_wthor/analyze_wthor_position_duplicates.py` で、Egaroucid の `hint 100` に渡す
実盤面キー `(black, white, side)` の重複を数えました。

| Metric | Value |
| --- | ---: |
| WTHOR `position_samples` | 8,035,282 |
| Unique hint positions | 5,574,955 |
| Duplicate hint positions | 2,460,327 |
| Duplicate fraction | 0.306190 |
| Invalid side samples | 0 |
| Analysis elapsed | 6.745 秒 |
| Analysis in-process peak RSS | 440.371 MiB |
| Resource-monitor elapsed | 7.104 秒 |
| Resource-monitor peak RSS | 293.832 MiB |

cache により、WTHOR 内で重複している約30.6%の `hint 100` 呼び出しを省けます。
また、中断・再開した shard 実行でも完了済みの hint score を再計算しなくて済みます。
ただし、unique hint positions はまだ 5.57 million あるため、Egaroucid 側の hint cost を下げるか、
評価対象をサンプリングしない限り、全局面ブレンド評価は依然としてかなり大きい実行になります。

WTHOR 先頭2局面での cache smoke test:

| Run | Elapsed sec | Peak RSS MiB | Lookups | Hits | Misses | Writes | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First run | 2.032 | 1325.613 | 2 | 0 | 2 | 2 | 2 |
| Second run, same DB | 1.030 | 8.523 | 2 | 2 | 0 | 0 | 2 |

2 shard の smoke run も通り、merge 後の出力にも cache stats が残ることを確認しました:
2 lookups、0 hits、2 misses、2 writes、2 rows。

### 再開可能な full 実行制御

WTHOR shard runner は `--positions-per-shard`、`--range-start` / `--range-end`、
`--time-limit-sec` に対応しました。これにより、全WTHORブレンド評価を巨大な shard 単位ではなく、
再開可能な小さい単位で進められます。
`--merge-completed` を使うと、全WTHORが完走する前でも完了済み shard だけから
`partial_merged` を更新できます。

`--positions-per-shard 5` のように shard 数が非常に多い場合でも、runner は全 shard を
`manifest.json` に展開しません。`shard_count` と先頭・末尾 preview だけを書き、表示数は
`--manifest-shard-preview` で調整します。そのため、`--range-start 30` から継続しても
巨大な manifest は作られません。

WTHOR 5局面、`positions_per_shard=2`、`blend_param=1.0`、`top_n=1,3` の smoke test:

- 1回目は `--time-limit-sec 0.001` で shard `0..2` だけ完了し、
  `stop_reason=time_limit` で停止しました。
- 2回目は同じ出力先を使い、shard 0 を skip して shard `2..4` を実行し、
  `stop_reason=max_shards_to_run` で停止しました。
- 3回目は shard 0 と 1 を skip して shard `4..5` を実行し、全 shard を merge して
  `stop_reason=finished` になりました。

merge 後の smoke 結果:

| Blend param | Top-N | Exact accuracy | Symmetric accuracy | Positions |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1 | 0.200000 | 0.400000 | 5 |
| 1.0 | 3 | 0.800000 | 0.800000 | 5 |

`20_test_with_wthor/output/blend_wthor_full_chunked` での full-run 進捗:

- chunk 001 で WTHOR 20 / 8,035,282 局面サンプルまで完了。
- chunk 002 では同じ出力先を使い、完了済みの最初の2 shard を skip して1 shard 追加し、
  `partial_merged` を更新しました。
- chunk 003 では `--range-start 30` と `--positions-per-shard 5` で shard `30..35` を完了し、
  `partial_merged` を更新しました。
- manifest/schedule 修正後、chunk 004 では `--range-start 35` から 5局面サンプル shard を6つ進め、
  `35..65` まで完了しました。
- chunk 005 では `--range-start 65` から 5局面サンプル shard を2つ進め、
  `65..75` まで完了しました。
- chunk 006 では `--positions-per-shard 25` を試し、shard `75..100` を完了しました。
  shard を大きくすると time-limit の粒度が粗くなることが分かりました。
- chunk 007 では `--resume-from-completed-prefix` を使い、自動で `100` から開始して、
  5局面サンプル shard を7つ進め、`100..135` まで完了しました。
- chunk 008 では同じ自動再開 p5 設定を60秒制限で使い、5局面サンプル shard を2つ進め、
  `135..145` まで完了しました。
- chunk 009 では `--jobs-per-shard 2` と `--positions-per-shard 20` を試し、
  shard `145..165` を完了しました。peak RSS は 2634.234 MiB でした。
- chunk 010 では `--jobs-per-shard 4` と `--positions-per-shard 20` を試し、
  `165..225` まで3 shard 完了しました。peak RSS は 5205.410 MiB でした。
- chunk 011 では `jobs_per_shard=4` のまま120秒制限にし、`225..345` まで6 shard 完了しました。
  peak RSS は 5203.469 MiB でした。
- chunk 012 では `jobs_per_shard=4` のまま180秒制限にし、`345..525` まで9 shard 完了しました。
  peak RSS は 5207.145 MiB でした。
- chunk 013 では `egaroucid_threads=2` を試し、`525..605` まで4 shard 完了しました。
  peak RSS は 5204.996 MiB でした。
- chunk 014 では `egaroucid_threads=4` を試し、`605..705` まで5 shard 完了しました。
  peak RSS は 5208.156 MiB でした。
- chunk 015 では `egaroucid_threads=8` を試し、`705..885` まで9 shard 完了しました。
  peak RSS は 5212.191 MiB でした。
- chunk 016 では `egaroucid_threads=16` を試し、`885..1065` まで9 shard 完了しました。
  peak RSS は 5221.848 MiB でした。
- chunk 017 では現時点の短時間ベンチ最良設定 `jobs_per_shard=4, egaroucid_threads=8` を使い、
  300秒制限で `1065..1845` まで39 shard 完了しました。peak RSS は 5226.602 MiB でした。
- chunk 018 では同じ設定を使い、300秒制限で `1845..2545` まで35 shard 完了しました。
  peak RSS は 5213.312 MiB でした。
- chunk 019 では同じ設定を使い、300秒制限で `2545..3205` まで33 shard 完了しました。
  直接実行だったため peak RSS は未計測です。`progress_summary.json` 上の elapsed は
  303.996 秒でした。
- chunk 020 では同じ設定を resource monitor 経由で実行し、`3205..3925` まで36 shard
  完了しました。peak RSS は 5272.352 MiB でした。
- chunk 021 では同じ設定を resource monitor 経由で実行し、`3925..4645` まで36 shard
  完了しました。peak RSS は 5216.648 MiB でした。
- chunk 022 では `jobs_per_shard=8, egaroucid_threads=4` を試し、`4645..4885` まで
  12 shard 完了しました。peak RSS は 10349.910 MiB でした。
- chunk 023 では `jobs_per_shard=4, egaroucid_threads=8` に戻し、`4885..5545` まで
  33 shard 完了しました。peak RSS は 5265.848 MiB でした。
- chunk 024 では `5545..6345` まで40 shard 完了しました。peak RSS は 5322.113 MiB でした。
  評価 shard は成功しましたが、旧 merge コマンドは shard パスを全部コマンドラインに並べたため、
  Windows の引数長制限 error 206 に当たりました。runner は merge 入力をテキストファイルに
  書き出し、`merge_wthor_blend_results.py --input-list` で読む方式に修正しました。その後の
  merge-only recovery で完了済み332 shard を回収しました。
- chunk 025 では通常の継続実行で input-list merge 経路を確認し、`6345..7085` まで
  37 shard 完了しました。peak RSS は 5254.488 MiB で、完了済み369 shard を正常に
  merge できました。
- chunk 026 では `7085..7865` まで39 shard 完了しました。peak RSS は 5214.230 MiB で、
  完了済み408 shard を正常に merge できました。
- chunk 027 では `7865..8525` まで33 shard 完了しました。peak RSS は 5209.961 MiB で、
  完了済み441 shard を正常に merge できました。
- chunk 028 では `8525..9125` まで30 shard 完了しました。peak RSS は 5290.145 MiB で、
  完了済み471 shard を正常に merge できました。
- chunk 029 では `9125..9885` まで38 shard 完了しました。peak RSS は 5211.504 MiB で、
  完了済み509 shard を正常に merge できました。
- chunk 030 では `9885..10665` まで39 shard 完了しました。peak RSS は 5250.785 MiB で、
  完了済み548 shard を正常に merge できました。
- chunk 031 では `10665..11565` まで45 shard 完了しました。peak RSS は 5282.750 MiB で、
  完了済み593 shard を正常に merge できました。
- chunk 032 では `11565..12405` まで42 shard 完了しました。peak RSS は 5210.551 MiB で、
  完了済み635 shard を正常に merge できました。
- chunk 033 では `12405..13245` まで42 shard 完了しました。peak RSS は 5278.707 MiB で、
  完了済み677 shard を正常に merge できました。
- 現在の完了数: 13,245 / 8,035,282 局面サンプル。
- chunk 002 resource: 94.232 秒、peak RSS 2061.105 MiB。
- chunk 003 resource: 91.211 秒、peak RSS 2803.953 MiB。
- runner 変更後の merge-only refresh で `manifest.json` は 2,863 bytes になり、
  `all_completed_position_samples=35` を確認しました。
- chunk 004 resource: 21.289 秒、peak RSS 1325.035 MiB。
- chunk 005 resource: 25.359 秒、peak RSS 1320.488 MiB。
- chunk 006 resource: 97.261 秒、peak RSS 1319.250 MiB。
- chunk 007 resource: 40.542 秒、peak RSS 1321.125 MiB。
- chunk 008 resource: 65.883 秒、peak RSS 1318.891 MiB。
- chunk 009 resource: 75.021 秒、peak RSS 2634.234 MiB。
- chunk 010 resource: 75.996 秒、peak RSS 5205.410 MiB。
- chunk 011 resource: 140.770 秒、peak RSS 5203.469 MiB。
- chunk 012 resource: 203.605 秒、peak RSS 5207.145 MiB。
- chunk 013 resource: 61.803 秒、peak RSS 5204.996 MiB。
- chunk 014 resource: 60.910 秒、peak RSS 5208.156 MiB。
- chunk 015 resource: 64.938 秒、peak RSS 5212.191 MiB。
- chunk 016 resource: 71.052 秒、peak RSS 5221.848 MiB。
- chunk 017 resource: 304.566 秒、peak RSS 5226.602 MiB。
- chunk 018 resource: 303.506 秒、peak RSS 5213.312 MiB。
- chunk 019 elapsed: 303.996 秒。直接実行だったため peak RSS は未計測です。
- chunk 020 resource: 306.519 秒、peak RSS 5272.352 MiB。
- chunk 021 resource: 310.645 秒、peak RSS 5216.648 MiB。
- chunk 022 resource: 126.995 秒、peak RSS 10349.910 MiB。
- chunk 023 resource: 307.399 秒、peak RSS 5265.848 MiB。
- chunk 024 resource: 309.350 秒、peak RSS 5322.113 MiB。評価後の merge 引数長エラーで
  return code は 1 でした。
- chunk 024 merge-only recovery: 2.034 秒、peak RSS 33.066 MiB。
- chunk 025 resource: 324.865 秒、peak RSS 5254.488 MiB。
- chunk 026 resource: 304.312 秒、peak RSS 5214.230 MiB。
- chunk 027 resource: 304.473 秒、peak RSS 5209.961 MiB。
- chunk 028 resource: 318.616 秒、peak RSS 5290.145 MiB。
- chunk 029 resource: 327.859 秒、peak RSS 5211.504 MiB。
- chunk 030 resource: 303.320 秒、peak RSS 5250.785 MiB。
- chunk 031 resource: 305.466 秒、peak RSS 5282.750 MiB。
- chunk 032 resource: 303.380 秒、peak RSS 5210.551 MiB。
- chunk 033 resource: 309.481 秒、peak RSS 5278.707 MiB。

現時点の実用的な WTHOR 継続設定は `jobs_per_shard=4` です。メモリは大きくなりますが、
chunk 011 では 140.770 秒で 120 局面サンプル進みました。一方、`jobs_per_shard=2` の chunk 009 は
75.021 秒で 20 局面サンプルでした。
`jobs_per_shard=4` の中では `egaroucid_threads=8` が短時間ベンチで最良でした:
64.938 秒で 180 局面サンプル進みました。`egaroucid_threads=16` は改善せず、メモリも少し増えました。
chunk 022 では `jobs_per_shard=8, egaroucid_threads=4` も試しましたが、126.995 秒で
240 局面サンプル、約 1.890 局面サンプル/秒でした。chunk 023 の
`jobs_per_shard=4, egaroucid_threads=8` は 307.399 秒で 660 局面サンプル、
約 2.147 局面サンプル/秒だったため、従来設定を継続します。

現在の `partial_merged` における先頭13,245局面サンプルの top-1 symmetric accuracy:

| Blend param | Top-1 symmetric |
| ---: | ---: |
| 0.0 | 0.585051 |
| 0.1 | 0.510608 |
| 0.2 | 0.510608 |
| 0.3 | 0.510608 |
| 0.4 | 0.510155 |
| 0.5 | 0.510381 |
| 0.6 | 0.507663 |
| 0.7 | 0.502001 |
| 0.8 | 0.479275 |
| 0.9 | 0.435636 |
| 1.0 | 0.382861 |

### 強さ評価ベンチマーク

強さ評価 runner は、全 player slot を事前起動する方式から lazy start に変更しました。
`blend_param=1.0` では Egaroucid `hint 100` を呼ばず、ブレンド GTP engine では hint output を
cache できます。完了済み task は `strength_games.jsonl` に追記され、`--resume` で未完了 task だけを
実行できます。`40_test_strength/run_strength_full.py` は full schedule 用の wrapper で、raw run log を
`run_strength_full.log` に追記します。`--time-limit-sec` にも対応したため、120,000対局の full schedule を
時間で区切って再開可能に進められます。また `--task-retries` にも対応し、失敗した task の詳細を
`strength_failed_tasks.jsonl` に保存するため、単発の engine timeout で保存済み対局まで無駄にしなくて
済むようにしました。

smoke 結果:

- `egaroucid_l1` vs `blend_1.0`、2対局: policy-only skip 後 2.034 秒。
- `egaroucid_l1` vs `blend_0.0`、1対局: 37.499 秒、peak RSS 2567.539 MiB。
- 全プレイヤー構成の短縮ベンチ `--max-games 2`: 152.064 秒、peak RSS 5109.445 MiB。
- time-limit smoke、`egaroucid_l1` vs `blend_1.0`、目標4対局、
  `--time-limit-sec 1`: 2/4 対局を完了し、`stop_reason=time_limit` で停止。
  wrapper elapsed 3.039 秒、peak RSS 2608.930 MiB。
- full schedule chunk 001 では、要求通り 32 並列で `--resume --time-limit-sec 120` を実行しました。
  92 / 120,000 対局を完了し、`stop_reason=time_limit` で停止しました。
  起動済み対局の終了待ち込みで elapsed は 729.340 秒、peak RSS は 112208.027 MiB でした。
  実行後に Egaroucid や Python の子プロセスが残っていないことも確認しました。
- full schedule chunk 002 では 32 worker threads は維持しつつ
  `--processes-per-player 8 --resume --time-limit-sec 60` を試しました。
  158 / 120,000 対局まで進み、elapsed は 771.611 秒、peak RSS は 111064.254 MiB でした。
  `processes_per_player` を下げても peak RSS は大きくは下がらなかったため、次は blended engine
  1プロセスあたりのメモリを下げる方向を確認します。
- full schedule chunk 003 では 32 worker threads と `--processes-per-player 8`
  は維持し、`--no-blend-cache` で blended engine 内の Egaroucid hint キャッシュを無効化しました。
  232 / 120,000 対局まで進み、elapsed は 719.563 秒、peak RSS は 117426.348 MiB でした。
  キャッシュ無効化でもメモリは下がらなかったため、主因は同時常駐している GTP / blended engine
  プロセス数そのものに近いと考えています。
- full schedule chunk 004 では 32 worker threads、`--processes-per-player 8`、
  `--no-blend-cache` は維持し、`--close-processes-after-game` を追加しました。
  302 / 120,000 対局まで進み、elapsed は 735.136 秒、peak RSS は 72669.684 MiB でした。
  chunk 003 比で peak memory は約38%下がり、長寿命の player 別 process pool が主なメモリ圧迫要因
  だったことを確認できました。
- full schedule chunk 005 では同じ close-after-game 設定を使いましたが、Egaroucid hint timeout は
  300秒のままでした。失敗前に50対局を保存し、その後 blended engine が
  Egaroucid prompt timeout を返したため、retry 実装前の runner は return code 1 で停止しました。
  resource monitor の elapsed は 579.076 秒、peak RSS は 70188.805 MiB でした。
  次の resume で 352 対局が保存済みであることを確認しました。
- full schedule chunk 006 では 32 worker threads と `--close-processes-after-game` を維持し、
  `--task-retries 2` を有効化し、`--egaroucid-timeout-sec 600` に上げました。
  422 / 120,000 対局まで進み、elapsed は 1078.991 秒、peak RSS は 74207.070 MiB でした。
  failed-task log は生成されず、正常終了しました。
- full schedule chunk 007 では同じ retry / timeout 600 / close-after-game 設定を使いました。
  490 / 120,000 対局まで進み、elapsed は 786.249 秒、peak RSS は 70102.605 MiB でした。
  failed-task log は生成されず、正常終了しました。
- full schedule chunk 008 でも同じ retry / timeout 600 / close-after-game 設定を使いました。
  558 / 120,000 対局まで進み、elapsed は 972.386 秒、peak RSS は 71333.969 MiB でした。
  failed-task log は生成されず、正常終了しました。
- full schedule chunk 009 でも同じ retry / timeout 600 / close-after-game 設定を使いました。
  622 / 120,000 対局まで進み、elapsed は 731.654 秒、peak RSS は 67624.945 MiB でした。
  failed-task log は生成されず、正常終了しました。
- full schedule chunk 010 でも同じ retry / timeout 600 / close-after-game 設定を使いました。
  690 / 120,000 対局まで進み、elapsed は 927.042 秒、peak RSS は 77386.398 MiB でした。
  failed-task log は生成されず、正常終了しました。

要求された full schedule は 120,000 対局です。短縮ベンチから見ても、32並列でも数日規模の実行になる見込みです。
また `hint 100` が60秒で戻らない局面があったため、strength runner の timeout 既定値を300秒に上げました。
最初の full schedule chunk では `parallel_matches=32` かつ `processes_per_player=32` がこの環境では
非常に大きいメモリを使うことも分かりました。chunk 002 では `processes_per_player` だけを下げても
不十分で、chunk 003 では blended engine の hint キャッシュを切っても不十分でした。
chunk 004 では game 終了ごとに engine を閉じることで、32 worker threads を保ったままメモリを
大きく下げられました。ただし `hint 100` level-21 探索が支配的なので、wall-clock time は依然として
大きいです。chunk 005 で 300秒 timeout でも足りない局面が出たため、chunk 006 以降の継続では
600秒 timeout を使います。
