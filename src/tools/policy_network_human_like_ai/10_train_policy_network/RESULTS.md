# Training Results

Related issue: #613

This file records tracked summaries for the human-like policy-network training
work. Detailed raw logs are kept under
`src/tools/policy_network_human_like_ai/report` and
`src/tools/policy_network_human_like_ai/10_train_policy_network/train_log`;
both directories are intentionally ignored by git.

関連 issue: #613

このファイルには、人間らしい policy network 学習作業の要約を残します。
詳細な生ログは
`src/tools/policy_network_human_like_ai/report` と
`src/tools/policy_network_human_like_ai/10_train_policy_network/train_log`
に保存します。どちらも意図的に git 管理外です。

## Current Status / 現在の状態

The production experiment has not been run yet because the number of selected
Egaroucid Train Data v2 games, `N`, still needs to be decided.

本番実験はまだ実行していません。Egaroucid Train Data v2 から選択する棋譜数
`N` の確認が必要です。

## Smoke Test / 煙突テスト

The full pipeline was checked with 20 selected games:

- selected games: 20
- valid converted games: 20
- invalid converted games: 0
- generated board records: 502
- training config: `16x1`
- epochs: 1
- TensorFlow: 2.10.0
- result: training finished successfully

20 局を使って全体の流れを確認しました。

- 選択棋譜: 20
- 変換成功棋譜: 20
- 変換失敗棋譜: 0
- 生成 board records: 502
- 学習 config: `16x1`
- epoch: 1
- TensorFlow: 2.10.0
- 結果: 学習完了

## WTHOR Evaluator Smoke Test / WTHOR 評価スクリプト煙突テスト

The WTHOR evaluator was checked on the first 256 WTHOR positions using an
existing policy-network weight file:
`src/tools/policy_network/trained/playerop_final_issue613_128x3/best_policy_network_weights.bin`.

既存の policy-network weight
`src/tools/policy_network/trained/playerop_final_issue613_128x3/best_policy_network_weights.bin`
を使い、WTHOR の先頭 256 局面で評価スクリプトを確認しました。

| Top N | Exact | Symmetry-aware |
| ---: | ---: | ---: |
| 1 | 69.922% | 71.875% |
| 2 | 83.594% | 85.547% |
| 3 | 89.062% | 91.016% |
| 4 | 95.703% | 95.703% |
| 5 | 97.656% | 97.656% |
| 8 | 100.000% | 100.000% |
| 10 | 100.000% | 100.000% |
| 16 | 100.000% | 100.000% |

These are smoke-test numbers only, not final experimental results.

これは smoke test の数値であり、本番実験の結果ではありません。
