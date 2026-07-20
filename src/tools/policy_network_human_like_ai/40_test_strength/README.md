# Policy Networkブレンド方策の総当たり戦

関連issue: #613

このフォルダのスクリプトは、XOT局面から先後を入れ替えた2局を1対戦セットとして、Egaroucid for Console単体とPolicy Networkブレンド方策の総当たり戦を実行する。

## 本番設定

`run_strength_full.py`の既定値は、次の本番設定に固定している。

| 項目 | 値 |
| --- | ---: |
| Egaroucid for Console | level 1、3、5、7、9、11、13、15、17、19 |
| ブレンド方策 | α=0.0、0.2、0.4、0.6、0.8、1.0 |
| ランダム打ち | 参加させない |
| 参加者数 | 16 |
| 参加者の組み合わせ数 | 120 |
| 1組当たり | 100対戦セット、200実対局 |
| 対戦セット総数 | 12,000 |
| 実対局総数 | 24,000 |
| 各参加者の合計 | 1,500対戦セット、3,000実対局 |
| XOT | `bin/problem/xot/openingslarge.txt` |
| XOTのシャッフルseed | 57 |
| スケジューラへ同時投入する対戦セット数の上限 | 16 |
| 各参加者が同時に行える対戦セット数 | 1 |
| 実際に同時実行できる対戦セット数の理論上限 | 8 |
| 実際に同時実行できる実対局数の理論上限 | 16 |
| 各参加者のエンジンプロセス数 | 2 |
| 1エンジンプロセスの探索スレッド数 | 1 |
| Policy Network推論 | TensorFlow、対応GPUがあればGPUを使用 |
| 対戦中の1手当たり推定石損測定 | 無効 |
| 資源使用状況の記録 | 有効、2秒間隔 |
| 確保する空き物理メモリ | 24,576 MiB |

Egaroucid for Consoleの各levelは、参加者ごとに独立したプロセスプールとして起動する。異なるlevel間で置換表は共有しない。

対戦中の1手当たり推定石損は測定しない。代わりに、各実対局の完全な棋譜を`strength_games.jsonl`へ保存する。この棋譜を使えば、対戦完了後に別処理として石損を測定できる。対戦中にも測定する場合だけ`--measure-move-stone-loss`を指定する。

## 必要ファイル

別PCでは、リポジトリのルートをカレントディレクトリにして、次の4ファイルが存在することを確認する。

```text
bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe
bin/problem/xot/openingslarge.txt
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_model.h5
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin
```

Python環境には`numpy`、`psutil`、TensorFlowを導入する。本番ランチャーは、過去の50セット本番と同じ`tensorflow/GPU`を使用するため、`--policy-backend tensorflow`を既定値としている。Kerasモデルを読み込めない場合にNumPyへ自動切替は行わず、実行をエラーで停止する。NVIDIA GPUの使用状況を取得するため、`nvidia-smi`も実行できる状態にする。

GPUを使用できないPCで`--policy-backend numpy`を明示すればCPU推論は可能だが、過去の本番と推論バックエンドが異なる別条件の実験になる。

## 事前確認

本番と同じ出力先を指定して乾式実行する。エンジン対局は開始せず、参加者、対戦数、パス、起動前の空きメモリと推定必要メモリを検査する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_100sets_16players_pc2 `
  --dry-run
```

出力で少なくとも次の値を確認する。

```text
random_legal_player False
match_sets_per_pair 100
actual_games_per_pair 200
total_match_sets 12000
total_actual_games 24000
measure_move_stone_loss_during_tournament False
```

本番開始後の出力では`policy_batch_server_runtime tensorflow/GPU`も確認する。`tensorflow/CPU`の場合は動作するが、過去の50セット本番とは実行デバイスが異なり、所要時間も大きく変わる。

起動時のメモリ容量検査に失敗した場合は、利用可能な物理メモリを増やすか、実測に基づいて`--minimum-available-memory-mib`または`--estimated-engine-memory-mib`を調整する。検査を無効化するオプションは用意していない。

## 本番実行

乾式実行と同じ出力先を指定し、`--dry-run`だけを外す。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_100sets_16players_pc2
```

進捗、途中時点の勝率、平均石差、推定Elo、経過時間、残り時間の推定値は標準出力へ表示し、同時に`run_strength_full.log`へ保存する。資源使用状況は`performance_live.json`へ随時保存する。

既に`strength_games.jsonl`が存在する出力先へ、`--resume`なしで結果を追記することはできない。別の実験を開始する場合は、新しい`--output-dir`を指定する。

## 中断後の再開

本番実行と同じ引数へ`--resume`を追加する。完了済みの対戦IDを`strength_games.jsonl`から読み込み、未完了の対戦セットだけを実行する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_100sets_16players_pc2 `
  --resume
```

再開時に参加者、対戦セット数、XOT、seedなどを変更してはならない。

## 主な出力

| ファイル | 内容 |
| --- | --- |
| `strength_games.jsonl` | 対戦セット単位の結果、先後2局の石差、完全棋譜 |
| `strength_summary.csv` | 参加者ごとの勝敗、勝率、平均石差、推定Elo |
| `strength_pair_results.csv` | 対戦相手別の勝敗、勝率、平均石差 |
| `strength_report.txt` | 進捗・勝率・平均石差・推定Eloの表 |
| `strength_results.json` | 集計結果と実験条件 |
| `strength_progress.json` | 完了数、残り数、終了理由 |
| `performance_live.json` | 実行中の最新資源使用状況 |
| `performance_samples.csv` | CPU、GPU、物理メモリ、子プロセス数の時系列 |
| `performance_summary.json` | 資源使用状況の集計 |

`--games-per-pair`は旧コマンドとの互換用に残しているが、新しいコマンドでは意味が明確な`--match-sets-per-pair`を使用する。
