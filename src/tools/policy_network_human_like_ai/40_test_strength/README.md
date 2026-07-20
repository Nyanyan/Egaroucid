# ブレンド方策の対戦強度評価

関連issue: #613

Egaroucid for Consoleの各levelと、Policy NetworkをEgaroucidの方策へ混合した各αを、XOT局面から総当たりで対戦させる。現在の実行コードにはランダムプレイヤーの実装もコマンドラインオプションもない。

## 対戦セットと統計単位

1対戦セットでは、同じXOT局面から同じ2参加者が色を入れ替えて2局行う。参加者0から見た2局の石差を平均し、その符号でセットの勝ち・引き分け・負けを決める。

```text
セット平均石差 = (黒を持った局の石差 + 白を持った局の石差) / 2
セットスコア = (セット勝ち数 + 0.5 × セット引き分け数) / セット数
```

95%信頼区間の標本数は実対局数ではなく対戦セット数である。同じXOT局面を使う先後2局には対応があるため、500セットを1,000個の独立対局として扱わない。

実対局単位の勝敗も別列へ保存するが、これは記述統計である。対戦相手別のセットスコアと95%信頼区間、およびセット平均石差を主要指標とする。

全組合せには同じset indexで同じXOT局面を割り当てる。そのため、ある参加者の15相手分を単純に7,500個の独立標本としてまとめることはできない。参加者全体のスコアとEloは記述的な点推定だけを出力し、信頼区間を付けない。Eloは0%・100%で任意の無限大や固定clip値にならないよう、対戦済みの各組合せへ0.5勝・0.5敗を加えた有限な参考値である。未対戦の組合せは推定の重みを0とし、対戦済み組合せのグラフが全参加者を連結するまでは途中Eloを表示しない。

## 本番設定

| 項目 | 既定値 |
| --- | ---: |
| Egaroucid単体 | level 1、3、5、…、19（10参加者） |
| ブレンド方策 | α=0.0、0.2、0.4、0.6、0.8、1.0（6参加者） |
| 参加者数 | 16 |
| 組合せ数 | 120 |
| 1組合せ | 500対戦セット、1,000実対局 |
| 全体 | 60,000対戦セット、120,000実対局 |
| 各参加者 | 7,500対戦セット、15,000実対局 |
| XOT | `bin/problem/xot/openingslarge.txt` |
| XOT shuffle seed | 57 |
| XOTの割当て | 各組合せへ同じ500局面列を割り当てる |
| 同時対戦セット数 | 最大20 |
| 同時実対局数 | 最大40 |
| 実対局worker thread | 40 |
| Egaroucid単体のprocess/参加者 | 2 |
| ブレンド系列のprocess/参加者 | 10 |
| 1 engine processの探索thread | 1 |
| Policy Network推論 | 1個の共有TensorFlow/GPU server |
| Egaroucid hint cache | 実験出力先ごとのSQLite |
| 空きメモリ下限 | 16,384 MiB |
| 想定マシン | 32論理thread、128 GiB RAM |

重いlevel 21探索を使うブレンド系列へ多くのprocessを割り当てる。10個のEgaroucid単体は各1セット、6個のブレンド系列は各5セットまで同時参加でき、参加者容量から見た上限は20セットである。共有cacheやpolicy serverを待つ実対局があるため、32論理threadに対して40実対局を投入して待ち時間を隠す。探索条件を旧実験と揃えるため、1 process内の探索thread数は1のままである。

各セットで一時的なthread poolを作る旧構造は廃止した。全実対局を1個の固定thread poolへ投入し、親threadが先後2局を結合する。残り全タスクを毎回走査せず、120組合せ別のqueueを使う。同じset index、すなわち同じXOT局面を全組合せへ一巡させることを最優先し、その中で参加者ごとの残り推定仕事量を使って遅い参加者を早めに処理する。古い局面が参加者容量のため全て待機中の場合だけ、queue内で最も古い未投入局面の1局面先まで投入できる。それ以上の新規投入は禁止する。実行中タスクの完了順によって途中の完了セット数には数セットの差が出るが、一部の組合せだけが何十セットも先行することはない。これにより、並列度を保ちながら、共有hint cacheの局面再利用と途中集計の公平性を保つ。初回のprocess起動時間を通常の対局時間と誤認しないよう、未観測参加者には観測済み時間の中央値を使い、極端な推定値も中央値の0.5倍から2倍へ制限する。

hintは実験出力先の共有SQLiteだけへ保存する。各Python process内の無制限dict cacheは5日規模の実験でメモリを増やし続けるため使用しない。`--no-hint-cache`を指定した場合はSQLite自体も作成・参照しない。

既定値の最悪時見積りは、重いEgaroucid process 70個、Python wrapper 50個、共有policy serverを合わせて92,200 MiBである。さらに16,384 MiBを空きとして残すため、開始時点で約106.0 GiB以上のavailable memoryを要求する。これは128 GiB機向けの設定であり、64 GiB・96 GiB機では既定値のまま実行しない。起動後も親processと各wrapperが空きメモリを確認し、下限を割る場合は新しいprocess・セットの開始を止める。

## 必要ファイル

リポジトリのルートをカレントディレクトリとし、次のファイルが存在することを確認する。

```text
bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe
bin/problem/xot/openingslarge.txt
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_model.h5
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin
```

Python環境には`numpy`、`psutil`、`tensorflow`が必要である。既定値ではTensorFlow/GPUを要求し、GPUが見つからない場合は開始前に停止する。暗黙にNumPyやCPUへ切り替えない。意図的な確認実行だけは`--allow-policy-cpu`または`--policy-backend numpy`を明示する。

## 実行方法

最初に、ファイル、実験数、並列数、メモリ見積りを確認する。engineやpolicy serverは起動しない。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_16players `
  --dry-run
```

本番を開始する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_16players
```

中断後は同じ引数と出力先で`--resume`を付ける。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_16players `
  --resume
```

`strength_manifest.json`には参加者、対戦数、XOT列、seed、探索条件、Python・NumPy・TensorFlow版と、実行exe・モデル・重み・XOT・対戦処理に関係するPython source一式のSHA-256を保存する。`strength_runtime_manifest.json`には実際に起動したpolicy backend、CPU/GPU、GPU modelとdriverを保存する。再開時に1項目でも異なれば停止する。identityを検証できない外部policy serverを使うオプションは設けていない。

本番で使う500個のXOTは、重複だけでなく、文字数・座標・着手合法性をengine起動前にすべて再生検査する。数日後に壊れた局面を発見して停止することを避ける。

短い動作・速度確認には、別の出力先で全体タスク数を制限する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --match-sets-per-pair 1 `
  --max-match-sets 64 `
  --status-every-match-sets 16 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/benchmark_64sets
```

## 95%信頼区間の目安

次表は、1組合せ内のセットスコアを`0、0.5、1`の値として扱い、引き分けがない場合の分散を使った計画用の保守的なt近似である。値は対戦相手別95%信頼区間の半幅、単位はパーセントポイントである。引き分けがあれば通常は狭くなる。

| 真のセットスコア | 50セット | 100セット | 200セット | 300セット | 500セット |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | ±14.210 | ±9.921 | ±6.972 | ±5.681 | ±4.393 |
| 60% | ±13.923 | ±9.721 | ±6.831 | ±5.566 | ±4.305 |
| 70% | ±13.024 | ±9.093 | ±6.390 | ±5.207 | ±4.026 |
| 80% | ±11.368 | ±7.937 | ±5.578 | ±4.545 | ±3.515 |
| 90% | ±8.526 | ±5.953 | ±4.183 | ±3.409 | ±2.636 |

500セットでは、最も区間が広い50%付近でも約`±4.39`ポイントになる。50セットの約`±14.21`ポイントと比べると、半幅は約31%になる。標本数を10倍にしても半幅はおおむね`1/sqrt(10)`倍であり、10分の1にはならない。

実出力では、観測されたセットスコアの標本分散を使い、全勝・全敗・全引き分けで区間が不当に0幅にならないようWilson区間との外包を取る。組合せ別の下限・上限・半幅は`strength_pair_results.csv`へ保存する。

この表を参加者全体の7,500セットへそのまま適用してはならない。15組合せは同じ500局面列を共有するため相関があり、参加者summaryとEloには95%信頼区間を出力しない。

## セット数と所要時間

16参加者ではセット数に120組合せを掛ける。

| 1組合せのセット数 | 全対戦セット | 全実対局 | 最新584セット/時での線形目安 | 2回の短縮ベンチによる幅 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 6,000 | 12,000 | 約10.3時間 | 約10.3～11.2時間 |
| 100 | 12,000 | 24,000 | 約20.5時間 | 約20.5～22.4時間 |
| 200 | 24,000 | 48,000 | 約41.1時間 | 約41.1～44.8時間 |
| 300 | 36,000 | 72,000 | 約61.6時間 | 約61.6～67.2時間 |
| 500 | 60,000 | 120,000 | 約102.7時間（約4.3日） | 約4.3～4.7日 |

32論理thread・128 GiB RAMのこのPCで、既定の20並列・64セットを最新コードで実行した短縮ベンチは394.346秒、584.259セット/時、失敗0件だった。CPU使用率は平均48.1%・最大79.7%、GPUは平均6.5%・最大18.0%、system RAMは最大92.9 GiB、子processは最大99個だった。同じ並列設定での直前の再測定は535.982セット/時であり、表の幅はこの2回を線形外挿したものである。

短い64セットでは開始時のmodel/process準備と、最後の四半期でCPU平均が15.4%まで落ちる打切り特有の並列度低下を大きく含む。一方、4日超の本番では他process、温度、再開の影響も受ける。したがって実運用では約4～5日を確保し、最初の500～1,000セットが完了した後は実行中に表示される実測ETAを優先する。

旧実装の非ランダム16参加者換算は50セットあたり約13.2時間だったため、短縮ベンチの範囲では約15～22%短縮した。新実装では、対戦中の石損測定、各セットでのthread生成、全残りtaskの反復走査、終盤に遅い参加者だけを残す単純な投入順を除いた。

## 障害時の扱い

- すべてのGTPコマンドに外側の応答期限を設ける。ネイティブEgaroucidの`genmove`も無期限待機しない。
- processが停止またはtimeoutした場合、途中盤面へコマンドを再送しない。同じXOT局面から先後2局をセット単位で再試行する。
- Windowsでは各top-level engineをkill-on-close Job Objectへ入れ、wrapperが異常終了しても子Egaroucidを残さない。
- 結果は`strength_games.jsonl`へ追記・flushしてから、親threadの集計へ反映する。
- JSONL末尾だけが異常終了で途切れた場合は、安全な最終行まで戻して再開する。途中行の破損、重複task ID、task内容の不一致は拒否する。
- 時間上限または空きメモリ下限に達した場合は新規セットの投入を止め、投入済みの対局を回収してから終了する。

## 主な出力

| ファイル | 内容 |
| --- | --- |
| `strength_manifest.json` | 実験同一性、入力・source SHA-256、task plan |
| `strength_runtime_manifest.json` | 実際のpolicy backend、device、GPU環境 |
| `strength_games.jsonl` | 対戦セット単位の先後2局、石差、完全棋譜 |
| `strength_summary.csv` | 参加者別の記述的セット成績、実対局成績、参考Elo（CIなし） |
| `strength_pair_results.csv` | 対戦相手別のセット成績と95%信頼区間 |
| `strength_paired_set_score_matrix.tsv` | セットスコア行列 |
| `strength_paired_disc_diff_matrix.tsv` | セット平均石差行列 |
| `strength_progress_matrix.tsv` | 実task planに対する組合せ別進捗 |
| `strength_report.txt` | 読みやすい集計表 |
| `strength_results.json` | 機械可読な集計結果と指標定義 |
| `strength_progress.json` | 完了数、残数、終了理由 |
| `performance_samples_<run_id>.csv` | 再開区間ごとのCPU、GPU、メモリ、子process数 |
| `performance_summary_<run_id>.json` | 再開区間ごとの資源集計 |
| `performance_samples.csv` | 全再開区間を結合した時系列 |
| `performance_summary.json` | 全再開区間を結合した資源集計 |
| `policy_batch_server_stats_<run_id>.json` | 再開区間ごとのpolicy batch統計 |

`RESULTS.md`は2026-07-18に旧実装・17参加者・50セットで行った過去結果であり、新しい16参加者・500セット実験とは別の実験である。
