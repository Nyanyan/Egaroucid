# ブレンド方策の対戦強度評価

関連issue: #613

Egaroucid for Console、Edax 4.6、Policy NetworkをEgaroucidの方策へ混合した各αを、XOT局面から総当たりで対戦させる。既定ではEgaroucidをlevel 1、5、9、13、17、Edaxをlevel 1、5、9、13、17、21、ブレンド方策をα=0.0から1.0まで0.2刻みで参加させる。合計17参加者である。

**各組のマッチ数は`--match-sets-per-pair N`で指定する。既定値は500。1マッチ（1対戦セット）は先後を入れ替えた2局なので、`--match-sets-per-pair 100`なら各組100マッチ・200局になる。** 実行例は[各組のマッチ数を指定する](#各組のマッチ数を指定する)を参照。

## 対戦セットと統計単位

1対戦セットでは、同じXOT局面から同じ2参加者が色を入れ替えて2局行う。参加者0から見た2局の石差を平均し、その符号でセットの勝ち・引き分け・負けを決める。

```text
セット平均石差 = (黒を持った局の石差 + 白を持った局の石差) / 2
セットスコア = (セット勝ち数 + 0.5 × セット引き分け数) / セット数
```

95%信頼区間の標本数は実対局数ではなく対戦セット数である。同じXOT局面を使う先後2局には対応があるため、500セットを1,000個の独立対局として扱わない。

実対局単位の勝敗も別列へ保存するが、これは記述統計である。対戦相手別のセットスコアと95%信頼区間、およびセット平均石差を主要指標とする。

全組合せには同じset indexで同じXOT局面を割り当てる。そのため、ある参加者の16相手分を単純に8,000個の独立標本としてまとめることはできない。参加者全体のスコアは記述値とする。Eloは0%・100%で任意の無限大や固定clip値にならないよう、対戦済みの各組合せへ0.5勝・0.5敗を加えた有限な相対値である。未対戦の組合せは推定の重みを0とし、対戦済み組合せのグラフが全参加者を連結するまでは途中Eloを表示しない。

最終Eloの95%信頼区間は、完了後に`strength_games.jsonl`を再解析して求める。set index、すなわちXOT局面をクラスタとし、同じ復元抽出列を全組合せへ適用するBCaクラスターブートストラップを使う。これにより、同じXOTを共有した組間の相関を保つ。2色の実対局や各組合せを別々に再標本化してはならない。

## 本番設定

| 項目 | 既定値 |
| --- | ---: |
| Egaroucid単体 | level 1、5、9、13、17（5参加者） |
| ブレンド方策 | α=0.0、0.2、0.4、0.6、0.8、1.0（6参加者） |
| Edax 4.6 | level 1、5、9、13、17、21（6参加者） |
| 参加者数 | 17 |
| 組合せ数 | 136 |
| 1組合せ | 500対戦セット、1,000実対局 |
| 全体 | 68,000対戦セット、136,000実対局 |
| 各参加者 | 8,000対戦セット、16,000実対局 |
| XOT | `bin/problem/xot/openingslarge.txt` |
| XOT shuffle seed | 57 |
| XOTの割当て | 各組合せへ同じ500局面列を割り当てる |
| 同時対戦セット数 | 最大20 |
| 同時実対局数 | 最大40 |
| 実対局worker thread | 40 |
| Egaroucid単体のprocess/参加者 | 2 |
| Edaxのprocess/参加者 | 2 |
| ブレンド系列のprocess/参加者 | 10 |
| 1 engine processの探索thread | 1 |
| Policy Network推論 | 1個の共有TensorFlow/GPU server |
| Egaroucid hint cache | 実験出力先ごとのSQLite |
| 空きメモリ下限 | 16,384 MiB |
| 想定マシン | 32論理thread、128 GiB RAM |

同時対局数を45局にする場合は、`--parallel-games 45 --parallel-match-sets 23 --blend-processes-per-player 12` を指定する。23セット分の先後2局を登録し、実際に同時実行する対局は45局までに制限する。

`--finish-low-levels-first` は、単体エンジンのlevel 1・5・9を低い順に優先する。全対戦が終わった参加者は子プロセスごと終了し、そのメモリ予算を重いAIへ回す。`--extra-engine-memory-mib 20480` では、初期設定に加えて最大20 GiBまでプロセス数を増やせる。実際の追加量は空きメモリでも制限し、未起動のプロセスに必要なメモリも予約する。現在の割り当てと完了状態は出力先の `engine_allocation_live.json` に保存する。

重いlevel 21探索を使うブレンド系列へ多くのprocessを割り当てる。Egaroucid単体5参加者とEdax 6参加者は各1セット、ブレンド系列6参加者は各5セットまで同時参加でき、参加者容量から見た上限は20セットである。共有cacheやpolicy serverを待つ実対局があるため、32論理threadに対して40実対局を投入して待ち時間を隠す。探索条件を旧実験と揃えるため、1 process内の探索thread数は1のままである。

全実対局を1個の固定thread poolへ投入し、親threadが先後2局を結合する。136組合せ別のqueueを使い、実行できる対戦の中で古いset indexを優先する。遅い対戦が残っていても、空いているAI同士は先の開始局面へ進める。組合せ間の先行数に上限は設けないため、途中の完了件数は組合せごとに異なる。全組合せで共通のXOT開始局面と最終的な対戦数は維持する。同じset indexでは残り推定仕事量を使って遅い参加者を優先し、初回の起動時間に左右されないよう推定時間を中央値の0.5倍から2倍へ制限する。

hintは実験出力先の共有SQLiteだけへ保存する。各Python process内の無制限dict cacheは5日規模の実験でメモリを増やし続けるため使用しない。`--no-hint-cache`を指定した場合はSQLite自体も作成・参照しない。

既定値の最悪時見積りは、重いEgaroucid process 60個、Edax process 12個、Python wrapper 50個、共有policy serverを合わせて82,672 MiBである。さらに16,384 MiBを空きとして残すため、開始時点で約96.7 GiB以上のavailable memoryを要求する。これは128 GiB機向けの設定であり、64 GiB・96 GiB機では並列数と参加者あたりのprocess数を減らす必要がある。起動後も親processと各wrapperが空きメモリを確認し、下限を割る場合は新しいprocess・セットの開始を止める。

## 必要ファイル

リポジトリのルートをカレントディレクトリとし、次のファイルが存在することを確認する。

```text
bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe
bin/versions/edax_4_6/wEdax-x86-64.exe
bin/versions/edax_4_6/data/eval.dat
bin/problem/xot/openingslarge.txt
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_model.h5
src/tools/policy_network_human_like_ai/10_train_policy_network/trained/wthor_final_arch_512x4_e50/selected_policy_network_weights.bin
```

Python環境には`numpy`、`psutil`、`tensorflow`が必要である。既定値ではTensorFlow/GPUを要求し、GPUが見つからない場合は開始前に停止する。暗黙にNumPyやCPUへ切り替えない。意図的な確認実行だけは`--allow-policy-cpu`または`--policy-backend numpy`を明示する。

## 実行方法

以下の実行例は`Egaroucid`リポジトリのルートから実行する。

### 各組のマッチ数を指定する

`--match-sets-per-pair N`は、すべての対戦相手の組合せに共通するマッチ数を指定する。`N`には1以上の整数を渡す。1マッチでは同じXOT局面から先後を入れ替えて2局行うため、各組の実対局数は`2 × N`となる。

| 指定例 | 各組のマッチ数（対戦セット数） | 各組の実対局数 |
| --- | ---: | ---: |
| `--match-sets-per-pair 50` | 50 | 100局 |
| `--match-sets-per-pair 100` | 100 | 200局 |
| `--match-sets-per-pair 250` | 250 | 500局 |
| `--match-sets-per-pair 500`（省略時の既定値） | 500 | 1,000局 |
| `--match-sets-per-pair 1000` | 1,000 | 2,000局 |

例えば、既定の17参加者で各組100マッチにする場合は次のように実行する。136組合せがあるため、全体では13,600マッチ・27,200局になる。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --match-sets-per-pair 100 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_100sets_17players_edax `
  --dry-run
```

設定を確認したら`--dry-run`を外して対戦を開始する。XOTファイルには`N`個以上の異なる合法な棋譜が必要であり、既定のファイル以外を使う場合は`--openings`で指定する。再開時は同じ`N`・出力先・その他の条件で`--resume`を付ける。`N`を変更する場合は新しい出力先を指定する。

`--max-match-sets`は大会全体のマッチ数を途中で打ち切る上限、`--parallel-match-sets`は同時に実行するマッチ数である。各組に指定数のマッチを行わせる場合は、`--match-sets-per-pair`を指定し、`--max-match-sets`は省略する。

### Edax 4.6の設定を変更する

Edaxは次の配置を既定で参照する。評価データは絶対パスで渡すため、Edaxのフォルダへ移動して起動する必要はない。

```text
bin/versions/edax_4_6/wEdax-x86-64.exe
bin/versions/edax_4_6/data/eval.dat
```

既定のEdax 6レベルを、level 1、3、5の3レベルへ変更する例を示す。以下のコマンドは`Egaroucid`リポジトリのルートから実行する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --edax-levels 1,3,5 `
  --match-sets-per-pair 500 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_edax_1_3_5 `
  --dry-run
```

この例では14参加者、91組合せ、45,500対戦セットになる。確認後に`--dry-run`を外すと対戦を開始する。既定の構成とは実験条件が異なるため、新しい出力先を指定する。

Egaroucid level 1とEdax level 1だけで、先後2局を確認する場合は次のように指定する。Policy Networkを使わないため、この実行には学習済みモデルやTensorFlowは不要である。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --baseline-levels 1 --alphas= --edax-levels 1 `
  --match-sets-per-pair 1 --parallel-match-sets 1 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/edax_smoke
```

どちらの例もXOTファイルが必要である。既定の`bin/problem/xot/openingslarge.txt`がない場合は、`--openings`で手元のXOTファイルを指定する。ファイルは1行に1局面分の合法な棋譜を記載する形式で、指定セット数以上の異なる棋譜が必要になる。

| オプション | 用途・既定値 |
| --- | --- |
| `--edax-levels` | 0〜60のlevelをカンマ区切りで指定。既定は`1,5,9,13,17,21`。`--edax-levels=`で不参加にできる |
| `--edax-exe` | 実行exeを変更。既定は上記の`wEdax-x86-64.exe` |
| `--edax-eval-file` | 評価データを変更。既定は指定exeと同じフォルダ内の`data/eval.dat` |
| `--edax-processes-per-player` | 各Edax levelのprocess数。既定は2、2以上の偶数 |
| `--engine-threads` | EgaroucidとEdax共通の探索並列数。既定は1 |
| `--estimated-edax-memory-mib` | Edax 1 processのメモリ見積り。既定は256 MiB |

Edaxは`edax_l1`などの名前で結果表・Elo集計に参加する。bookとponderは無効、hash tableは22 bitで固定し、`--engine-threads`をEdaxの`-n`へ渡す。これらの起動オプションは[Edax 4.6の公式実装](https://github.com/abulmo/edax-reversi/blob/v4.6/src/options.c)に合わせている。EdaxのlevelはEdax自身の探索設定であり、同じ数字のEgaroucid levelと同じ強さを保証するものではない。

Edax同士で比較する場合は、`--baseline-levels= --alphas= --edax-levels 1,3`のように指定できる。参加者の合計は2以上必要である。再開には同じ引数と`--resume`を指定し、exeや評価データを変更した場合は新しい出力先を使う。

### 既定の17参加者で実行する

αの6参加者、Egaroucidの5参加者、Edaxの6参加者を指定なしで登録する。旧構成の16参加者の結果と混同しないよう、既定の出力先を`xot_500sets_17players_edax`へ変更している。

最初に、ファイル、実験数、並列数、メモリ見積りを確認する。engineやpolicy serverは起動しない。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --match-sets-per-pair 500 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_17players_edax `
  --dry-run
```

本番を開始する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --match-sets-per-pair 500 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_17players_edax
```

中断後は同じ引数と出力先で`--resume`を付ける。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/run_strength_full.py `
  --match-sets-per-pair 500 `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_17players_edax `
  --resume
```

`strength_manifest.json`には参加者、対戦数、XOT列、seed、探索条件、Python・NumPy版、インストール済みのTensorFlow関連distribution版と、実行exe・モデル・重み・XOT・対戦処理に関係するPython source一式のSHA-256を保存する。Edaxが参加する場合は、その実行exeと評価データのSHA-256も保存する。`strength_runtime_manifest.json`には実際に起動したpolicy backend、実際にimportされたTensorFlow版、CPU/GPU、GPU modelとdriverを保存する。再開時に1項目でも異なれば停止する。identityを検証できない外部policy serverを使うオプションは設けていない。

対戦が完了したらElo 95%信頼区間を生成する。既定では20,000反復、PCG64、seed 613を使う。元の棋譜、manifest、既存集計は変更せず、補助CSVとJSONだけを出力する。

```powershell
python src/tools/policy_network_human_like_ai/40_test_strength/reanalyze_strength_elo.py `
  --output-dir src/tools/policy_network_human_like_ai/40_test_strength/output/xot_500sets_17players_edax `
  --bootstrap-replicates 20000 `
  --bootstrap-seed 613 `
  --workers 16
```

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

既定の500セットは、1組合せあたり先後計1,000局である。このとき、対戦相手別のセットスコアの95%信頼区間は、50%付近・セット引き分けなしの計画値で**半幅約±4.39ポイント、全幅約8.79ポイント**になる。ここで全幅は「上限−下限」、半幅は「±」で表す片側の長さである。

セットスコアを勝ち1、引き分け0.5、負け0として扱う。平均のt区間は`平均 ± t × 標準偏差 / √N`であり、標準誤差は標本数の平方根に反比例する。[NISTの平均の信頼区間](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)にある式を用い、本スクリプトの`conservative_score_half_width`で次表を計算した。`N`は独立なXOT局面のセット数とし、計画上の分散を最大の`0.5 × 0.5 = 0.25`に置いている。

| 1組合せのセット数N | 1組合せの実対局 | 95%区間の半幅 | 95%区間の全幅 | 17参加者の全実対局 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 100 | ±14.21ポイント | 28.42ポイント | 13,600 |
| 100 | 200 | ±9.92ポイント | 19.84ポイント | 27,200 |
| 200 | 400 | ±6.97ポイント | 13.94ポイント | 54,400 |
| 300 | 600 | ±5.68ポイント | 11.36ポイント | 81,600 |
| 500（既定） | 1,000 | ±4.39ポイント | 8.79ポイント | 136,000 |
| 1,000 | 2,000 | ±3.10ポイント | 6.21ポイント | 272,000 |
| 2,000 | 4,000 | ±2.19ポイント | 4.39ポイント | 544,000 |
| 5,000 | 10,000 | ±1.39ポイント | 2.77ポイント | 1,360,000 |
| 10,000 | 20,000 | ±0.98ポイント | 1.96ポイント | 2,720,000 |

幅を半分にするには、おおむね4倍のセット数が必要になる。全幅10ポイント以内なら約400セット、5ポイント以内なら約1,600セット、2ポイント以内なら約10,000セットが計画上の目安である。まずは既定の500セットで比較し、より狭い区間が必要なら2,000セットを候補にできる。これらは期待する区間幅の目安であり、特定の差を検出できる確率を保証するものではない。

セット数は`--match-sets-per-pair 2000`のように変更する。その場合は、少なくとも2,000個の異なるXOT棋譜と、新しい`--output-dir`を用意する。17参加者では136組合せなので、全実対局数は`272 × N`となる。`--dry-run`の`strength_dry_run.json`にも表の半幅と全幅を保存する。JSONの値は0〜1のスコア単位であり、100倍すると表のパーセントポイントになる。

実出力では、観測されたセットスコアの標本分散を使い、全勝・全敗・全引き分けで区間が0幅にならないよう、t区間とWilson区間の両方を含む範囲を採用する。[NISTのWilson区間の説明](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)も参照。引き分けを0.5としてWilson式へ入れる部分は本実装の補助的な処理であり、一般の三値分布で厳密に95%の被覆率を保証する手法ではない。表も計画用の近似であり、実際の幅は結果によって変わる。例えば500セットが250勝0分250敗なら、実装が出す全幅は約8.80ポイントになる。組合せ別の下限・上限・半幅は`strength_pair_results.csv`へ保存する。

Eloの95%信頼区間の長さは、セット数だけでは決まらず、相手との強さの差や全組合せの結果に依存する。また、16組合せは同じXOT局面列を共有するため相関があり、既定の参加者あたり8,000セットをすべて独立な観測としてこの表へ当てはめることもできない。最終Eloには、set index/XOT局面をクラスタとして全組合せを同時に20,000回再標本化し、各反復で17参加者平均を1500へ中心化したBCa 95%信頼区間を別途出力する。Eloでの全幅は`strength_elo_ci95.csv`の上限と下限の差で確認する。

この区間は参加者ごとのpointwise区間であり、17参加者すべてを同時に95%で覆う区間ではない。対象は同じXOT母集団から別の局面列を抽出した場合の変動であり、モデル再学習、探索条件変更、別のopening corpusの不確実性は含まない。2参加者の差は、個別区間の重なりではなく、保存された同一bootstrap反復内のElo差で評価する。

## セット数と所要時間（旧参加者構成での参考値）

16参加者ではセット数に120組合せを掛ける。以下の時間は、Edaxを含まない旧構成（Egaroucid 10参加者＋αの6参加者）で測定した値である。現在の17参加者の構成では総対局数と探索時間が変わるため、所要時間は改めて実測する必要がある。

| 1組合せのセット数 | 全対戦セット | 全実対局 | 最新584セット/時での線形目安 | 2回の短縮ベンチによる幅 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 6,000 | 12,000 | 約10.3時間 | 約10.3～11.2時間 |
| 100 | 12,000 | 24,000 | 約20.5時間 | 約20.5～22.4時間 |
| 200 | 24,000 | 48,000 | 約41.1時間 | 約41.1～44.8時間 |
| 300 | 36,000 | 72,000 | 約61.6時間 | 約61.6～67.2時間 |
| 500 | 60,000 | 120,000 | 約102.7時間（約4.3日） | 約4.3～4.7日 |

32論理thread・128 GiB RAMのこのPCで、旧参加者構成を20並列・64セットで実行した短縮ベンチは394.346秒、584.259セット/時、失敗0件だった。CPU使用率は平均48.1%・最大79.7%、GPUは平均6.5%・最大18.0%、system RAMは最大92.9 GiB、子processは最大99個だった。同じ並列設定での直前の再測定は535.982セット/時であり、表の幅はこの2回を線形外挿したものである。

短い64セットでは開始時のmodel/process準備と、最後の四半期でCPU平均が15.4%まで落ちる打切り特有の並列度低下を大きく含む。また、長時間の本番では他process、温度、再開の影響も受ける。現在の構成では、最初の500～1,000セットが完了した後に表示される実測ETAを使って所要時間を判断する。

旧実装の非ランダム16参加者換算は50セットあたり約13.2時間だったため、短縮ベンチの範囲では約15～22%短縮した。新実装では、対戦中の石損測定、各セットでのthread生成、全残りtaskの反復走査、終盤に遅い参加者だけを残す単純な投入順を除いた。

## 障害時の扱い

- すべてのGTPコマンドに外側の応答期限を設ける。ネイティブEgaroucidの`genmove`も無期限待機しない。
- processが停止またはtimeoutした場合、途中盤面へコマンドを再送しない。同じXOT局面から先後2局をセット単位で再試行する。
- Windowsでは各top-level engineをkill-on-close Job Objectへ入れ、wrapperが異常終了しても子Egaroucidを残さない。
- 結果は`strength_games.jsonl`へ追記・flushしてから、親threadの集計へ反映する。
- JSONL末尾だけが異常終了で途切れた場合は、安全な最終行まで戻して再開する。途中行の破損、重複task ID、task内容の不一致は拒否する。
- 時間上限または空きメモリ下限に達した場合は新規セットの投入を止め、投入済みの対局を回収してから終了する。

### 対局用AIと追加スレッドを合計70枠に配分する

`--adaptive-engine-threads --engine-threads 1 --engine-thread-budget 70`を指定すると、まず1スレッドのエンジンで同時対局を増やす。各対局の双方を同時に予約し、先後を入れ替えた2局で最低4枠を確保する。残り対局数、メモリ、`--parallel-games`の上限でこれ以上増やせなくなったら、余った枠を探索スレッドへ配分する。たとえば70枠なら17セット・34局の双方に68枠を使い、残り2枠を探索へ配分できる。CPU使用率による増減は行わない。

合計には探索エンジンのスレッド数に加え、α=1、幾何平均モデル、ランダムAIも対局ごとに1枠ずつ数える。探索を呼び出すwrapperや共有推論サーバーなど、補助プロセスは別途動く。参加者ごとの実際の割当は、残り対局と相手側の予約から決まり、35などの固定上限は設けない。全対戦が完了し、実行中対局もないAIは0枠とし、待機プロセスも終了する。

メモリは保持中のエンジンと開始予定の対局に必要な分を見積もる。空きメモリ下限と2 GiBの余裕を残し、増設を妨げる待機中エンジンは終了する。この方式では、未使用プロセス枠のメモリ予約や`--extra-engine-memory-mib`による増設上限は使わない。

**Edaxは着手間でスレッド数を増減し、同じプロセスとTTを維持する。** 対局開始時からXBoard接続を使い、`cores`命令の完了を`ping`で確認する。減らした枠は変更完了後に返却し、新規対局の予約と競合しても70枠を超えない。実行中の探索を中断して変更することはない。[Edax 4.6のスレッド変更処理](https://github.com/abulmo/edax-reversi/blob/v4.6/src/search.c#L789)はworkerだけを変更し、TTを消去しない。XBoardの`new`がlevelを60に戻すため、新局では指定levelを設定し直す。

Egaroucidとそのブレンドは対局中の割当を維持し、設定変更が必要なときだけ次局開始時に再起動する。ブレンドのwrapperも対局中は保持する。対応エンジンの次の着手を待つ間や、再配分できる対局がない場合は70枠を下回る。`--max-engine-threads`で1プロセスの上限を指定でき、Edaxの実行中変更は最大64スレッドまで対応する。

新規対局は、予定対局数に対する完了数が少ないモデルを優先する。実行中の対局も開始済みとして進捗に加えるため、遅れを埋める分のプロセスを割り当てた後は、他のモデルにも枠が回る。双方のモデルが遅れている対戦を優先し、相手のプロセスやメモリを確保できない場合は、開始できる別の対戦へ進む。各ペアでは元の開始局面の順序を保ち、この優先順位は再開前に完了した対局も含めて計算する。

`engine_allocation_live.json`の`allocated_compute_slots`が全AIを含む予約合計で、`allocated_player_processes`が対局双方のプロセス数、`allocated_search_threads`が探索スレッドの合計を示す。各参加者の`allocated_game_threads`は対局ごとの探索割当、`active_search_threads`は起動済みの探索スレッド合計である。`live_thread_changes`でEdaxの着手間変更回数を確認できる。メモリ見積もりには対局後に待機しているエンジンの追加スレッドも含める。

同じファイルの`completed_sets`、`target_sets`、`active_sets`には、各モデルの完了組数、予定組数、実行中の組数を記録する。1組は先後を入れ替えた2対局で、双方が終わった時点で完了数へ加える。

## 主な出力

| ファイル | 内容 |
| --- | --- |
| `strength_manifest.json` | 実験同一性、入力・source SHA-256、task plan |
| `strength_runtime_manifest.json` | 実際のpolicy backendとその版、device、GPU環境 |
| `strength_games.jsonl` | 対戦セット単位の先後2局、石差、完全棋譜 |
| `strength_summary.csv` | 参加者別の記述的セット成績、実対局成績、Elo点推定 |
| `strength_pair_results.csv` | 対戦相手別のセット成績と95%信頼区間 |
| `strength_paired_set_score_matrix.tsv` | セットスコア行列 |
| `strength_paired_disc_diff_matrix.tsv` | セット平均石差行列 |
| `strength_progress_matrix.tsv` | 実task planに対する組合せ別進捗 |
| `strength_report.txt` | 読みやすい集計表 |
| `strength_results.json` | 機械可読な集計結果と指標定義 |
| `strength_elo_ci95.csv` | Elo点推定とXOTクラスタBCa 95%信頼区間 |
| `strength_elo_ci95.json` | Elo区間の手法、seed、補正値、由来、結果 |
| `strength_elo_bootstrap_replicates.csv` | 各bootstrap反復の全参加者Elo |
| `strength_elo_bootstrap_cluster_indices.csv` | 各反復で抽出したset index列 |
| `strength_progress.json` | 完了数、残数、終了理由 |
| `performance_samples_<run_id>.csv` | 再開区間ごとのCPU、GPU、メモリ、子process数 |
| `performance_summary_<run_id>.json` | 再開区間ごとの資源集計 |
| `performance_samples.csv` | 全再開区間を結合した時系列 |
| `performance_summary.json` | 全再開区間を結合した資源集計 |
| `policy_batch_server_stats_<run_id>.json` | 再開区間ごとのpolicy batch統計 |

`RESULTS.md`の先頭には2026-07-21に当時の構成・16参加者・50セットで完走した本番結果を記録している。後半の2026-07-18節は旧実装・17参加者の過去結果である。どちらも現在のEdaxを含む構成とは別の実験である。
