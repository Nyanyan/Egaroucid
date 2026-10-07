# 学習済みモデルの混合率と、着手一致率・棋力の測定

今回学習したDense512×4とConv8＋BNを、それぞれEgaroucidレベル21と混合して評価します。BATを実行すると、着手一致率の測定、棋力の測定、信頼区間の計算、比較グラフの作成が順番に進みます。

## 実行するBAT

| BAT | 対象 |
| --- | --- |
| `run_blend_experiment_all.bat` | Dense512×4、Conv8＋BNの両方 |
| `run_blend_experiment_dense512.bat` | Dense512×4のみ |
| `run_blend_experiment_conv8_bn.bat` | Conv8＋BNのみ |

この文書とBATは、過去の実験用として`legacy`に保存しています。BATは親フォルダのPython処理を呼び出します。以下の相対パスと出力先は、従来どおり親の`50_rating_vs_human_match`フォルダを基準にします。

通常は `run_blend_experiment_all.bat` をダブルクリックしてください。先に両モデルの着手一致率を測定し、その後、各モデルの棋力を順番に測定します。進捗はコンソールとログファイルの両方に出力します。

実験を始めずにファイルと設定を確認する場合は、このディレクトリで次を実行します。

```powershell
.\run_blend_experiment_all.bat --dry-run
```

Pythonは学習時と同じTensorFlow入りの環境を使います。PATH上の `python` が別の環境なら、`POLICY_PYTHON` に実行ファイルのパスを指定してください。スクリプトから呼ぶときは `POLICY_NO_PAUSE=1` を設定すると、終了時のキー入力待ちを省略できます。

## 評価条件

| 項目 | 既定値 |
| --- | --- |
| 混合率α | 0、0.2、0.4、0.6、0.8、1 |
| 混合する探索方策 | Egaroucid for Console 7.8.1 SIMD、レベル21、定石なし |
| 混合式 | `p(NN)^α × softmax(Egaroucid評価値 / 1.0)^(1−α)` |
| 着手一致率 | 保存済みtest集合から30,000局面を抽出、seed=613、top-1とtop-3 |
| データ分割 | 学習時に保存した棋譜単位90% / 5% / 5%の分割を再利用 |
| 棋力の対戦相手 | Egaroucidレベル1、3、5、…、19と、上記6種類の混合方策 |
| 対局数 | 16参加者の総当たり。各組合せ500組、1組につき同じXOT定石で先後交換の2対局 |
| 組の勝敗 | 先後交換した2対局の平均石差の符号で判定 |
| XOTの順序 | seed=57、全組合せで同じ定石順 |
| Eloの再解析 | XOT定石単位のクラスターブートストラップ20,000回、seed=613 |

α=0はEgaroucidのみ、α=1は学習済みモデルのみです。中間のαは、NNのsoftmax前出力とEgaroucidの評価値を重み付きで加えてから正規化します。これにより、小さい確率が浮動小数点の丸めで0になっても、混合後に有力になる手を失いません。着手一致率と棋力測定で、この計算をそろえています。

着手一致率では両モデルに同じ局面とEgaroucidの探索結果を使います。レベル21は全合法手の評価値を取得し、レベル1〜19の単体比較は以前と同じ `hint 3` を使います。盤面を変えない回転・鏡映で移り合う着手を同値とする一致率の定義も引き継いでいます。

既定の規模では、**1モデルあたり12万対局、両モデルで24万対局**です。以前と同様、数日単位の実行を想定しています。まず設定だけを見る場合は `--dry-run` を使ってください。

## 使用するファイル

再開時にモデルが自動で切り替わらないよう、次の完了済み学習結果を固定しています。

- Dense512×4: `../10_train_policy_network/trained/wthor_game_split_512x4_e50_20261001_165329_704`
- Conv8＋BN: `../10_train_policy_network/trained/wthor_game_split_conv8_bn_e50_20261001_170120_867`

各ディレクトリの `selected_model.h5` と `dataset_split.json` を使います。Conv8＋BNではDense専用のバイナリ重みは不要です。別の学習結果を使う場合は、`--dense512-run` または `--conv8-bn-run` と、新しい `--output-root` を指定してください。

盤面データの既定の場所は `%EGAROUCID_DATA%/train_data/board_data/records1` です。棋譜データと分割ファイルのチェックサムを照合してから評価します。

エンジンとXOT定石集は、まずこのリポジトリの `bin` を参照します。存在しない場合は、隣の `Egaroucid` リポジトリを参照します。各リポジトリのルートからの相対パスは次のとおりです。

- `bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/Egaroucid_for_Console_7_8_1_SIMD.exe`
- `bin/problem/xot/openingslarge.txt`

別の場所にある場合は `--egaroucid-exe` と `--openings` で指定できます。

## 出力と再開

既定の出力先は `output/game_split_dense_conv8_bn_l21` です。`dense512` と `conv8_bn` の各ディレクトリに、`human`、`strength`、工程別ログを保存します。共通の探索結果は `shared_human_hints` に保存します。

全工程が終わると、出力先直下に次の比較結果ができます。

- `rating_vs_human_match_dense512_conv8_bn.csv`: モデル・混合率ごとのElo、top-1・top-3一致率と信頼区間
- `rating_vs_human_match_dense512_conv8_bn.png` / `.svg`: 横軸Elo、縦軸top-1一致率のグラフ
- `rating_vs_human_match_dense512_conv8_bn.json`: グラフの集計条件

モデル単独のBATでは、ファイル名にそのモデル名だけが入ります。両モデルの個別実行が終わった後に比較グラフを作るには、次を実行してください。

```powershell
.\run_blend_experiment_all.bat --stage report
```

途中で中断した場合は、**同じBATを同じ引数で再実行**してください。完了済みの工程を確認し、棋力測定では保存済みの対局から再開します。着手一致率の探索は完了済みレベルのキャッシュを再利用し、未完了レベルはそのレベルの探索をやり直します。モデル、データ、実験設定、実行コードが変わった場合は再開を拒否するため、その場合は別の出力先を指定します。同じ出力先への同時実行も防いでいます。

`--stage human` で着手一致率だけ、`--stage strength` で棋力だけを実行することもできます。`--stage report` は両方の測定が完了してから使います。

## グラフの読み方

DenseとConv8＋BNでは別々の総当たりを行うため、参加者全体の平均を1500にしたEloをそのまま重ねると基準がずれます。比較グラフでは、共通のEgaroucidレベル1、3、…、19の平均が1500になるように、それぞれのEloを平行移動します。信頼区間もブートストラップの各反復に同じ処理を行い、2.5%点と97.5%点から計算します。元の総当たり内のEloとBCa信頼区間も、各 `strength` ディレクトリに残します。

この比較は共通の対戦相手を基準にした記述的な比較です。両モデルを同じ総当たりで直接対戦させた結果や、有意差の検定ではありません。着手一致率の区間は以前と同じ局面単位のWilson区間であり、同じ棋譜内の局面間の相関は補正していません。

旧実験の保存済みCSVは、今回の集計には混ぜません。今回は棋譜単位のtest分割と再学習したモデルを使用し、混合計算の数値誤差への対処も加わっているためです。

## 実行規模の変更

既定の棋力測定は、エンジン1プロセスにつき1スレッド、単体参加者につき2プロセス、混合参加者につき8プロセスです。同時対局は最大17組・34対局となります。起動時のメモリ見積もりは79,200 MiBで、別に16,384 MiBの空きを確保します。共有のGPU推論サーバーにはTensorFlowを使用します。

規模を小さくする場合は、例えば次のように、別の出力先を指定します。

```powershell
.\run_blend_experiment_all.bat --positions 3000 --match-sets-per-pair 50 --parallel-match-sets 8 --blend-processes-per-player 4 --bootstrap-replicates 2000 --output-root output\small_comparison
```

同じ結果に異なる条件が混ざらないよう、再開するときも同じ引数を指定してください。
