# オセロクエストの弱いプレイヤーを模倣するルール方策

人が決めたオセロのルールに重みを付け、オセロクエストのレート **[800,1000)** の着手から、その重みを学習する実験です。隅・C打ち・X打ち・返す石数・合法手数などを **43個の特徴量** にし、序盤・中盤・終盤の3組、計129個の係数を学習します。

このフォルダーには、ルールの定義、学習、着手確率の出力、人間との着手一致率の測定、Egaroucidレベル21による平均石損の測定をまとめています。学習済みの重みは [trained_policy.json](trained_policy.json)、実測結果は [results_summary.json](results_summary.json) にあります。

- [まず実行する](#まず実行する)
- [どうやって手を選ぶか](#どうやって手を選ぶか)
- [使用する43個の特徴量](#使用する43個の特徴量)
- [学習した重みをどう読むか](#学習した重みをどう読むか)
- [学習方法とデータ分割](#学習方法とデータ分割)
- [学習と評価の実行方法](#学習と評価の実行方法)
- [評価指標と今回の結果](#評価指標と今回の結果)
- [ファイルと出力の対応](#ファイルと出力の対応)
- [ルールを変更して再実験する](#ルールを変更して再実験する)

## まず実行する

以下のコマンドは、Egaroucidリポジトリのルートから、このフォルダーへ移動して実行します。長い学習・石損計算は、進捗を確認できるPowerShellのコンソールで実行してください。

```powershell
cd src/tools/policy_network_human_like_ai/80_rule_base_AI
```

学習済み方策を使うだけなら、Python、NumPy、Numbaが必要です。次の例は初期局面について、全合法手の確率と、ルールごとの加点・減点をJSONで表示します。

```powershell
python predict.py --player 0x0000000810000000 --opponent 0x0000001008000000
```

この例では `e6`、`f5`、`c4`、`d3` がそれぞれ確率0.25になります。出力の `probability` は着手確率、`logit` はsoftmax前の合計得点、`contributions` はその得点を構成する各特徴量の寄与です。`phase` は序盤・中盤・終盤の重みの混合比です。

`--player` には**手番側**の石、`--opponent` には相手側の石を、64ビットのビットボードで渡します。色の黒白は固定しません。ビット番号は `a1=63`、`b1=62`、`a2=55`、`h8=0` です。別のモデルは `--model <モデルのJSON>` で指定します。

Pythonからも利用できます。

```python
from rule_policy import RulePolicy

model = RulePolicy.load("trained_policy.json")
probabilities = model.predict(0x0000000810000000, 0x0000001008000000)
move = int(probabilities.argmax()) if probabilities.sum() > 0 else None
```

返り値はビット番号順の64要素の配列で、非合法手の確率は0です。合法手がなければ全要素が0になります。その場合は呼び出し側でパスか終局かを判定します。`move` は座標文字列ではなく、上記のビット番号です。

## どうやって手を選ぶか

各合法手について、その手を1回置いた盤面まで計算し、特徴量を求めます。相手の応手を枝分かれさせる探索は行いません。Egaroucidは平均石損の測定に使い、この方策の推論や重みの学習には使いません。

手を選ぶ処理は次の順です。

1. 全合法手を列挙する。
2. 各手について、後述する43個の特徴量 `f_i(s,a)` を計算する。
3. 盤上の石数に応じて、序盤・中盤・終盤の係数を混ぜる。
4. 「特徴量 × 係数」を合計し、各手の得点 `z(s,a)` を求める。
5. 全合法手の得点をsoftmaxで正規化し、着手確率にする。

候補手を置く**前**の盤上の石数を `n` とすると、3組の係数の混合比は次のとおりです。同じ局面の全候補手には、同じ混合比を使います。

```text
t = (n - 4) / 60
g_early  = max(1 - 2t, 0)
g_middle = 1 - |2t - 1|
g_late   = max(2t - 1, 0)

w_i(s) = g_early * w_i,early + g_middle * w_i,middle + g_late * w_i,late
z(s,a) = Σ_i f_i(s,a) * w_i(s)
p(a|s) = exp(z(s,a) - max_b z(s,b)) / Σ_b exp(z(s,b) - max_c z(s,c))
```

softmaxの和と最大値は、すべてその局面の**合法手**について取ります。

| 盤上の石数 | 序盤の係数 | 中盤の係数 | 終盤の係数 |
|---:|---:|---:|---:|
| 4 | 100% | 0% | 0% |
| 19 | 50% | 50% | 0% |
| 34 | 0% | 100% | 0% |
| 49 | 0% | 50% | 50% |

今回の一致率・石損の測定では、**最大確率の1手**を選びます。同点ならビット番号の小さい手を選びます。確率からのランダム抽選は行いません。隅の加点も他の特徴量の得点と合計して比較するため、隅以外を選ぶ場合もあります。

## 使用する43個の特徴量

実装は [rule_policy.py](rule_policy.py) の `FEATURE_NAMES` と `extract()` です。以下の「着手後」は、候補手を置いて石を返した直後、相手が応手する前の盤面を指します。「自分」は手番側、「隣接」は縦・横・斜めの8方向です。個数を定数で割った特徴量も、値が必ず0〜1に収まるという意味ではありません。

### 1. 着手するマスの種類：10個

盤面の回転・鏡映で同じになるマスを、次の10種類に分類します。該当する種類の特徴量が1、残りが0になります。

```text
    a b c d e f g h
 1  K C A B B A C K
 2  C X D E E D X C
 3  A D F G G F D A
 4  B E G H H G E B
 5  B E G H H G E B
 6  A D F G G F D A
 7  C X D E E D X C
 8  K C A B B A C K
```

| 図の記号 | 特徴量名 | マスの例・意味 |
|---|---|---|
| K | `square_corner` | a1などの隅 |
| C | `square_C` | b1・a2など、隅に辺方向で隣接するCマス |
| A | `square_edge_A` | c1・a3など、隅から辺方向へ2マス離れたAマス |
| B | `square_edge_B` | d1・a4など、辺中央寄りのBマス |
| X | `square_X` | b2など、隅に斜めで隣接するXマス |
| D | `square_inner_12` | c2・b3など |
| E | `square_inner_13` | d2・b4など |
| F | `square_inner_22` | c3など |
| G | `square_inner_23` | d3・c4など |
| H | `square_center` | d4・e4・d5・e5の中央4マス |

通常の初期盤面では中央4マスが最初から埋まっているため、`square_center` は合法手の特徴としては使われません。分類を10種類にそろえるために残してあり、その3係数は今回の学習済みモデルでも0です。

### 2. C打ち・X打ちと隅の所有状態：6個

Cマス・Xマスに打つ場合、着手前の盤面で、そのマスに対応する隅がどうなっているかを分けます。例えばb1・a2・b2に対応する隅はa1です。これらの特徴量は、上の `square_C` / `square_X` に**追加して**得点へ入ります。

| 特徴量名 | 1になる条件 |
|---|---|
| `C_corner_empty` | C打ちで、対応する隅が空き |
| `C_corner_own` | C打ちで、対応する隅が自分の石 |
| `C_corner_opponent` | C打ちで、対応する隅が相手の石 |
| `X_corner_empty` | X打ちで、対応する隅が空き |
| `X_corner_own` | X打ちで、対応する隅が自分の石 |
| `X_corner_opponent` | X打ちで、対応する隅が相手の石 |

C・X以外の手では、この6個はすべて0です。これにより、例えば「隅が空いているX打ちは避けるが、自分が隅を持っていれば打ちやすい」という傾向を学習できます。

### 3. 返す石数・返し方：14個

返す相手石の数を `k` とします。新しく置く1石は `k` に含めません。

| 特徴量名 | 値 |
|---|---|
| `flips_1` | `k=1` なら1、ほかは0 |
| `flips_2` | `k=2` なら1、ほかは0 |
| `flips_3` | `k=3` なら1、ほかは0 |
| `flips_4` | `k=4` なら1、ほかは0 |
| `flips_5` | `k=5` なら1、ほかは0 |
| `flips_6` | `k=6` なら1、ほかは0 |
| `flips_7` | `k=7` なら1、ほかは0 |
| `flips_8_plus` | `k>=8` なら1、ほかは0 |
| `flips_scaled` | `k / 18` |
| `flips_squared` | `k² / 324` |
| `most_flips` | 全合法手の中で返す石数が最大なら1 |
| `fewest_flips` | 全合法手の中で返す石数が最小なら1 |
| `flip_directions` | 石を返す方向数 / 8 |
| `longest_flip_ray` | 1方向で連続して返す石数の最大値 / 6 |

最大・最小の同率はすべて1です。全候補の返す枚数が同じなら、`most_flips` と `fewest_flips` は両方1になります。

枚数別の8個によって「2〜3枚くらいを好む」といった単調でない傾向を表し、`flips_scaled` と `flips_squared` で枚数全体の傾向を補います。また、「絶対枚数が多い」と「いま打てる手の中では最多取り」を分けて表しています。

### 4. 合法手数・隅を渡すか・パス：4個

| 特徴量名 | 値 |
|---|---|
| `opponent_mobility` | 着手後の相手の合法手数 / 32 |
| `own_mobility` | 着手後の盤面で、自分に仮にもう一度手番が来た場合の合法手数 / 32 |
| `opponent_corner_moves` | 着手後に相手が合法手として打てる隅の数 / 4 |
| `force_pass` | 着手後の相手の合法手が0なら1。終局の場合も1 |

`own_mobility` は、相手の応手を読んだ後の合法手数ではありません。相手が応手する前の盤面で計算します。

### 5. 着手マスの周囲：3個

着手前の盤面について数えます。盤外のマスは数えません。

| 特徴量名 | 値 |
|---|---|
| `move_adjacent_empty` | 着手マスに隣接する空きマス数 / 8 |
| `move_adjacent_own` | 着手マスに隣接する自分の石数 / 8 |
| `move_adjacent_opponent` | 着手マスに隣接する相手の石数 / 8 |

### 6. 石の露出・辺・潜在的な合法手数の変化：6個

ここでは、各個数の **着手後 − 着手前** を使います。空きマスに隣接する石を「空きマスに接する石」とし、複数の空きマスに接していても1石として数えます。

| 特徴量名 | 値 |
|---|---|
| `own_frontier_change` | 空きマスに接する自分の石数の増減 / 18 |
| `opponent_frontier_change` | 空きマスに接する相手の石数の増減 / 18 |
| `edge_discs_gained` | 辺上で新たに自分の石になるマス数 / 10。置いた石と返した石を含む |
| `anchored_edge_gain` | 自分が持つ隅から辺沿いに自石が連続する区間について、その石数の増減 / 10 |
| `own_potential_mobility_change` | 相手石に隣接する空きマス数の増減 / 32 |
| `opponent_potential_mobility_change` | 自分の石に隣接する空きマス数の増減 / 32 |

`anchored_edge_gain` は、着手前・着手後のそれぞれの盤面で、その時点で自分が持つ隅を起点に2本の辺をたどって数え、後の個数から前の個数を引きます。同じ石を重複して数えません。例えば新しく隅を取る手では、その隅と、そこから連続する自分の辺の石を着手後の個数に含めます。`potential_mobility` は「相手石に接している空きマス」を候補の多さの目安にする特徴で、実際に石を挟めるかまでは判定しません。

合計は **10 + 6 + 14 + 4 + 3 + 6 = 43個** です。特徴量の種類は手作業で決め、各特徴量を好むか避けるか、その強さと局面の進み具合による違いをデータから学習しています。

## 学習した重みをどう読むか

以下は保存済みモデルの係数の一部です。小数第3位までに丸めています。全43個×3組の値は [trained_policy.json](trained_policy.json) の `features` と、同じ順序の `weights` にあります。

| 特徴量 | 序盤 | 中盤 | 終盤 |
|---|---:|---:|---:|
| 隅 `square_corner` | +0.722 | +1.530 | +1.337 |
| Cマス `square_C` | −0.092 | −0.153 | +0.027 |
| 空き隅に対応するC打ち `C_corner_empty` | −0.748 | −0.741 | −0.579 |
| 自分の隅に対応するC打ち `C_corner_own` | +0.933 | +0.819 | +0.258 |
| Xマス `square_X` | −0.463 | −0.396 | −0.200 |
| 空き隅に対応するX打ち `X_corner_empty` | −1.765 | −1.528 | −0.166 |
| 自分の隅に対応するX打ち `X_corner_own` | +0.456 | +0.926 | +0.058 |
| 最多取り `most_flips` | +0.488 | +0.771 | +0.223 |
| 最少取り `fewest_flips` | +0.715 | +0.193 | −0.078 |
| 相手の合法手数 `opponent_mobility` | −2.468 | −4.670 | −2.704 |
| 相手が打てる隅 `opponent_corner_moves` | −2.746 | −9.556 | −3.239 |

ほかの入力が同じなら、「特徴量 × 係数」の値が大きい手ほど選ばれやすくなります。例えば中盤の係数だけを使う盤上34石の局面では、次のように読みます。

- 空き隅に対応するC打ちの位置に関する得点は、`square_C + C_corner_empty` なので約 **−0.894**。
- 空き隅に対応するX打ちでは、`square_X + X_corner_empty` なので約 **−1.925**。
- 相手に次の隅の合法手を1つ与えると、該当特徴は `1/4` なので約 **−2.389**。
- 相手の合法手を1つ増やすと、該当特徴の寄与は `−4.670/32` なので約 **−0.146**。

これらに、返す石数や辺の石などの得点も加わります。特徴量同士は重なるため、1つの係数だけから最終的な選択を判断せず、`predict.py` の `contributions` で合計への寄与を確認してください。

比較用の「手作業のルール」は [rule_policy.py](rule_policy.py) の `manual_weights()` です。全局面で同じ係数を使います。以下の10個以外の係数は0です。個数の正規化は、学習したルールと同じです。

| 特徴量 | 手作業の係数 |
|---|---:|
| `square_corner` | +4 |
| `C_corner_empty` | −1.5 |
| `X_corner_empty` | −2 |
| `square_edge_A` | +0.5 |
| `square_edge_B` | +0.5 |
| `flips_scaled` | +0.5 |
| `opponent_mobility` | −1 |
| `opponent_corner_moves` | −4 |
| `anchored_edge_gain` | +2 |
| `own_frontier_change` | −0.5 |

**学習時の初期重みはすべて0**であり、この手作業の重みを初期値にしてはいません。

## 学習方法とデータ分割

着手した本人のレートが800以上1000未満の局面を使います。同じ棋譜の黒番と白番のレートは別に判定された既存データを読み込みます。

| 分割 | 局面数 | 用途 |
|---|---:|---|
| train | 560,909 | 重みの最適化 |
| val | 31,087 | L2係数と採用する反復の選択 |
| test | 30,423 | 選択後の着手一致率・平均石損の測定 |

既存の棋譜単位の分割を維持し、分割間に同じ棋譜IDがないこと、ファイルのSHA-256と件数、人間着手が合法であることを検査します。

学習する目的関数は、人間が実際に選んだ手の確率を高める交差エントロピーとL2正則化です。局面は等しい重みで扱い、石損による重み付けは行いません。

```text
loss = mean_train[-log p(人間の着手 | 局面)] + (λ / 2) * Σ_i,phase w_i,phase²
```

SciPyのL-BFGS-Bで係数を更新し、損失と勾配の計算はPyTorchで行います。既定はGPU計算で、CPU側のPyTorchスレッド数は4です。L2係数 `λ=0.0001` と `0.00001` を、それぞれ0初期値から最大120反復試します。各反復の検証データで、後述する対称性を考慮したtop-1一致率が最大の重みを採用し、同率なら人間着手の負の対数尤度が小さい方を選びます。seedは613です。テストは重みや反復の選択に使いません。

回転・鏡映で対応する手の特徴量と確率が同じになるため、元局面とその8通りの変換は、学習目的と勾配が同じです。8回の重複計算を避けて元局面で学習します。val/testも水増ししません。対称性は `test_rules.py` で確認します。

## 学習と評価の実行方法

### 環境と入力データ

実行を確認した環境はWindows、Python 3.9、NumPy 1.24.3、Numba 0.57.1、SciPy 1.11.2、PyTorch 2.7.1+cu118、RTX 3090、psutil 5.9.8です。Windowsのファイルロックを使うため、現在の学習・評価スクリプトはWindows向けです。

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

学習の既定は `--device cuda` です。CPUで学習する場合は、後述の個別実行で `--device cpu` を指定します。学習済みモデルの推論とレベル21のエンジン評価はGPUを使いません。

石損計算には、リポジトリ内の次のエンジンと、その同梱 `resources/` が必要です。エンジンの場所は [../00_data_processing/miss_analysis.py](../00_data_processing/miss_analysis.py) の `default_engine()` に従います。

```text
bin/versions/Egaroucid_for_Console_7_8_1_Windows_SIMD/
  Egaroucid_for_Console_7_8_1_SIMD.exe
  resources/
    eval.egev2
    eval_move_ordering_end.egev
    hash/hash25.eghs
```

この配置はEgaroucidリポジトリのルートを基準にしています。

データの格納場所は、次のどちらかの環境変数で設定します。

| 環境変数 | 指定する場所 |
|---|---|
| `POLICY_DATA_ROOT` | `20261002_policy_network_human_like_data` に相当するディレクトリ。優先して使う |
| `EGAROUCID_DATA` | 上記ディレクトリの親。`POLICY_DATA_ROOT` が未設定の場合に使う |

次のパスは例です。自分の保存先に置き換えてください。

```powershell
$env:POLICY_DATA_ROOT = 'X:/othello-data/20261002_policy_network_human_like_data'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
```

既定で読み込むディレクトリ構成は次のとおりです。

```text
<POLICY_DATA_ROOT>/othello_quest_split/board_split/800-1000/
  manifest.json
  train/*.dat, *.game_ids
  val/*.dat,   *.game_ids
  test/*.dat,  *.game_ids
```

### 初回に学習から平均石損まで実行する

```powershell
powershell -ExecutionPolicy Bypass -File ./run_visible.ps1 -Stage all -Workers 8
```

学習、テスト一致率の測定、レベル21の平均石損、配布用JSONの出力を順に行います。学習済みモデルを使った一致率測定は `train.py` の学習後の処理に含まれます。CPU上限は、石損計算が最大8プロセス・各1スレッドです。`-Workers` に9以上を指定すると起動時に拒否します。コンソールには処理段階、進捗、経過時間、推定残り時間を表示し、ログにも保存します。

### 学習だけ実行する

```powershell
powershell -ExecutionPolicy Bypass -File ./run_visible.ps1 -Stage train
```

学習の最適化を最初から行い、学習済み重み・検証履歴・テスト一致率・予測した手を保存します。入力とコードが一致すれば、計算済みの特徴量を再利用します。学習の途中の係数から再開する機能はありません。

学習後、配布用の `trained_policy.json` と `results_summary.json` を更新する場合は次を実行します。

```powershell
python export_results.py
```

### 石損計算だけ実行・再開する

```powershell
powershell -ExecutionPolicy Bypass -File ./run_visible.ps1 -Stage loss -Workers 8
```

`output/quest_800-1000/` に保存されたモデルと予測した手を使います。石損の計算結果は組ごとに保存し、同じ保存先で再実行すれば未完了分から再開します。Ctrl+Cまたはコンソールを閉じると、その実行と子エンジンが終了します。

モデル・入力局面・エンジン・評価条件の識別情報が異なる場合は、再開を拒否します。コードだけを修正し、評価内容に変更がないことを確認した場合は、次の指定で互換性を確認して再開できます。

```powershell
powershell -ExecutionPolicy Bypass -File ./run_visible.ps1 -Stage loss -Workers 8 -ResumeCompatibleCode
```

この指定でも、モデルや盤面などの不一致は許可しません。READMEだけの変更には、この指定は不要です。

### 別の条件で再実験する

進行中の石損計算が参照している学習結果を、再学習で上書きしないでください。別の保存先を使う実行例です。上記の環境変数を設定した、ユーザーに見えるPowerShellで実行します。

```powershell
python -u train.py --output ./output/my_rules --device cuda --iterations 120 --l2 0.0001 0.00001 --seed 613
python -u evaluate_loss.py --input ./output/my_rules --output ./output/my_rules/loss_l21 --workers 8
python export_results.py --input ./output/my_rules
```

`export_results.py` は、指定した実験の結果をこのフォルダー直下の配布用JSONへ反映します。
石損結果は `<--inputの保存先>/loss_l21/result.json` から読むため、配布用JSONへ反映する実験では、石損の保存先を上の例の構成にしてください。

| スクリプト | 主な引数 |
|---|---|
| `train.py` | `--data`：`manifest.json` のある棋力帯ディレクトリ、`--output`：保存先、`--device`：`cuda` / `cpu`、`--iterations`、`--l2`、`--seed`、`--baselines`：既存NNの評価フォルダー |
| `evaluate_loss.py` | `--input`：学習結果、`--output`：石損の保存先、`--workers`：1〜8、`--hash-level`：置換表のサイズ指定、`--resume-compatible-code` |
| `predict.py` | `--model`、`--player`、`--opponent` |
| `export_results.py` | `--input`、`--wait-for-loss`：全件の石損完了を待つ、`--commit`：配布用JSONだけをコミットする |

`--baselines` の既定は、リポジトリの親にある `experiment_bat/20261010_quest_rating_bands/evaluation` です。既存NNの保存予測が見つかった場合、同じ局面・棋譜ID・合法手・予測ファイルのハッシュを確認して比較します。見つからなければNN比較を省略します。

動作確認用に石損を一部だけ計算する場合は、全件とは別の保存先を指定します。

```powershell
python -u evaluate_loss.py --max-positions 3 --workers 1 --output ./output/l21_smoke_demo
```

この結果は部分評価として記録され、全件の平均石損としては出力しません。

## 評価指標と今回の結果

### 人間との着手一致率

最大確率の1手を選び、その手が人間の実際の着手と同じなら正解です。また、自石・相手石の配置を変えない回転・鏡映で、人間着手から移せる手も正解に含めます。盤面自体を変えてしまう変換で別の手を正解にすることはありません。

全30,423テスト局面を使った保存済み結果です。各NNも同じテスト局面・棋譜IDを照合し、保存予測から再集計しています。

| 方策 | 一致件数 | 一致率 |
|---|---:|---:|
| 学習したルール | 14,159 | **46.54%** |
| 手作業の重み | 12,304 | 40.44% |
| 最多取り | 9,964 | 32.75% |
| 合法手から一様ランダム | 期待値 | 23.29% |
| 既存の追加学習Conv8+BN | 15,372 | 50.53% |
| Questのみで学習した既存Conv8+BN | 15,318 | 50.35% |

今回のルールは、手作業の重みから約6.10ポイント改善しましたが、既存NNを上回りませんでした。一様ランダムの値は確率から直接計算した期待値で、1回の抽選結果ではありません。`results_summary.json` には対称性を含めない完全一致率と、同じ棋譜をまとめて再抽出するブートストラップによる一致率差の95%区間も保存しています。

### レベル21による平均石損

同じ全30,423テスト局面について、ルール方策が最大確率で選んだ手を評価します。方策からランダムに引いた手や、全合法手にわたる損失の確率平均ではありません。

参照エンジンはEgaroucid for Console 7.8.1 SIMD、レベル21、book無効、各プロセス1スレッド、置換表保持、既定のhash levelは25です。`setboard`、`play`、`analyze` で、選択手と最良の他の手の評価を得ます。両評価は、着手した側から見た予測石差です。

```text
1局面の石損 = max(選択手の評価, 最良の他の手の評価) - 選択手の評価
平均石損 = 全テスト局面の石損合計 / 30,423
```

合法手1つの局面は探索を省いて0石損とします。この組の `scores.npy` の3列はすべて0を保存し、評価値の2列も便宜上の0です。同じ局面と選択手の組は計算を共有し、集計では元の出現回数だけ数えます。欠測は0に置き換えません。これはL21の推定値による1着手あたりの損失で、終局石差そのものではありません。

平均石損の出力先は `results_summary.json` の `disc_loss_l21.metrics.mean_disc_loss` です。`disc_loss_l21.status` が `pending` の間は、集計結果がまだ反映されていません。計算自体の状態は `output/quest_800-1000/loss_l21/status.json` の `running` / `complete` / `failed` と進捗を確認してください。

## ファイルと出力の対応

| ファイル | 役割 |
|---|---|
| [rule_policy.py](rule_policy.py) | 合法手・反転石・43特徴量・3段階の係数・softmax・手作業の比較用重み |
| [train.py](train.py) | 重みの学習、検証での採用、全テストの一致率、既存NNとの比較 |
| [predict.py](predict.py) | 任意局面の着手確率と特徴量ごとの得点内訳を表示 |
| [evaluate_loss.py](evaluate_loss.py) | 保存された最大確率手のL21平均石損を計算・再開 |
| [run_visible.ps1](run_visible.ps1) | 可視コンソールで学習・石損計算を起動し、ログを保存 |
| [export_results.py](export_results.py) | 配布用モデルと集計JSONの出力、完了待ち、結果だけのコミット |
| [common.py](common.py) | データ読込、ハッシュ検査、一致率、棋譜単位のブートストラップ |
| [test_rules.py](test_rules.py) | 初期盤面、反転石、対称性、確率、欠測石損の扱いを検証 |
| [trained_policy.json](trained_policy.json) | 推論にそのまま使える43×3の学習済み重み |
| [results_summary.json](results_summary.json) | 実験条件、一致率、平均石損、モデル・予測・入力のハッシュ |

大きい作業ファイルはGit対象外の `output/quest_800-1000/` に保存します。

```text
output/quest_800-1000/
  identity.json                  入力・コード・学習条件
  features/{train,val,test}/     特徴量の再利用用キャッシュ
  training_history.json          各反復の検証結果
  selected_model.json            採用した重みと選択根拠
  weights.csv                    特徴量ごとの序盤・中盤・終盤の係数
  agreement.json                 一致率の全結果と比較条件
  trained_rules_predictions.npy  全テスト局面で選んだ手
  test_probabilities.npy         全テスト局面の64マスの確率
  test_records.npy               予測に対応するテスト局面
  test_game_ids.npy              予測に対応する棋譜ID
  visible_*.log                  可視実行のログ
  loss_l21/
    identity.json                石損計算の入力・モデル・エンジン条件
    pairs.npy                    重複をまとめた局面と選択手の組
    row_to_pair.npy              元の各局面から組への対応
    scores.npy                   選択手評価・他の最良手評価・石損
    status.json                  現在の進捗、速度、残り時間
    per_position_loss.npy        元の全テスト局面ごとの石損
    result.json                  全件完了後の平均石損と検証結果
```

`selected_model.json` は学習した保存先のモデル、`trained_policy.json` はそこから書き出した配布用モデルです。`predict.py` の既定は後者であり、学習直後のモデルを直接調べたい場合は `--model ./output/quest_800-1000/selected_model.json` を指定します。

## ルールを変更して再実験する

1. `rule_policy.py` の `FEATURE_NAMES` と `extract()` を変更する。特徴量の順序と出力列をそろえる。
2. `python -m unittest test_rules -v` で、合法手・独立した座標計算による反転石・回転と鏡映・確率の整合性を確認する。
3. 新しい `--output` で学習する。特徴量や学習コードを変更した場合、古い特徴量キャッシュの再利用は拒否される。
4. 検証で採用されたモデルを、同じテストとL21条件で評価する。
5. `export_results.py --input <新しい実験の保存先>` でモデルと結果を出力する。

特徴量を追加する際も、盤面と着手を同時に回転・鏡映したとき、対応する特徴量が同じになるようにします。単に特定の座標を好む特徴を足すと、現在の「8通りの変換は同じ学習になる」という前提が変わるため、データの水増し方針も見直す必要があります。
