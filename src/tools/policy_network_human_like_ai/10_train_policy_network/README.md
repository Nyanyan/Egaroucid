# WTHORポリシーネットワーク学習

関連issue: #613

このディレクトリでは、WTHOR棋譜から展開した局面を使い、人間の着手を予測する
Othello/Reversi用のポリシーネットワークを学習します。

WTHORデータの読み込み・棋譜分割・入力特徴量の作成は、[00_data_processing](../00_data_processing/README.md) で行います。

このREADMEは現在の学習手順を説明します。RESULTSと保存済みモデルは、変更前の局面単位の分割による実験結果です。
棋譜単位の分割で評価するには、新しい出力先で再学習してください。
以前の作業記録はローカルアーカイブ
`src/tools/policy_network_human_like_ai/report/legacy` に移動済みで、現行の採用判断には使いません。

## 用語

- `games` は、1局分の完全な棋譜を意味します。
- `局面サンプル` は、board-dataのバイナリファイルに保存された展開済み局面を意味します。
  学習時の入力は局面サンプルですが、train・validation・test の分割は棋譜単位で行います。
- `records1` などの既存ディレクトリ名は、現在のデータ配置の一部であるため、パス名としてのみ使います。

## 入力

入力は黒白固定ではなく、手番目線です。

- 手番側の石がある64マス
- 相手側の石がある64マス
- 合計128個の浮動小数点入力

board-dataのバイナリデータには `player` と `opponent` として保存されているため、学習コードはそれをそのまま使います。

## データセットと分割

入力データ:

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

既定でWTHOR由来の全棋譜を読み込みます。`--wthor` は同じ入力を使う互換オプションです。
seed `613` で棋譜の順序をシャッフルし、棋譜数の5%ずつを validation と test、残りを train に割り当てます。
5%に満たない端数は切り捨て、残りを train に含めます。
同じ棋譜から生成された局面は、すべて同じ分割に所属します。

現在のデータと既定の seed では、次のように分かれます。
棋譜ごとの手数が異なるため、局面数の割合は90%・5%・5%から少しずれます。

| 用途 | 棋譜数 | 局面サンプル数 |
| --- | ---: | ---: |
| train | 120,763 | 7,231,670 |
| validation | 6,709 | 401,726 |
| test | 6,709 | 401,886 |
| 合計 | 134,181 | 8,035,282 |

WTHORのboard dataは棋譜順に並んでいるため、初期局面から棋譜の境界を復元します。
複数の `.dat` ファイルにまたがる棋譜も、同じ棋譜として扱います。
この実験では初期局面を含むWTHORデータを使い、`transcript_release/0002` の棋譜は使いません。
棋譜を識別できないデータはエラーとして扱います。

分割モードの既定値は `--split-mode game` です。従来の `shuffled` も同じ棋譜単位の分割になります。
`sampled` は、棋譜単位で分割した後、それぞれの分割内から重複なしで局面を抽出します。
`--max-train-samples` と `--max-val-samples` は `sampled` の場合だけ有効です。
`--max-position-samples` を指定すると、上限内に収まる棋譜を先頭から丸ごと読み込み、その集合を分割します。
上限が棋譜の途中に来た場合、その棋譜は含めません。

出力先の `dataset_split.json` に seed・比率・棋譜数・局面数・入力ファイルのSHA-256を保存します。
また、`dataset_split.npz` に各分割の局面インデックスを保存するため、再評価でも学習時と同じ分割を使えます。

## モデル採用ルール

各学習条件では、指定したエポック数まで必ず学習し、学習完了時点のモデルだけを使います。
途中エポックのモデルは採用しません。

複数のモデル構成を比較するときは、それぞれ指定エポック数まで学習した後のモデルを検証データで評価します。
代表モデルは、検証データ1位一致率が最も高い構成です。同率の場合は、検証データ上位3手一致率、パラメータ数の少なさの順で比較します。
テストデータは代表モデルを決めるためには使わず、採用後の最終確認にだけ使います。

## 一致率

一致率は、合法手だけを残したポリシーネットワーク出力で計算します。

各局面について、正方形の盤面に対する8通りの変換
（そのまま、90度・180度・270度回転、縦・横・2本の対角線を軸にした反転）を調べます。
そのうち、手番側・相手側の石配置を変えない変換だけを一致判定に使います。
WTHORで実際に打たれた手と、その手をこれらの変換で移した合法手を同一の着手として扱います。
それ以外の対称的な位置にある手は別の着手です。

- 1位一致率: 同一とみなす着手のいずれかが、合法手内で最も確率の高い手だった局面サンプルの割合です。
- 上位n手一致率: 同一とみなす着手のいずれかが、合法手内の確率上位n手に入った局面サンプルの割合です。

盤面に該当する対称性がなければ、実際に打たれた1手だけが一致対象になります。

## 最大メインメモリ使用量

RESULTSの「最大メインメモリ使用量(MiB)」は、学習コマンド実行中に `resource_monitor.py` が観測した最大のメインメモリ使用量です。
GPUメモリ量ではなく、モデルファイルサイズでもありません。

## 学習コマンド例

分割結果を先に作る場合は、`00_data_processing/prepare_wthor_data.py` を実行し、
学習コマンドの `--split-manifest` に生成された `dataset_split.json` を指定します。
これにより、入力のSHA-256を照合したうえで同じ棋譜分割を再利用できます。
分割のseed・比率は保存済みの設定を使い、学習時の件数上限や `--split-mode sampled` は併用できません。
手順は [WTHORデータの加工](../00_data_processing/README.md) を参照してください。

512×4のモデルを棋譜単位の分割で学習する例:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_game_split_512x4_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_game_split_512x4_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode game `
    --configs w512_d4_a0.03:512:4:0.03:0.0:0.0:0.001 `
    --epochs 50 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_game_split_512x4_e50
```

代表モデルの生成物は `selected_model.h5` と `selected_policy_network_weights.bin` です。

## 保存済みモデルの再評価

モデルサイズ比較で保存した8モデルを、上記の一致率だけで検証データとテストデータに再評価するコマンド:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\evaluate_model_size_experiment.py
```

`EGAROUCID_DATA` を使わない場合は、`--board-data-dir` でboard dataのディレクトリを指定します。
集計結果は次のファイルに保存されます。

- `model_size_results.json`: ヒット数、局面サンプル数、分割・入力・モデル・評価コードのSHA-256を含む完全な結果
- `model_size_results.csv`: モデルごとの一致率とリソース使用量の一覧

モデルごとの詳細な評価結果は
`../20_test_with_wthor/output/model_size_symmetry_aware` に生成されます。
