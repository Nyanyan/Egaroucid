# WTHORポリシーネットワーク学習

関連issue: #613

このディレクトリでは、WTHOR棋譜から展開した局面を使い、人間の着手を予測する
Othello/Reversi用のポリシーネットワークを学習します。

現行のREADMEとRESULTSは、WTHORを直接学習データにした固定条件の実験だけを対象にしています。
以前の作業記録はローカルアーカイブ
`src/tools/policy_network_human_like_ai/report/legacy` に移動済みで、現行の採用判断には使いません。

## 用語

- `games` は、1局分の完全な棋譜を意味します。
- `局面サンプル` は、board-dataのバイナリファイルに保存された展開済み局面を意味します。
  学習器はこの局面サンプルを単位として扱います。
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

`--wthor --split-mode shuffled` を使うと、WTHOR由来の全局面サンプルを読み込みます。
seed `613` で局面サンプル全体をシャッフルし、学習データ、検証データ、テストデータに分けます。

| 用途 | 局面サンプル数 |
| --- | ---: |
| 学習 | 6,428,225 |
| 検証 | 803,528 |
| テスト | 803,529 |
| 合計 | 8,035,282 |

## モデル採用ルール

各学習条件では、指定したエポック数まで必ず学習し、学習完了時点のモデルだけを使います。
途中エポックのモデルは採用しません。

複数のモデル構成を比較するときは、それぞれ指定エポック数まで学習した後のモデルを検証データで評価します。
代表モデルは、検証データ1位一致率が最も高い構成です。同率の場合は、検証データ上位3手一致率、パラメータ数の少なさ、学習時間の短さの順で比較します。
テストデータは代表モデルを決めるためには使わず、採用後の最終確認にだけ使います。

## 一致率

一致率は、合法手だけを残したポリシーネットワーク出力で計算します。

- 1位一致率: WTHORで実際に打たれた手が、合法手内で最も確率の高い手だった局面サンプルの割合です。
- 上位n手一致率: WTHORで実際に打たれた手が、合法手内の確率上位n手に入った局面サンプルの割合です。

## 最大メインメモリ使用量

RESULTSの「最大メインメモリ使用量(MiB)」は、学習コマンド実行中に `resource_monitor.py` が観測した最大のメインメモリ使用量です。
GPUメモリ量ではなく、モデルファイルサイズでもありません。

## 学習コマンド例

現在の代表モデルと同じ条件で学習する例:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_final_arch_512x4_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_final_arch_512x4_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs w512_d4_a0.03:512:4:0.03:0.0:0.0:0.001 `
    --epochs 50 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_final_arch_512x4_e50
```

代表モデルの生成物は `selected_model.h5` と `selected_policy_network_weights.bin` です。
