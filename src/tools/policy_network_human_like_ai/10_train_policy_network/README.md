# WTHORポリシーネットワーク学習

関連issue: #613

このディレクトリでは、WTHOR棋譜から展開した局面を使い、人間の着手を予測する
Othello/Reversi用の軽量ポリシーネットワークを学習します。

現在のREADMEとRESULTSは、WTHORを直接学習データにした実験だけを対象にしています。
以前の作業記録は、ローカルアーカイブ
`src/tools/policy_network_human_like_ai/report/legacy` に移動しました。

## 用語

- `games` は、1局分の完全な棋譜を意味します。
- `局面サンプル` は、board-dataのバイナリファイルに保存された展開済み局面を意味します。
  学習器はこの局面サンプルを単位として扱います。
- `records1` などの既存ディレクトリ名は、現在のデータ配置の一部であるため、
  パス名としてのみ使います。

## 入力

入力は固定の黒白ではなく、手番相対です。

- 手番側の石がある64マス。
- 相手側の石がある64マス。
- 合計128個の浮動小数点入力。

board-dataのバイナリデータには `player` と `opponent` として保存されているため、
学習コードはそれをそのまま使います。

## データセットと分割

入力データ:

```text
$EGAROUCID_DATA/train_data/board_data/records1
```

`--wthor --split-mode shuffled` を使うと、WTHOR由来の全局面サンプルを読み込みます。
seed `613` で局面サンプル全体をシャッフルし、学習データ、検証データ、テストデータに
分割します。この分割の単位は局面サンプルです。

| 用途 | 局面サンプル数 |
| --- | ---: |
| 学習 | 6,428,225 |
| 検証 | 803,528 |
| テスト | 803,529 |
| 合計 | 8,035,282 |

## 学習後に使うエポックの選び方

WTHORのシャッフル分割を使う学習では、各エポック終了時に検証データで一致率を計算します。
このとき、合法手だけを候補に残してポリシーネットワークの出力を順位付けします。
WTHORで実際に打たれた手が1位だった局面サンプルの割合を「検証データ1位一致率」と呼びます。

学習後に使うモデルは、検証データ1位一致率が最も高かったエポックのモデルです。
このモデルが `best_model.h5` として保存されます。指定した最終エポックのモデルを
無条件に使うわけではありません。

複数の学習条件を比較するときも、まず検証データ1位一致率で採用モデルを決めます。
テストデータは採用判定には使わず、採用後の最終確認だけに使います。検証データ1位一致率が
完全に同じ場合は、検証データ上位3手一致率、パラメータ数の少なさ、学習時間の短さの順で
比較します。

## 上位N手一致率

上位N手一致率は、WTHORで実際に打たれた手が、合法手だけに絞ったポリシーネットワークの
予測順位で上位N手に入った割合です。

例:

- 1位一致率: 実際の着手が、合法手内で最も高い確率だった割合。
- 上位3手一致率: 実際の着手が、合法手内の上位3手に入った割合。

## 最大メインメモリ使用量

RESULTSの「最大メインメモリ使用量(MiB)」は、学習コマンド実行中に
`resource_monitor.py` が観測した最大のメインメモリ常駐量です。GPUメモリ量ではなく、
モデルファイルサイズでもありません。保存されるJSONでは、過去ログとの互換性のため
内部キー名として `peak_rss_mib` が残っていますが、表では使いません。

## 学習コマンド例

現在採用しているモデル系の例:

```powershell
python src\tools\policy_network_human_like_ai\10_train_policy_network\run_with_resource_log.py `
  --log src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50.log `
  --summary src\tools\policy_network_human_like_ai\10_train_policy_network\train_log\wthor_large_768x4_dropout_lr_e50_resource.json `
  -- python src\tools\policy_network_human_like_ai\10_train_policy_network\train_policy_network.py `
    --wthor `
    --split-mode shuffled `
    --configs w768d4_drop005_lr5e4:768:4:0.03:0.05:0.0:0.0005 `
    --epochs 50 `
    --patience 8 `
    --batch-size 8192 `
    --eval-batch-size 65536 `
    --predict-batch-size 8192 `
    --output-dir src\tools\policy_network_human_like_ai\10_train_policy_network\trained\wthor_large_768x4_dropout_lr_e50
```

生成されたモデル、生ログ、リソース概要、アーカイブ済みの作業記録はgit管理外です。
