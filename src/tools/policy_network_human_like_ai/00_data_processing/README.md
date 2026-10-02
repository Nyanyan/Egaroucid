# ポリシーネットワーク用データの加工

棋譜の抽出と盤面データへの変換は、このディレクトリで行います。
加工後のデータを使った学習手順は、[10_train_policy_network](../10_train_policy_network/README.md) に記載しています。

## 加工スクリプト

| スクリプト | 処理 |
| --- | --- |
| `select_transcripts.py` | `transcript_release/0002` から指定した数の棋譜を重複なしで抽出します。 |
| `convert_selected_transcripts_to_board_data.py` | 抽出した棋譜を盤面データへ変換します。ランダム序盤の手は盤面上で再生し、その後の局面だけを書き出します。 |

両スクリプトは Python の標準ライブラリだけで実行できます。
入力・出力の既定パスは、スクリプトの場所から求めたリポジトリルートを基準にします。

## 棋譜の抽出と変換

次の例は、リポジトリルートの `Egaroucid` ディレクトリから PowerShell で実行します。
まず、seed 613 で10,000局を抽出します。

```powershell
python src\tools\policy_network_human_like_ai\00_data_processing\select_transcripts.py `
  --num-games 10000 `
  --seed 613
```

抽出元は `train_data/transcript_release/0002`、保存先は
`train_data/transcript/Egaroucid_Train_Data_v2_selected` です。
抽出元の各ファイルに10,000局が含まれることを前提にしています。
保存先には、ランダム序盤の手数ごとの棋譜ファイルと、抽出元を記録した `selection_manifest.csv`、集計した `selection_summary.json` を書き出します。

続いて、抽出した棋譜を盤面データへ変換します。

```powershell
python src\tools\policy_network_human_like_ai\00_data_processing\convert_selected_transcripts_to_board_data.py
```

変換後の盤面データは `train_data/board_data/Egaroucid_Train_Data_v2_selected/records0/*.dat` に保存します。
各局面は、手番側の石・相手側の石・手番の色・着手・最終得点を含む19バイトのデータです。
また、変換結果を `conversion_summary.json` に、不正な棋譜があればその理由を `invalid_games.csv` に保存します。

保存先を変更する場合は、抽出時の `--output-root` と変換時の `--input-root` に同じパスを指定します。
変換後の保存先は、変換時の `--output-root` で変更できます。
両スクリプトとも、出力先ディレクトリがすでに存在する場合は、その中身を削除してから再生成します。

加工が終わったら、[学習ディレクトリ](../10_train_policy_network/README.md) の `train_policy_network.py` を実行します。
この selected データを学習する場合は `--wthor` を省きます。WTHOR の既存データを学習する場合は、同READMEの `--wthor` を使った手順に従ってください。
