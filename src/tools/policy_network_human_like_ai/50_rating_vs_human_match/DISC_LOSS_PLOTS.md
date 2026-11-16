# 平均石損と着手一致率の図

保存済みの平均石損とWTHORの着手予測から、全体図と右側の拡大図を作ります。横軸は平均石損が小さいほど右に進みます。各表示範囲について、Zebraあり・なしの軸範囲を揃えます。

```text
python plot_disc_loss_agreement.py --data output/disc_loss_dataset/plot_data.json --output-dir output/disc_loss_dataset/figures
```

PNG・SVG・PDFと計測条件のJSONを出力します。既定ではZebraあり・なしの両方を作ります。片方だけ作る場合は `--zebra with` または `--zebra without` を指定します。図・データ・ラベル位置はリポジトリのignore対象である `output` 配下に保存します。

`--arrange-labels` を付けると、文字同士に加え、点・折れ線・縦線・凡例・ほかのラベルの引き出し線と重ならないように再配置し、位置を保存します。保存するJSONには種類ごとの衝突検査結果も記録します。

横軸を4→0にするには `--x-max 4` を指定します。この版では、石損が4を超えるモデルの点とラベルを描きません。人間の縦線が範囲外の場合、凡例に「範囲外」と付記します。元のデータと平均値は変更しません。全範囲版と4→0版のラベル位置は別々に保存します。

```text
python plot_disc_loss_agreement.py --data output/disc_loss_dataset/plot_data.json --output-dir output/disc_loss_dataset/figures --x-max 4 --arrange-labels
```

データJSONには、各モデルの `model` / `mean_disc_loss` / `wthor_accuracy`、縦線の `references`、モデル順 `covariance_model_order`、平均値の共分散 `covariance_x` / `covariance_y` を入れます。共分散とモデル行の対応が一致しない入力は拒否します。

`prepare_disc_loss_plot_data.py` は、保存済みのQuest石損配列と帯ID、WTHOR予測NPZ、人間のWTHOR石損配列からこのJSONを作ります。各引数は `--help` で確認できます。WTHOR予測を複数指定した場合は同じ局面・棋譜・対称手マスクであることを照合し、重複モデルは先に指定したファイルを優先します。Questは各帯の件数を揃え、帯を等しく重み付けします。WTHORの人間の石損配列には、一致率評価と同じ標本のL21石損を指定してください。探索や学習は行いません。

提案手法3系列の系列名・色・線種・マーカーと、各系列内0-1のα表記は論文用の図に合わせています。Egaroucidは `Lv. N` と表示し、通常モデル・Lv. 21・α=0.7・α=0.8の4点に一致率を表示します。石損4以上・6以上の追加学習モデルは独立した点として表示し、連続したαの系列として接続しません。

帯は、両軸を含む系列ごとの近似95%同時信頼帯です。モデル間の共分散を保った平均値の多変量正規近似から20,000標本を生成し、各系列の全頂点・両軸で標準化偏差の最大値を校正します。石損0に固定されたL21座標は定数として扱います。WTHORとQuestの標本は独立とし、Questの帯内およびWTHORの各行も独立と仮定します。棋譜内の相関と重複盤面の依存は補正していません。全系列をまとめた同時保証や、未測定モデルの性能に対する保証ではありません。

```text
python plot_disc_loss_agreement.py --data output/disc_loss_dataset/plot_data.json --output-dir output/disc_loss_dataset/figures --edit with-zebra
```

`--edit without-zebra` でZebraなし版を編集できます。ラベルのドラッグ、文字のダブルクリック、Sキーでの位置・画像保存に対応します。2版のラベル位置は別々に保存します。
