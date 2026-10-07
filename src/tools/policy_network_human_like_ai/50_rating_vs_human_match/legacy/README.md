# 過去の描画スクリプト・実験手順・BAT

このフォルダには、16モデル・共通30,000局面を対象とした旧図を保存しています。現在の23モデルのグラフは、親フォルダの`plot_rating_vs_human_match.py`で生成します。

Dense512×4とConv8＋BNの旧実験用BATも、このフォルダに保存しています。手順は[MODEL_BLEND_EXPERIMENT.md](MODEL_BLEND_EXPERIMENT.md)を参照してください。BATは親フォルダの実行処理を呼び出すため、出力先や相対パスの基準は以前と同じです。

旧図を再生成するときは、このフォルダで次のコマンドを実行してください。

```powershell
python .\plot_rating_vs_human_match.py
```

`--adjust-labels`でラベル位置の編集画面を開けます。旧スクリプトは、このフォルダに移したCSVと`label_positions.json`を既定で読み込みます。旧図の条件と測定結果は[RESULTS.md](RESULTS.md)を参照してください。

旧テストは親フォルダの新しい描画モジュールと混在させず、別のプロセスで実行します。

```powershell
python -m unittest discover -s . -p test_plot_rating_vs_human_match.py -v
```
