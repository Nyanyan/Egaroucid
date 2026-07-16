# CSV transcript converter

`train_data/transcript/records1_raw` の年度別CSVを読み、合法手を再生して終局を
確認できた棋譜だけを、1行1棋譜の連番 `.txt` に変換します。パスは省略時に
リポジトリ内の `records1_raw` と `records1` を使用します。
ヘッダー付きCSVと、旧年度にある1行1棋譜だけのCSVの両形式に対応します。

```powershell
python src/tools/wthor_convert/csv_to_transcript.py `
  --start-year 1977 `
  --output-start-number 0
```

この例では `1977.csv` が `0000000.txt`、`1978.csv` が `0000001.txt` に
なります。途中の年度までに限定する場合は `--end-year 2020`、入出力先を変える
場合は `--input-dir` と `--output-dir` を指定してください。

出力先に同名ファイルがある場合は上書きします。CSVに年度の欠番がある場合も、
出力番号は処理対象CSVの順に連続します。
