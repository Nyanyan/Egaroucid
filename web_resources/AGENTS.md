# Webページの翻訳表記

英訳の手順は `GEMINI.md` を参照してください。以下の翻訳表記の条件は、使用する翻訳ツールに関係なく適用します。

- ページ本文のすべてを機械翻訳した履歴を確認できる場合に限り、ページ全体の機械翻訳を示す注意書きを付けます。
- 既存の英文を残して一部だけ翻訳・追加・修正した場合は、この注意書きを新たに付けません。既存の注意書きを別のページへ一括で追加することも禁止します。
- 注意書きに翻訳ツール名を含めません。全文機械翻訳として確認済みのページを部分更新するときは、ツール名を含まない注意書きを維持します。
- 注意書きはタイトル直後に太字で置き、日本語版へのリンクを付けて、次の文面を使います。

```text
This page was machine translated from the Japanese version. Please refer to the Japanese version for the official content.
```

全文機械翻訳として確認済みのページは、使い方 (`en/usage/index.md`)、Book詳細 (`en/usage/book/index.md`)、技術解説 (`en/technology/explanation/index.md`) の3つです。トップ、ダウンロード、Consoleには注意書きを付けません。

注意書きの許可対象は、`check_translation_notices.py` の `FULL_TRANSLATIONS` に記録したページだけです。対象を追加するときは、ページを全文英訳として追加または置換した履歴を確認し、そのコミットIDを根拠として記録してください。部分更新の履歴だけでは対象に追加できません。

変更後はリポジトリ直下で次の検査を実行してください。

```powershell
python web_resources/check_translation_notices.py
python -m unittest discover -s web_resources -p test_translation_notices.py
```

検査はすべての英語原稿と、存在する英語の中間生成HTML・公開用HTMLを確認します。HTML生成は英語原稿を、公開用コピーは英語原稿と中間生成HTMLを、書き込み前に検査します。許可対象以外のページへの注意書き、ツール名を含む注意書き、許可対象ページでの注意書きの欠落を検出した場合は停止します。
