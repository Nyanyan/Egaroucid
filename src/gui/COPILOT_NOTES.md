# GitHub Copilot 開発メモ

## 言語対応について

### 重要: 言語周りは必ず `language.get` を使用すること

ハードコードされた文字列をUIに表示する際は、必ず多言語対応のために `language.get()` を使用してください。

**例:**
```cpp
// ❌ 悪い例
Button button;
button.init(..., U"New CSV", ...);

// ✅ 良い例
Button button;
button.init(..., language.get("opening_setting", "new_category"), ...);
```

### 言語パックの編集

新しいUIテキストを追加する場合は、以下の4つの言語ファイルすべてを更新してください:

1. `bin/resources/languages/japanese.json`
2. `bin/resources/languages/english.json`
3. `bin/resources/languages/simplified_chinese.json`
4. `bin/resources/languages/traditional_chinese_taiwan.json`

**例:**
```json
"opening_setting": {
    "opening_setting": "強制する進行の設定",
    "add": "追加",
    "new_category": "新しいカテゴリ",
    "category_name": "カテゴリ名"
}
```

### 用語の統一

- CSV → カテゴリ (ユーザーに表示する際)
- 内部的には`csv_file`などの変数名は維持可能
- ユーザーに見える文字列は必ず`language.get()`経由で提供

## UI実装のベストプラクティス

### カラーコード
- 選択中の項目: `getData().colors.dark_blue`
- 無効化された項目: グレーアウト `ColorF(0.3, 0.3, 0.3)`
- 通常の項目: `getData().colors.green` / `getData().colors.dark_green` (交互)

### ボタンとUI要素
- 必ず`language.get()`を使用
- アイコンと組み合わせる場合も多言語テキストを使用
- ツールチップやヘルプテキストも言語パック経由

## 最終更新
2025年12月2日
