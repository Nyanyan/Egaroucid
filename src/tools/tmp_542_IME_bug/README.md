# tmp_542_IME_bug

Siv3D で IME の未消費文字入力が `TextArea` に流入する現象を確認するための最小再現プロジェクトです。

## Files

- `tmp_542_IME_bug.sln`
- `tmp_542_IME_bug.vcxproj`
- `App/Main.cpp`

## Repro steps

1. `tmp_542_IME_bug.sln` を Visual Studio で開く
2. 起動後、Main scene で IME をかな入力状態にする
3. Main scene 上で `A` を2回、`D` を2回、`E` を1回押す
4. `Open Sub Scene (TextArea)` ボタンを押す
5. Sub scene の `TextArea` に、遷移直後から文字が入っているか確認する

## Notes

- Main scene では意図的に `TextInput::UpdateText()` を呼んでいません。
- そのため IME で確定した文字入力イベントが保留され、`TextArea` 初回アクティブ時に反映されるかを観察できます。
