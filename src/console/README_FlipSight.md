# FlipSightボット接続

Egaroucid for ConsoleをFlipSightのボット対局APIに直接接続します。対応種目は標準の8×8オセロ（std）です。FlipSightのボット管理画面でも、対応ゲームを「標準」だけに設定してください。

## ビルド

CMakeでは `INCLUDE_FLIPSIGHT` を有効にします。既定値はOFFです。

```powershell
cmake -S . -B build/flipsight -DINCLUDE_FLIPSIGHT=ON
cmake --build build/flipsight --config Release
```

Windowsは標準のWinHTTPを使います。Linux・macOSではlibcurlの開発用ヘッダーとライブラリが必要です。コンパイラーを直接呼ぶ場合は `INCLUDE_FLIPSIGHT` を定義し、Windowsでは `winhttp`、Linux・macOSでは `curl` をリンクしてください。

既存のVisual Studioプロジェクトでも有効にできます。

```powershell
msbuild Egaroucid_for_Console.vcxproj /p:Configuration=SIMD /p:Platform=x64 /p:IncludeFlipSight=true
```

GGSとFlipSightは同じ実行ファイルに組み込めます。起動時にはどちらか一つの接続モードを選んでください。

## 起動

生成した実行ファイルの隣に、通常のコンソール版と同じ `resources/` を配置してください。このリポジトリでは `bin/resources/` にあるコンソール用のリソースを使います。評価ファイルは `eval.egev2` と `eval_move_ordering_end.egev`、定石ブックは `book.egbk3` です。以下はWindowsのCMake Releaseビルドを、リポジトリ直下から起動する例です。`<token>` をFlipSightで発行したボット用トークンに置き換えます。

```powershell
.\bin\Release\Egaroucid_for_Console.out.exe -flipsight "<token>" -thread 8 -hash 24
```

`-level` を省略すると、FlipSightから受け取った残り時間に合わせて探索します。レベルを指定する場合は、同じコマンドに `-level 3` などを追加します。指定できるレベルは0〜60です。指定レベルの探索は、残り時間から通信の余裕を差し引いた時間を上限に実行します。定石ブックを使わず、指定レベルの探索だけで着手を選ぶ場合は `-nobook` も追加してください。

初期化後に対局の待ち受けを始め、対応する申し込みを自動で承諾します。Ctrl+Cで通信・探索・ponderを停止して終了します。

| 引数 | 動作 |
| --- | --- |
| `-flipsight <token>` | ボット用トークンを渡して接続する |
| `-thread N` | 探索に割り当てるスレッド数を指定する |
| `-level N` | 指定レベルで探索する。省略時は残り時間に合わせる |
| `-flipsightnoponder` | 相手の手番中の先読みを無効にする |
| `-flipsightnostream` | イベントストリームを使わずポーリングする |
| `-flipsightapi <https_url>` | 接続するAPIのURLを変更する |

## 探索と通信

先後を入れ替えた2盤面の着手履歴が共通で、自分の手番が一方だけの場合は、一つの探索に全スレッドを使います。着手履歴が分かれた場合は、探索とponderにスレッドを分配します。両方が自分の手番になった場合は、二つの探索を同時に実行します。複数の対戦がある場合は、対戦間でもスレッドを分配します。各探索の呼び出し元には最低1スレッドが必要なので、`-thread 1` で2盤面を同時探索する場合も、呼び出し元は2本動きます。

レベル0〜10を指定した場合は、今後の着手と通信に必要な時間が残っていると見込めるときだけ、探索開始から送信まで最低1秒空けます。待機中も他の盤面の探索と送信は進みます。また、この起動プロセスのAPIリクエストを最小300ms間隔・直近60秒で200件以内に抑えます。HTTP 429の待機期限も、同じプロセス内で共有します。

イベントストリームで局面を受け取り、定期的なポーリングでも状態を確認します。送信結果が不明な場合は新しい状態を取得し、同じ手番・revision・局面である場合に限り、計算済みの同じ手を再送します。反則として拒否された手は、そのrevisionでは再送しません。トークンやサーバーの応答本文はログに出力しません。

## オフラインテスト

```powershell
cmake -S . -B build/flipsight -DINCLUDE_FLIPSIGHT=ON -DBUILD_FLIPSIGHT_TESTS=ON
cmake --build build/flipsight --config Release
ctest --test-dir build/flipsight -C Release --output-on-failure
```

通信形式、局面と時計、同時探索、ponder、送信結果が不明な場合の再送、送信間隔を検証します。実エンジンのテストには `bin/resources/eval.egev2` と `bin/resources/eval_move_ordering_end.egev` が必要です。テストはFlipSightへ接続しません。

参考: [issue #667](https://github.com/Nyanyan/Egaroucid/issues/667)、[公式API仕様](https://othelloclub.com/ja/flipsight/bot/api)。通信の復旧処理は、指定された `FlipSight_Bot` の実装も参考にしています。
