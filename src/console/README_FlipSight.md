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
.\bin\Release\Egaroucid_for_Console.out.exe -flipsight "<token>" -fsthread 4 -hash 24
```

`-fsthread M` は、サイトへ報告する盤面あたりのスレッド数です。省略時のMは4です。FlipSightモードのスレッド数は `-fsthread` で指定します。

`-level` を省略すると、FlipSightから受け取った残り時間に合わせて探索します。レベルを指定する場合は、同じコマンドに `-level 3` などを追加します。指定できるレベルは0〜60です。指定レベルの探索は、残り時間から通信の余裕を差し引いた時間を上限に実行します。定石ブックを使わず、指定レベルの探索だけで着手を選ぶ場合は `-nobook` も追加してください。

初期化後に対局の待ち受けを始め、対応する申し込みを自動で承諾します。Ctrl+Cで通信・探索・ponderを停止して終了します。

| 引数 | 動作 |
| --- | --- |
| `-flipsight <token>` | ボット用トークンを渡して接続する |
| `-fsthread M` | サイトへ盤面あたりMスレッドと報告する。起動時は2Mスレッドを確保し、対局の追加に応じて増員する（既定値4） |
| `-level N` | 指定レベルで探索する。省略時は残り時間に合わせる |
| `-flipsightnoponder` | 相手の手番中の先読みを無効にする |
| `-flipsightnostream` | イベントストリームを使わずポーリングする |
| `-flipsightapi <https_url>` | 接続するAPIのURLを変更する |

## 探索と通信

起動時には、探索用のスレッドを2M本確保します。同時対局に必要な本数が現在の確保数を超えたら、進行中の探索を止めずに増員します。必要な本数は、シンクロ対局1組につき2M本、通常対局1局につきM本です。対局が終了しても、確保したスレッドはそのまま保持します。

たとえばM=2の場合、起動時は4本です。シンクロ対局を2組同時に行うと8本になり、そこへ通常対局を1局追加すると10本になります。

シンクロ対局の2盤面の着手履歴が共通で、自分の手番が一方だけの場合は、その探索に2Mスレッドを割り当てます。着手履歴が分かれた場合や、両方が自分の手番の場合は、各盤面にMスレッドずつ割り当てて探索・ponderを行います。通常対局にはMスレッドを割り当てます。ほかの対局が増えても、この割り当ては維持します。

置換表は、通常対局なら1局の終了時、シンクロ対局なら両盤面の終了時に、対局単位で1回老化させます。同じ終了通知を再受信しても重複して老化させません。局面・評価値・着手のキャッシュを保持したまま、登録情報の重要度を下げます。老化は別のワーカーで実行し、ほかの対局の探索と通信を継続します。FlipSightモードでは、置換表の登録数に応じた自動老化の代わりに、この対局終了時の老化を使います。

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
