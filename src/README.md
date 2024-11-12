# Egaroucid Source Code

Egaroucidのソースコードです



## ラインナップ

* Egaroucid
  * GUIを搭載したオセロAIソフト
  * メインのアプリ
* Egaroucid for Console
  * コンソールアプリ
  * Egaroucidと同じ思考エンジン
* Egaroucid for Web
  * Web版
  * Wasmとして使用する
  * 簡易的な思考エンジン
* Egaroucid Light
  * コンソールアプリ
  * Web版と同じ思考エンジン
  * 動作確認用

詳細は[公式サイト](https://www.egaroucid.nyanyan.dev/ja/)をご覧ください。



## ビルド方法

### Egaroucid

Windowsでのビルドを想定しています。

* 必要なものを準備する

  * [VisualStudio](https://visualstudio.microsoft.com/ja/)

  * [Siv3Dライブラリ](https://siv3d.github.io/ja-jp/)

* ```Main.cpp```をビルドする

  * 以下のフォルダ内のコードをインクルードしている
    * ```engine```
    * ```gui```
  * C++17以上でビルドする必要がある

### Egaroucid for Console

様々なプラットフォームでビルドできます

* 必要なものを準備する
  * ```g++```か```cmake```
* ```Egaroucid_console.cpp```をビルドする
  * 詳細: https://www.egaroucid.nyanyan.dev/ja/console/
  * 以下のフォルダ内のコードをインクルードしている
    * ```engine```
    * ```console```

### Egaroucid for Web

Wasmとして使います。

* 必要なものを準備する

  * ```em++```
    * バージョン3.1.20で動作確認済み

* ```Egaroucid_web.cpp```をビルドする

  * コマンドは以下

  * ```
    em++ Egaroucid_web.cpp -s WASM=1 -o ai.js -s "EXPORTED_FUNCTIONS=['_init_ai', '_ai_js', '_calc_value', '_stop', '_resume', '_malloc', '_free']" -O3 -s TOTAL_MEMORY=629145600 -s ALLOW_MEMORY_GROWTH=1
    ```

  * 以下のフォルダ内のコードをインクルードしている
    * ```web```
  
  * ```ai.js```と```ai.wasm```をJavaScriptなどから呼び出して使う

### Egaroucid Light

Egaroucid for Webの思考エンジンをそのまま流用したもの

* 必要なものを準備する
  * ```g++```
* ```Egaroucid_light.cpp```をビルドする

開発者自身動作確認用途でしか使っていないので、特にサポートはありません。



## 各フォルダの説明

* ```console```
  * Egaroucid for ConsoleのUI関連
* ```engine```
  * EgaroucidおよびEgaroucid for Consoleの思考エンジン
* ```gui```
  * EgaroucidのGUI関連
* ```tools```
  * Egaroucid制作時に使用したツール
  * 評価関数の学習に使うコードなどはここに入っている
* ```web```
  * Egaroucid for WebおよびEgaroucid Lightのコード



## 参考資料

### Egaroucid

[公式サイト](https://www.egaroucid.nyanyan.dev/ja/)をご覧ください。

[使い方](https://www.egaroucid.nyanyan.dev/ja/usage/)も公開しています。

### Egaroucid for Console

[公式サイトの専用ページ](https://www.egaroucid.nyanyan.dev/ja/console/)をご覧ください。

以下のコマンドで使用できるオプションやコマンドを確認できます。

```
$ Egaroucid_for_console.exe -help
```

### Egaroucid for Web

[遊べるページ](https://www.egaroucid.nyanyan.dev/ja/web/)で公開しています。

各関数の説明は以下の通りです。

* ```int _init_ai()```
  * initialize Egaroucid for Web.
  * always returns 0

* ```int _ai_js(int arr_board[64], int level, int ai_player)```
  * calculate the best move. 
  * ```arr_board```: array representing a board
    * ```0``` for black disc
    * ```1``` for white disc
    * ```-1``` for empty square
  * ```level```: Egaroucid's level
    * ```0``` to ```15```
  * ```ai_player```: Who will put a disc?
    * ```0``` for black
    * ```1``` for white
  * returns ```coord * 1000 + value```
    * ```coord```: ```0``` for h8, ```1``` for g8, ..., ```63``` for a1
    * ```value```: ```-64``` to ```64```, estimated Egaroucid's score
* ```void _calc_value(int arr_board[64], int res[74], int level, int ai_player)```
  * calculate score of each legal moves
  * ```arr_board```: array representing a board
    * ```0``` for black disc
    * ```1``` for white disc
    * ```-1``` for empty square
  * ```res```: array to store result
    * ```res[10 + coord] == value for the coord```
  * ```level```: Egaroucid's level
    * ```0``` to ```15```
  * ```ai_player```: Who will put a disc?
    * ```0``` for black
    * ```1``` for white
* ```void _stop()```
  * stop all calculation
* ```void _resume()```
  * restart all calculation

