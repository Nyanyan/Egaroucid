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

Windows、MaxOS、Linuxでビルドできます。

#### Windows

* 必要なものを準備する

  * [VisualStudio](https://visualstudio.microsoft.com/ja/)

  * [Siv3Dライブラリ](https://siv3d.github.io/ja-jp/)

* ```Main.cpp```をビルドする

  * 以下のフォルダ内のコードをインクルードしている
    * ```engine```
    * ```gui```
  * C++17以上でビルドする必要がある

#### MacOS

GitHubコントリビューターによってMac版のコンパイルが可能になりました。詳細な手順はそのうち追記します。

#### Linux

GitHubコントリビューターによってLinux版のコンパイルが可能になりました。

cmakeコマンドで、```-DBUILD_GUI=ON```オプションをつけてビルドことができます。ArchLinux環境にて、g++ 15.2.1,  clang 17.0.6で動作を確認済みとのことです。

- ユーザが自分でOpenSiv3Dをビルドしてインストールする。方法は[Siv3D公式サイト](https://siv3d.github.io/ja-jp/download/ubuntu/)の記述を参照のこと
- Egaroucidをgit cloneする
- ```cmake -S . -B build -DBUILD_GUI=ON```
- ```cmake --build build```

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

公開APIとして用意した各関数の説明は以下の通りです。

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



## Egaroucid Engine Library (Experimental)

### Build (CMake)

```bash
cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF
cmake --build build_lib --config Release
```

### Public Header

```c
#include <egaroucid/egaroucid.h>
```

### Board Convention

- `board[64]`: `0=a1`, `1=b1`, ..., `63=h8`
- cell value: `-1` empty, `0` black, `1` white
- player: `0` black to move, `1` white to move

### Minimal Usage (C)

```c
#include <stdio.h>
#include <string.h>
#include <egaroucid/egaroucid.h>

int main(void) {
    if (egaroucid_global_init("bin/resources") != EGAROUCID_OK) {
        return 1;
    }

    egaroucid_engine* engine = egaroucid_create();
    if (!engine) {
        return 1;
    }

    int board[64];
    for (int i = 0; i < 64; ++i) {
        board[i] = EGAROUCID_EMPTY;
    }
    board[27] = EGAROUCID_WHITE; /* d4 */
    board[28] = EGAROUCID_BLACK; /* e4 */
    board[35] = EGAROUCID_BLACK; /* d5 */
    board[36] = EGAROUCID_WHITE; /* e5 */

    egaroucid_search_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.size = sizeof(opt);
    opt.level = 21;
    opt.use_book = 1;
    opt.use_multi_thread = 1;
    opt.time_limit_ms = -1; /* currently ignored */

    egaroucid_search_result res;
    memset(&res, 0, sizeof(res));
    res.size = sizeof(res);

    if (egaroucid_search_array(engine, board, EGAROUCID_BLACK, &opt, &res) == EGAROUCID_OK) {
        printf("best move=%d value=%d\n", res.move, res.value);
    }

    int legal_moves[64];
    int n_legal = 0;
    if (egaroucid_get_legal_moves(board, EGAROUCID_BLACK, legal_moves, &n_legal, NULL) == EGAROUCID_OK) {
        printf("n_legal=%d\n", n_legal);
    }

    if (res.move >= 0) {
        int flipped[64];
        int n_flipped = 0;
        if (egaroucid_get_flipped_discs(board, EGAROUCID_BLACK, res.move, flipped, &n_flipped, NULL) == EGAROUCID_OK) {
            printf("n_flipped=%d\n", n_flipped);
        }
    }

    egaroucid_destroy(engine);
    return 0;
}
```

