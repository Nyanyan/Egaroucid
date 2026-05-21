# Egaroucid ライブラリ機能 (実験的)

Egaroucid の探索エンジンを、GUI / Console とは独立したライブラリとして利用できます。  
この API は実験的機能です。将来のバージョンで変更される可能性があります。

## 紹介

- 最善手探索 API (`egaroucid_search_array`)
- 合法手生成 API (`egaroucid_get_legal_moves`)
- 返る石計算 API (`egaroucid_get_flipped_discs`)
- C ABI なので C/C++ 以外の FFI からも利用しやすい

公開ヘッダは <code>#include &lt;egaroucid/egaroucid.h&gt;</code>です。

## ビルド方法

<code>cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF</code>

<code>cmake --build build_lib --config Release</code>

## コードでの使い方

### 盤面の表現

- `board[64]` のインデックスは `0=a1, 1=b1, ..., 63=h8`
- マス値は `-1=空き`, `0=黒`, `1=白`
- 手番は `0=黒番`, `1=白番`

### 最小手順

1. <code>egaroucid_global_init(resource_dir)</code>
2. <code>egaroucid_create()</code>
3. <code>egaroucid_search_array(...)</code>
4. 必要なら <code>egaroucid_get_legal_moves(...)</code> と <code>egaroucid_get_flipped_discs(...)</code>
5. <code>egaroucid_destroy()</code>

### 呼び出し例

<code>#include &lt;egaroucid/egaroucid.h&gt;</code><br>
<code>egaroucid_global_init("bin/resources");</code><br>
<code>egaroucid_engine* engine = egaroucid_create();</code><br>
<code>egaroucid_search_options opt = {0};</code><br>
<code>opt.size = sizeof(opt); opt.level = 21; opt.use_book = 1;</code><br>
<code>egaroucid_search_result res = {0};</code><br>
<code>res.size = sizeof(res);</code><br>
<code>egaroucid_search_array(engine, board, EGAROUCID_BLACK, &opt, &res);</code><br>
<code>int legal[64], n_legal = 0;</code><br>
<code>egaroucid_get_legal_moves(board, EGAROUCID_BLACK, legal, &n_legal, NULL);</code><br>
<code>int flipped[64], n_flipped = 0;</code><br>
<code>egaroucid_get_flipped_discs(board, EGAROUCID_BLACK, res.move, flipped, &n_flipped, NULL);</code><br>
<code>egaroucid_destroy(engine);</code>

## 注意

- この API は実験的です。
- `time_limit_ms` は現在は受け付けるだけで未使用です。
- ライセンスは Egaroucid 本体と同じ GPL-3.0-or-later です。
