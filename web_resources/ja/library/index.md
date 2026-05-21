# Egaroucid ライブラリ機能 (実験的)

Egaroucid の探索エンジンを、GUI / Console とは独立したライブラリとして利用できます。  
この API は実験的機能です。将来のバージョンで変更される可能性があります。

## 紹介

- 最善手探索 API (<code>egaroucid_search_array</code>)
- 合法手生成 API (<code>egaroucid_get_legal_moves</code>)
- 返る石計算 API (<code>egaroucid_get_flipped_discs</code>)
- C ABI なので C/C++ 以外の FFI からも利用しやすい

公開ヘッダは <code>#include &lt;egaroucid/egaroucid.h&gt;</code>です。

## ビルド方法

<code>cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF</code>

<code>cmake --build build_lib --config Release</code>

## コードでの使い方

### 盤面の表現

- <code>board[64]</code> のインデックスは <code>0=a1, 1=b1, ..., 63=h8</code>
- マス値は <code>-1=空き</code>, <code>0=黒</code>, <code>1=白</code>
- 手番は <code>0=黒番</code>, <code>1=白番</code>

### 最小手順

1. <code>egaroucid_global_init(resource_dir)</code>
2. <code>egaroucid_create()</code>
3. <code>egaroucid_search_array(...)</code>
4. 必要なら <code>egaroucid_get_legal_moves(...)</code> と <code>egaroucid_get_flipped_discs(...)</code>
5. <code>egaroucid_destroy()</code>

### 呼び出し例

<code>#include &lt;stdio.h&gt;<br>
#include &lt;string.h&gt;<br>
#include &lt;egaroucid/egaroucid.h&gt;<br>
<br>
int main(void) {<br>
    egaroucid_status st = egaroucid_global_init("bin/resources");<br>
    if (st != EGAROUCID_OK) {<br>
        printf("egaroucid_global_init failed: %d\n", st);<br>
        return 1;<br>
    }<br>
<br>
    egaroucid_engine* engine = egaroucid_create();<br>
    if (engine == NULL) {<br>
        printf("egaroucid_create failed\n");<br>
        return 1;<br>
    }<br>
<br>
    int board[64];<br>
    for (int i = 0; i &lt; 64; ++i) {<br>
        board[i] = EGAROUCID_EMPTY;<br>
    }<br>
    board[27] = EGAROUCID_WHITE; /* d4 */<br>
    board[28] = EGAROUCID_BLACK; /* e4 */<br>
    board[35] = EGAROUCID_BLACK; /* d5 */<br>
    board[36] = EGAROUCID_WHITE; /* e5 */<br>
<br>
    egaroucid_search_options opt;<br>
    memset(&opt, 0, sizeof(opt));<br>
    opt.size = sizeof(opt);<br>
    opt.level = 21;<br>
    opt.use_book = 1;<br>
    opt.book_accuracy_level = 0;<br>
    opt.use_multi_thread = 1;<br>
    opt.show_log = 0;<br>
    opt.time_limit_ms = -1; /* 現在は未使用 */<br>
<br>
    egaroucid_search_result res;<br>
    memset(&res, 0, sizeof(res));<br>
    res.size = sizeof(res);<br>
<br>
    st = egaroucid_search_array(engine, board, EGAROUCID_BLACK, &opt, &res);<br>
    if (st != EGAROUCID_OK) {<br>
        printf("egaroucid_search_array failed: %d\n", st);<br>
        egaroucid_destroy(engine);<br>
        return 1;<br>
    }<br>
<br>
    printf("best move=%d value=%d depth=%d nodes=%llu\n",<br>
           res.move, res.value, res.depth, (unsigned long long)res.nodes);<br>
<br>
    int legal[64];<br>
    int n_legal = 0;<br>
    st = egaroucid_get_legal_moves(board, EGAROUCID_BLACK, legal, &n_legal, NULL);<br>
    if (st == EGAROUCID_OK) {<br>
        printf("n_legal=%d\n", n_legal);<br>
    }<br>
<br>
    if (res.move &gt;= 0) {<br>
        int flipped[64];<br>
        int n_flipped = 0;<br>
        st = egaroucid_get_flipped_discs(board, EGAROUCID_BLACK, res.move, flipped, &n_flipped, NULL);<br>
        if (st == EGAROUCID_OK) {<br>
            printf("n_flipped=%d\n", n_flipped);<br>
        }<br>
    }<br>
<br>
    egaroucid_destroy(engine);<br>
    return 0;<br>
}</code>

## 注意

- この API は実験的です。
- <code>time_limit_ms</code> は現在は受け付けるだけで未使用です。
- ライセンスは Egaroucid 本体と同じ GPL-3.0-or-later です。
