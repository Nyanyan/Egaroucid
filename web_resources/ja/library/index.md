# Egaroucid ライブラリ機能 (実験的)

Egaroucid の探索エンジンを、GUI / Console とは独立したライブラリとして利用できます。  
この API は実験的機能です。将来のバージョンで変更される可能性があります。

## 紹介
Egaroucidライブラリでは以下の機能を提供しています。

- 最善手探索 API (<code>egaroucid_search_array</code>)
- 合法手生成 API (<code>egaroucid_get_legal_moves</code>)
- 返る石計算 API (<code>egaroucid_get_flipped_discs</code>)
- C ABI なので C/C++ 以外の FFI からも利用しやすい

公開ヘッダは <code>#include &lt;egaroucid/egaroucid.h&gt;</code>です。

## サンプルコードの実行方法

まず、コードを入手してください。

<code class="code_block">git clone https://github.com/Nyanyan/Egaroucid.git<br>
cd Egaroucid</code>

ライブラリ利用サンプルは <code>examples/cpp/simple.cpp</code> にあります。  
公開ヘッダは <code>include/egaroucid/egaroucid.h</code> です。

以下のコマンドをリポジトリルートで順に実行してください。  
（GUI / Console はビルドせず、ライブラリとサンプルのみをビルドします）

<code class="code_block">cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF<br>
cmake --build build_lib --config Release --target egaroucid_cpp_example</code>

Windows (Visual Studio) の場合、以下のコマンドでサンプルコードを実行できます:
<code class="code_block">.\build_lib\examples\Release\egaroucid_cpp_example.exe</code>

Linux / macOS (Ninja / Makefiles) の場合、以下のコマンドでサンプルコードを実行できます:
<code class="code_block">./build_lib/examples/egaroucid_cpp_example</code>

このサンプルはEgaroucidレポジトリ内の<code>bin/resources</code>にあるリソースファイルを参照します。そのため、リポジトリのルートで実行してください。

## ライブラリの使い方

### 盤面の表現

- <code>board[64]</code> のインデックスは <code>0=a1, 1=b1, ..., 63=h8</code>
- マス値は <code>-1=空き</code>, <code>0=黒</code>, <code>1=白</code>
- 手番は <code>0=黒番</code>, <code>1=白番</code>

### 大まかな使い方

1. <code>egaroucid_global_init(resource_dir)</code>
2. <code>egaroucid_create()</code>
3. <code>egaroucid_search_array(...)</code>
4. 必要なら <code>egaroucid_get_legal_moves(...)</code> と <code>egaroucid_get_flipped_discs(...)</code>
5. <code>egaroucid_destroy()</code>

### サンプルコード

<code class="code_block">#include &lt;stdio.h&gt;<br>
#include &lt;string.h&gt;<br>
<br>
#include &lt;egaroucid/egaroucid.h&gt;<br>
<br>
static char cell_to_char(int cell) {<br>
    if (cell == EGAROUCID_BLACK) {<br>
        return 'X';<br>
    }<br>
    if (cell == EGAROUCID_WHITE) {<br>
        return 'O';<br>
    }<br>
    return '.';<br>
}<br>
<br>
static void print_board(const int board[64]) {<br>
    printf("  a b c d e f g h\n");<br>
    for (int row = 0; row &lt; 8; ++row) {<br>
        printf("%d ", row + 1);<br>
        for (int col = 0; col &lt; 8; ++col) {<br>
            printf("%c ", cell_to_char(board[row * 8 + col]));<br>
        }<br>
        printf("\n");<br>
    }<br>
}<br>
<br>
static void move_to_coord(int move, char coord_out[3]) {<br>
    if (move &lt; 0 || move &gt;= 64) {<br>
        coord_out[0] = '-';<br>
        coord_out[1] = '-';<br>
        coord_out[2] = '\0';<br>
        return;<br>
    }<br>
    coord_out[0] = (char)('a' + (move % 8));<br>
    coord_out[1] = (char)('1' + (move / 8));<br>
    coord_out[2] = '\0';<br>
}<br>
<br>
int main(void) {<br>
    egaroucid_status st = egaroucid_global_init("bin/resources");<br>
    if (st != EGAROUCID_OK) {<br>
        printf("egaroucid_global_init failed: %d\n", st);<br>
        return 1;<br>
    }<br>
<br>
    egaroucid_engine *engine = egaroucid_create();<br>
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
    printf("initial board:\n");<br>
    print_board(board);<br>
<br>
    egaroucid_search_options opt;<br>
    memset(&amp;opt, 0, sizeof(opt));<br>
    opt.size = sizeof(opt);<br>
    opt.level = 21;<br>
    opt.use_book = 1;<br>
    opt.book_accuracy_level = 0;<br>
    opt.use_multi_thread = 1;<br>
    opt.show_log = 0;<br>
    opt.time_limit_ms = -1; /* currently ignored */<br>
<br>
    egaroucid_search_result res;<br>
    memset(&amp;res, 0, sizeof(res));<br>
    res.size = sizeof(res);<br>
<br>
    st = egaroucid_search_array(engine, board, EGAROUCID_BLACK, &amp;opt, &amp;res);<br>
    if (st != EGAROUCID_OK) {<br>
        printf("egaroucid_search_array failed: %d\n", st);<br>
        egaroucid_destroy(engine);<br>
        return 1;<br>
    }<br>
<br>
    char best_move_coord[3];<br>
    move_to_coord(res.move, best_move_coord);<br>
    printf(<br>
        "best move=%s (%d) value=%d depth=%d nodes=%llu\n",<br>
        best_move_coord,<br>
        res.move,<br>
        res.value,<br>
        res.depth,<br>
        (unsigned long long)res.nodes<br>
    );<br>
<br>
    int legal[64];<br>
    int n_legal = 0;<br>
    st = egaroucid_get_legal_moves(board, EGAROUCID_BLACK, legal, &amp;n_legal, NULL);<br>
    if (st == EGAROUCID_OK) {<br>
        printf("n_legal=%d\n", n_legal);<br>
    }<br>
<br>
    if (res.move &gt;= 0) {<br>
        int flipped[64];<br>
        int n_flipped = 0;<br>
        st = egaroucid_get_flipped_discs(board, EGAROUCID_BLACK, res.move, flipped, &amp;n_flipped, NULL);<br>
        if (st == EGAROUCID_OK) {<br>
            printf("n_flipped=%d\n", n_flipped);<br>
            if (n_flipped &gt; 0) {<br>
                board[res.move] = EGAROUCID_BLACK;<br>
                for (int i = 0; i &lt; n_flipped; ++i) {<br>
                    board[flipped[i]] = EGAROUCID_BLACK;<br>
                }<br>
                printf("board after best move:\n");<br>
                print_board(board);<br>
            }<br>
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
