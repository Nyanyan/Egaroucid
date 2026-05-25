# Egaroucid Library Feature (Experimental)

You can use Egaroucid's search engine as a standalone library independent from GUI / Console.  
This API is experimental and may change in future versions.

## Introduction
The Egaroucid library provides the following features:

- Best-move search API (<code>egaroucid_search_array</code>)
- Legal move generation API (<code>egaroucid_get_legal_moves</code>)
- Flipped-disc calculation API (<code>egaroucid_get_flipped_discs</code>)

The public header is <code>#include &lt;egaroucid/egaroucid.h&gt;</code>.

## How to Use

First, get the source code.

<code class="code_block">git clone https://github.com/Nyanyan/Egaroucid.git<br>
cd Egaroucid</code>

The library sample code is in <code>examples/cpp/simple.cpp</code>.  
The public header is <code>include/egaroucid/egaroucid.h</code>.

Run the following commands in the repository root in order.  
(GUI / Console are not built, and only the library feature is enabled.)

<code class="code_block">cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF</code>

By default, this builds a static library. If you need a shared library with SONAME for Linux packaging, add <code>-DBUILD_SHARED_LIBS=ON</code>. For shared libraries, CMake sets <code>VERSION</code> from the Egaroucid version and <code>SOVERSION</code> from the major version.

<code class="code_block">cmake -S . -B build_lib -DBUILD_ENGINE_LIB=ON -DBUILD_SHARED_LIBS=ON -DBUILD_CONSOLE=OFF -DBUILD_GUI=OFF</code>

If you only want to build the library:
<code class="code_block">cmake --build build_lib --config Release --target egaroucid</code>

If you want to run the sample code (build the sample executable too):
<code class="code_block">cmake --build build_lib --config Release --target egaroucid_cpp_example</code>

Using only <code>--target egaroucid</code> does not generate the sample executable.  
If you specify <code>--target egaroucid_cpp_example</code>, the dependent <code>egaroucid</code> library is also built.

For Windows (Visual Studio), run the sample with:
<code class="code_block">.\build_lib\examples\Release\egaroucid_cpp_example.exe</code>

For Linux / macOS (Ninja / Makefiles), run the sample with:
<code class="code_block">./build_lib/examples/egaroucid_cpp_example</code>

This sample refers to resource files in <code>bin/resources</code> inside the Egaroucid repository.  
So please run it from the repository root.

## Library Usage

### Board Representation

- The index of <code>board[64]</code> is <code>0=a1, 1=b1, ..., 63=h8</code>
- Cell value is <code>-1=empty</code>, <code>0=black</code>, <code>1=white</code>
- Player value is <code>0=black to move</code>, <code>1=white to move</code>

### High-Level Usage

1. <code>egaroucid_global_init(resource_dir)</code>
2. <code>egaroucid_create()</code>
3. <code>egaroucid_search_array(...)</code>
4. If needed, <code>egaroucid_get_legal_moves(...)</code> and <code>egaroucid_get_flipped_discs(...)</code>
5. <code>egaroucid_destroy()</code>

### Sample Code

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

## Notes

- This API is experimental.
- <code>time_limit_ms</code> is currently accepted but unused.
- The license is GPL-3.0-or-later, same as Egaroucid itself.
