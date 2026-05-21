#include <stdio.h>
#include <string.h>

#include <egaroucid/egaroucid.h>

static char cell_to_char(int cell) {
    if (cell == EGAROUCID_BLACK) {
        return 'X';
    }
    if (cell == EGAROUCID_WHITE) {
        return 'O';
    }
    return '.';
}

static void print_board(const int board[64]) {
    printf("  a b c d e f g h\n");
    for (int row = 0; row < 8; ++row) {
        printf("%d ", row + 1);
        for (int col = 0; col < 8; ++col) {
            printf("%c ", cell_to_char(board[row * 8 + col]));
        }
        printf("\n");
    }
}

static void move_to_coord(int move, char coord_out[3]) {
    if (move < 0 || move >= 64) {
        coord_out[0] = '-';
        coord_out[1] = '-';
        coord_out[2] = '\0';
        return;
    }
    coord_out[0] = (char)('a' + (move % 8));
    coord_out[1] = (char)('1' + (move / 8));
    coord_out[2] = '\0';
}

int main(void) {
    egaroucid_status st = egaroucid_global_init("bin/resources");
    if (st != EGAROUCID_OK) {
        printf("egaroucid_global_init failed: %d\n", st);
        return 1;
    }

    egaroucid_engine *engine = egaroucid_create();
    if (engine == NULL) {
        printf("egaroucid_create failed\n");
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

    printf("initial board:\n");
    print_board(board);

    egaroucid_search_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.size = sizeof(opt);
    opt.level = 21;
    opt.use_book = 1;
    opt.book_accuracy_level = 0;
    opt.use_multi_thread = 1;
    opt.show_log = 0;
    opt.time_limit_ms = -1; /* currently ignored */

    egaroucid_search_result res;
    memset(&res, 0, sizeof(res));
    res.size = sizeof(res);

    st = egaroucid_search_array(engine, board, EGAROUCID_BLACK, &opt, &res);
    if (st != EGAROUCID_OK) {
        printf("egaroucid_search_array failed: %d\n", st);
        egaroucid_destroy(engine);
        return 1;
    }

    char best_move_coord[3];
    move_to_coord(res.move, best_move_coord);
    printf(
        "best move=%s (%d) value=%d depth=%d nodes=%llu\n",
        best_move_coord,
        res.move,
        res.value,
        res.depth,
        (unsigned long long)res.nodes
    );

    int legal[64];
    int n_legal = 0;
    st = egaroucid_get_legal_moves(board, EGAROUCID_BLACK, legal, &n_legal, NULL);
    if (st == EGAROUCID_OK) {
        printf("n_legal=%d\n", n_legal);
    }

    if (res.move >= 0) {
        int flipped[64];
        int n_flipped = 0;
        st = egaroucid_get_flipped_discs(board, EGAROUCID_BLACK, res.move, flipped, &n_flipped, NULL);
        if (st == EGAROUCID_OK) {
            printf("n_flipped=%d\n", n_flipped);
            if (n_flipped > 0) {
                board[res.move] = EGAROUCID_BLACK;
                for (int i = 0; i < n_flipped; ++i) {
                    board[flipped[i]] = EGAROUCID_BLACK;
                }
                printf("board after best move:\n");
                print_board(board);
            }
        }
    }

    egaroucid_destroy(engine);
    return 0;
}
