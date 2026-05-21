#include <cinttypes>
#include <cstdio>

#include <egaroucid/egaroucid.h>

int main() {
    egaroucid_status status = egaroucid_global_init("bin/resources");
    if (status != EGAROUCID_OK) {
        std::printf("egaroucid_global_init failed: %d\n", status);
        return 1;
    }

    egaroucid_engine *engine = egaroucid_create();
    if (engine == nullptr) {
        std::printf("egaroucid_create failed\n");
        return 1;
    }

    int board[64];
    for (int i = 0; i < 64; ++i) {
        board[i] = EGAROUCID_EMPTY;
    }
    // Initial Othello position in 0-based index (0=a1, 63=h8).
    board[27] = EGAROUCID_WHITE;  // d4
    board[28] = EGAROUCID_BLACK;  // e4
    board[35] = EGAROUCID_BLACK;  // d5
    board[36] = EGAROUCID_WHITE;  // e5

    egaroucid_search_options options{};
    options.size = sizeof(options);
    options.level = 21;
    options.use_book = 1;
    options.book_accuracy_level = 0;
    options.use_multi_thread = 1;
    options.show_log = 0;
    options.time_limit_ms = -1;  // currently ignored

    egaroucid_search_result result{};
    result.size = sizeof(result);
    status = egaroucid_search_array(engine, board, EGAROUCID_BLACK, &options, &result);
    if (status != EGAROUCID_OK) {
        std::printf("egaroucid_search_array failed: %d\n", status);
        egaroucid_destroy(engine);
        return 1;
    }

    std::printf(
        "move=%d value=%d depth=%d nodes=%" PRIu64 " nps=%.0f end=%d\n",
        result.move,
        result.value,
        result.depth,
        result.nodes,
        result.nps,
        result.is_end_search
    );

    egaroucid_destroy(engine);
    return 0;
}
