#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <egaroucid/egaroucid.h>

namespace {

constexpr int kBoardSize = 8;
constexpr int kBoardCells = 64;

constexpr int kDirections[8][2] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    {0, -1},           {0, 1},
    {1, -1},  {1, 0},  {1, 1}
};

inline int opponent_of(int player) {
    return (player == EGAROUCID_BLACK) ? EGAROUCID_WHITE : EGAROUCID_BLACK;
}

inline int index_of(int row, int col) {
    return row * kBoardSize + col;
}

inline bool in_bounds(int row, int col) {
    return 0 <= row && row < kBoardSize && 0 <= col && col < kBoardSize;
}

char cell_to_char(int cell) {
    if (cell == EGAROUCID_BLACK) {
        return 'X';
    }
    if (cell == EGAROUCID_WHITE) {
        return 'O';
    }
    return '.';
}

std::string move_to_coord(int move) {
    if (move < 0 || move >= kBoardCells) {
        return "pass";
    }
    const char file = static_cast<char>('a' + (move % kBoardSize));
    const char rank = static_cast<char>('1' + (move / kBoardSize));
    std::string coord;
    coord.push_back(file);
    coord.push_back(rank);
    return coord;
}

void print_board(const int board[kBoardCells]) {
    std::printf("  a b c d e f g h\n");
    for (int row = kBoardSize - 1; row >= 0; --row) {
        std::printf("%d ", row + 1);
        for (int col = 0; col < kBoardSize; ++col) {
            std::printf("%c ", cell_to_char(board[index_of(row, col)]));
        }
        std::printf("\n");
    }
}

int collect_flips(const int board[kBoardCells], int player, int move, int flipped[kBoardCells]) {
    if (move < 0 || move >= kBoardCells || board[move] != EGAROUCID_EMPTY) {
        return 0;
    }
    const int opponent = opponent_of(player);
    const int move_row = move / kBoardSize;
    const int move_col = move % kBoardSize;
    int n_flipped = 0;

    for (const auto &dir : kDirections) {
        int row = move_row + dir[0];
        int col = move_col + dir[1];
        int candidate[kBoardCells];
        int n_candidate = 0;

        while (in_bounds(row, col)) {
            const int idx = index_of(row, col);
            if (board[idx] == opponent) {
                candidate[n_candidate++] = idx;
                row += dir[0];
                col += dir[1];
                continue;
            }
            if (board[idx] == player && n_candidate > 0) {
                for (int i = 0; i < n_candidate; ++i) {
                    flipped[n_flipped++] = candidate[i];
                }
            }
            break;
        }
    }
    return n_flipped;
}

bool has_legal_move(const int board[kBoardCells], int player) {
    int flipped[kBoardCells];
    for (int move = 0; move < kBoardCells; ++move) {
        if (collect_flips(board, player, move, flipped) > 0) {
            return true;
        }
    }
    return false;
}

bool apply_move(int board[kBoardCells], int player, int move) {
    int flipped[kBoardCells];
    const int n_flipped = collect_flips(board, player, move, flipped);
    if (n_flipped <= 0) {
        return false;
    }
    board[move] = player;
    for (int i = 0; i < n_flipped; ++i) {
        board[flipped[i]] = player;
    }
    return true;
}

const char *player_name(int player) {
    return (player == EGAROUCID_BLACK) ? "black" : "white";
}

}  // namespace

int main(int argc, char **argv) {
    const char *resource_dir = (argc >= 2) ? argv[1] : "bin/resources";
    const int max_plies = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 10;

    egaroucid_status status = egaroucid_global_init(resource_dir);
    if (status != EGAROUCID_OK) {
        std::printf("egaroucid_global_init failed: %d (resource_dir=%s)\n", status, resource_dir);
        return 1;
    }

    egaroucid_engine *engine = egaroucid_create();
    if (engine == nullptr) {
        std::printf("egaroucid_create failed\n");
        return 1;
    }

    int board[kBoardCells];
    for (int i = 0; i < kBoardCells; ++i) {
        board[i] = EGAROUCID_EMPTY;
    }
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

    int player = EGAROUCID_BLACK;
    for (int ply = 1; ply <= max_plies; ++ply) {
        std::printf("\nPly %d (to move: %s)\n", ply, player_name(player));
        print_board(board);

        const bool can_move = has_legal_move(board, player);
        if (!can_move) {
            const bool opponent_can_move = has_legal_move(board, opponent_of(player));
            if (!opponent_can_move) {
                std::printf("Both players cannot move. Game over.\n");
                break;
            }
            std::printf("%s passes.\n", player_name(player));
            player = opponent_of(player);
            continue;
        }

        egaroucid_search_result result{};
        result.size = sizeof(result);
        status = egaroucid_search_array(engine, board, player, &options, &result);
        if (status != EGAROUCID_OK) {
            std::printf("egaroucid_search_array failed: %d\n", status);
            egaroucid_destroy(engine);
            return 1;
        }

        std::printf(
            "engine move=%s (%d) value=%d depth=%d nodes=%" PRIu64 " nps=%.0f end=%d\n",
            move_to_coord(result.move).c_str(),
            result.move,
            result.value,
            result.depth,
            result.nodes,
            result.nps,
            result.is_end_search
        );

        if (!apply_move(board, player, result.move)) {
            std::printf("sample-side move application failed for move=%d\n", result.move);
            egaroucid_destroy(engine);
            return 1;
        }
        player = opponent_of(player);
    }

    std::printf("\nFinal board:\n");
    print_board(board);
    egaroucid_destroy(engine);
    return 0;
}
