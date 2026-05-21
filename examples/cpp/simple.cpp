#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <egaroucid/egaroucid.h>

namespace {

constexpr int kBoardSize = 8;
constexpr int kBoardCells = 64;

inline int opponent_of(int player) {
    return (player == EGAROUCID_BLACK) ? EGAROUCID_WHITE : EGAROUCID_BLACK;
}

inline int index_of(int row, int col) {
    return row * kBoardSize + col;
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

        int legal_moves[kBoardCells];
        int n_legal_moves = 0;
        uint64_t legal_mask = 0ULL;
        status = egaroucid_get_legal_moves(board, player, legal_moves, &n_legal_moves, &legal_mask);
        if (status != EGAROUCID_OK) {
            std::printf("egaroucid_get_legal_moves failed: %d\n", status);
            egaroucid_destroy(engine);
            return 1;
        }
        std::printf("legal moves=%d legal_mask=0x%016" PRIx64 "\n", n_legal_moves, legal_mask);

        if (n_legal_moves == 0) {
            int opponent_legal_count = 0;
            status = egaroucid_get_legal_moves(board, opponent_of(player), nullptr, &opponent_legal_count, nullptr);
            if (status != EGAROUCID_OK) {
                std::printf("egaroucid_get_legal_moves(opponent) failed: %d\n", status);
                egaroucid_destroy(engine);
                return 1;
            }
            if (opponent_legal_count == 0) {
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

        int flipped[kBoardCells];
        int n_flipped = 0;
        uint64_t flipped_mask = 0ULL;
        status = egaroucid_get_flipped_discs(board, player, result.move, flipped, &n_flipped, &flipped_mask);
        if (status != EGAROUCID_OK) {
            std::printf("egaroucid_get_flipped_discs failed: %d\n", status);
            egaroucid_destroy(engine);
            return 1;
        }

        std::printf(
            "engine move=%s (%d) flips=%d flip_mask=0x%016" PRIx64 " value=%d depth=%d nodes=%" PRIu64 " nps=%.0f end=%d\n",
            move_to_coord(result.move).c_str(),
            result.move,
            n_flipped,
            flipped_mask,
            result.value,
            result.depth,
            result.nodes,
            result.nps,
            result.is_end_search
        );

        if (result.move < 0 || result.move >= kBoardCells || n_flipped <= 0) {
            std::printf("library returned an illegal/non-playable move: %d\n", result.move);
            egaroucid_destroy(engine);
            return 1;
        }

        board[result.move] = player;
        for (int i = 0; i < n_flipped; ++i) {
            board[flipped[i]] = player;
        }
        player = opponent_of(player);
    }

    std::printf("\nFinal board:\n");
    print_board(board);
    egaroucid_destroy(engine);
    return 0;
}
