/*
    Egaroucid Project

    Regression tests for choosing which synchro game to ponder after our move.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DINCLUDE_GGS -DIS_GGS_TOURNAMENT test_ggs_ponder_slot.cpp \
            -o test_ggs_ponder_slot.exe -lws2_32
*/

#include <iostream>
#include <stdexcept>
#include <string>
#include "../../engine/engine_all.hpp"
#include "../../console/console_all.hpp"

namespace {

const std::string ME = "egrcd";
const std::string OPPONENT = "nyanyan";
// 14 discs, the first position of the 2026-10-05 match .68.
const std::string START_BOARD = "------------------OOXX----XOX-----XOOX----XXX------------------- X";
// 62 discs, from the end of the same match.
const std::string LATE_BOARD = "XXXXXXXX-XXOOOXOOOXOXXOOOOOXOOXOOXOOOOXOOOXOOOXOO-XXXXOOOOOOOOOO X";
const std::string FULL_BOARD = std::string(64, 'X') + " X";

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A synchro game where black is to move. our_turn decides who plays black.
GGS_Board make_game(const std::string &game_id, int synchro_id, const std::string &board_text, bool our_turn) {
    GGS_Board ggs_board;
    std::pair<Board, int> board_player = convert_board_from_str(board_text);
    require(board_player.second == BLACK, "test boards must have black to move: " + board_text);
    ggs_board.game_id = game_id;
    ggs_board.is_synchro = true;
    ggs_board.synchro_id = synchro_id;
    ggs_board.board = board_player.first;
    ggs_board.player_to_move = BLACK;
    ggs_board.player_black = our_turn ? ME : OPPONENT;
    ggs_board.player_white = our_turn ? OPPONENT : ME;
    return ggs_board;
}

struct Slots {
    GGS_Board boards[2][HW2 + 1];
    int n_discs[2] = {0, 0};
    bool ai_searchings[2] = {false, false};

    void store(const GGS_Board &ggs_board) {
        const int n = ggs_board.board.n_discs();
        boards[ggs_board.synchro_id][n] = ggs_board;
        n_discs[ggs_board.synchro_id] = n;
    }

    int choose(const GGS_Board &update) const {
        return ggs_synchro_ponder_slot(update, boards, n_discs, ai_searchings, ME);
    }
};

void test_opponent_thinking_in_other_game() {
    Slots slots;
    // Our move in .68.0 was just accepted; the opponent is still thinking in .68.1.
    const GGS_Board update = make_game(".68.0", 0, START_BOARD, false);
    slots.store(update);
    slots.store(make_game(".68.1", 1, START_BOARD, false));
    require(slots.choose(update) == 1, "ponder the game where the opponent is thinking");
}

void test_our_turn_in_other_game() {
    Slots slots;
    // The opponent's move in .68.1 arrived in the same batch, so our search there starts next.
    const GGS_Board update = make_game(".68.0", 0, START_BOARD, false);
    slots.store(update);
    slots.store(make_game(".68.1", 1, START_BOARD, true));
    require(slots.choose(update) == 0, "keep the updated game when it is our turn in the other game");
}

void test_other_game_not_usable() {
    const GGS_Board update = make_game(".68.0", 0, START_BOARD, false);
    {
        Slots slots;
        slots.store(update);
        require(slots.choose(update) == 0, "no board stored for the other game");
    }
    {
        Slots slots;
        slots.store(update);
        slots.store(make_game(".67.1", 1, START_BOARD, false));
        require(slots.choose(update) == 0, "the other slot holds a game of another match");
    }
    {
        Slots slots;
        slots.store(update);
        slots.store(make_game(".68.1", 1, FULL_BOARD, false));
        require(slots.choose(update) == 0, "the other game has ended");
    }
    {
        Slots slots;
        slots.store(update);
        slots.store(make_game(".68.1", 1, START_BOARD, false));
        slots.ai_searchings[1] = true;
        require(slots.choose(update) == 0, "a search is running in the other game");
    }
    {
        Slots slots;
        GGS_Board non_synchro = make_game(".68", 0, START_BOARD, false);
        non_synchro.is_synchro = false;
        non_synchro.synchro_id = -1;
        require(slots.choose(non_synchro) == -1, "a non-synchro game keeps its own slot");
    }
}

// The position after our first legal move from board_text, with white (the
// opponent) to move in a game where we play black.
GGS_Board make_game_after_our_move(const std::string &game_id, int synchro_id, const std::string &board_text) {
    GGS_Board ggs_board = make_game(game_id, synchro_id, board_text, true);
    uint64_t legal = ggs_board.board.get_legal();
    require(legal != 0ULL, "test board must have a legal move: " + board_text);
    Flip flip;
    calc_flip(&flip, &ggs_board.board, first_bit(&legal));
    ggs_board.board.move_board(&flip);
    ggs_board.player_to_move = WHITE;
    return ggs_board;
}

void test_identical_games_keep_updated_game() {
    // We moved from START_BOARD in .68.0, and the opponent faces the same
    // position in .68.1. If it plays our move, our next position equals the
    // one we just reached in .68.0, so keep pondering .68.0.
    Slots slots;
    slots.store(make_game(".68.0", 0, START_BOARD, true));
    const GGS_Board update = make_game_after_our_move(".68.0", 0, START_BOARD);
    slots.store(update);
    slots.store(make_game(".68.1", 1, START_BOARD, false));
    require(slots.choose(update) == 0, "identical games keep pondering the game we just moved in");
}

void test_diverged_games_redirect() {
    // The opponent faces a different position in .68.1 than the one we moved
    // from in .68.0, so our next position is only reachable through .68.1.
    const std::string other_board = "------------------XXOO----OXO-----OXXO----OOO------------------- X";
    Slots slots;
    slots.store(make_game(".68.0", 0, START_BOARD, true));
    const GGS_Board update = make_game_after_our_move(".68.0", 0, START_BOARD);
    slots.store(update);
    slots.store(make_game(".68.1", 1, other_board, false));
    require(slots.choose(update) == 1, "diverged games ponder the game where the opponent is thinking");
}

void test_late_endgame_keeps_updated_game() {
    Slots slots;
    const GGS_Board update = make_game(".68.0", 0, LATE_BOARD, false);
    slots.store(update);
    slots.store(make_game(".68.1", 1, LATE_BOARD, false));
#if IS_GGS_TOURNAMENT
    // Our next search keeps the ponder of the updated game alive here.
    require(slots.choose(update) == 0, "late endgame keeps pondering the game we just moved in");
#else
    require(slots.choose(update) == 1, "non-tournament searches stop every ponder");
#endif
}

void test_search_start_ponder_scope() {
#if IS_GGS_TOURNAMENT
    require(ggs_search_start_terminates_all_ponders(GGS_TERMINATE_ALL_PONDERS_MAX_DISCS), "boundary disc count stops all ponders");
    require(!ggs_search_start_terminates_all_ponders(GGS_TERMINATE_ALL_PONDERS_MAX_DISCS + 1), "beyond the boundary only the searched game stops");
#else
    require(ggs_search_start_terminates_all_ponders(HW2), "non-tournament searches stop every ponder");
#endif
}

}

int main() {
    try {
        bit_init();
        mobility_init();
        flip_init();
        test_search_start_ponder_scope();
        test_opponent_thinking_in_other_game();
        test_our_turn_in_other_game();
        test_other_game_not_usable();
        test_identical_games_keep_updated_game();
        test_diverged_games_redirect();
        test_late_endgame_keeps_updated_game();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
