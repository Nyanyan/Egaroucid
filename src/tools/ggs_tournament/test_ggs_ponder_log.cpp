/*
    Egaroucid Project

    Regression tests for the "ponder summary" and "ponder check" lines, which
    the GGS client writes to the search log.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            -DINCLUDE_GGS -DIS_GGS_TOURNAMENT test_ggs_ponder_log.cpp \
            -o test_ggs_ponder_log.exe -lws2_32
*/

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include "../../engine/engine_all.hpp"
#include "../../console/console_all.hpp"

namespace {

// 14 discs, the first position of the 2026-10-05 match .68.
const std::string START_BOARD = "------------------OOXX----XOX-----XOOX----XXX------------------- X";

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(const std::string &actual, const std::string &expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(message + "\n  expected: " + expected + "\n  actual:   " + actual);
    }
}

Ponder_elem make_elem(const Board &board, int policy, int count, int depth, uint_fast8_t mpc_level, bool is_end, double value) {
    Ponder_elem elem;
    Board flip_board = board;
    calc_flip(&elem.flip, &flip_board, policy);
    elem.value = count ? value : INF;
    elem.count = count;
    elem.level = 0;
    elem.depth = depth;
    elem.mpc_level = mpc_level;
    elem.selfplay_count = 0;
    elem.is_endgame_search = is_end;
    elem.is_complete_search = is_end && mpc_level == MPC_100_LEVEL;
    return elem;
}

std::vector<int> legal_moves(Board board) {
    std::vector<int> moves;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        moves.emplace_back(cell);
    }
    return moves;
}

// Ponder results for every legal move in ai_ponder's order (most visited first).
// The first three moves are a midgame, a selective endgame and a complete
// result; the rest were never visited.
std::vector<Ponder_elem> make_results(const Board &board) {
    const std::vector<int> moves = legal_moves(board);
    require(moves.size() >= 4, "test root needs at least four legal moves");
    std::vector<Ponder_elem> results;
    results.emplace_back(make_elem(board, moves[0], 5, 28, MPC_88_LEVEL, false, 1.5));
    results.emplace_back(make_elem(board, moves[1], 3, 40, MPC_93_LEVEL, true, -2.25));
    results.emplace_back(make_elem(board, moves[2], 2, 30, MPC_100_LEVEL, true, -4));
    for (size_t i = 3; i < moves.size(); ++i) {
        results.emplace_back(make_elem(board, moves[i], 0, 0, MPC_74_LEVEL, false, 0));
    }
    return results;
}

GGS_Board make_ggs_board(const std::string &game_id, const Board &board, int player_to_move) {
    GGS_Board ggs_board;
    ggs_board.game_id = game_id;
    ggs_board.is_synchro = true;
    ggs_board.synchro_id = 1;
    ggs_board.board = board;
    ggs_board.player_to_move = player_to_move;
    return ggs_board;
}

Board child_of(const Board &board, const Ponder_elem &elem) {
    Board child = board.copy();
    Flip flip = elem.flip;
    child.move_board(&flip);
    return child;
}

void test_summary_text() {
    const Board root = convert_board_from_str(START_BOARD).first;
    const std::vector<Ponder_elem> results = make_results(root);
    const GGS_Ponder_Summary summary = ggs_make_ponder_summary(".68.1", 1, root, BLACK, 1234, results);
    require(summary.visits == 10, "visits are the sum of the counts");
    require(summary.children.size() == results.size(), "every legal move is a child");
    std::string expected =
        "ponder summary game .68.1 slot 1 elapsed 1234 visits 10 moves " + std::to_string(results.size()) +
        " root " + START_BOARD + " top " +
        idx_to_coord(results[0].flip.pos) + ":5:28@88%:m:1.50 " +
        idx_to_coord(results[1].flip.pos) + ":3:40@93%:e:-2.25 " +
        idx_to_coord(results[2].flip.pos) + ":2:30@100%:c:-4.00 " +
        idx_to_coord(results[3].flip.pos) + ":0:0@0%:m:nan";
    if (results.size() >= 5) {
        expected += " " + idx_to_coord(results[4].flip.pos) + ":0:0@0%:m:nan";
    }
    require_equal(ggs_ponder_summary_text(summary), expected, "summary line");
}

std::string board_key(const GGS_Board &ggs_board) {
    return ggs_board.board.to_str(ggs_board.player_to_move);
}

void test_check_relations() {
    const Board root = convert_board_from_str(START_BOARD).first;
    const std::vector<Ponder_elem> results = make_results(root);
    GGS_Ponder_Summary summaries[2];
    summaries[1] = ggs_make_ponder_summary(".68.1", 1, root, BLACK, 1234, results);
    summaries[1].sequence = 1;

    // The opponent played the second most visited move.
    const GGS_Board child = make_ggs_board(".68.1", child_of(root, results[1]), WHITE);
    require_equal(
        ggs_ponder_check_text(child, summaries, 2),
        "ponder check game .68.1 discs 15 relation child rank 2/" + std::to_string(results.size()) +
            " visits 3 depth 40@93% kind e elapsed 1234 ponder_game .68.1 board " + board_key(child),
        "child relation"
    );

    // A move Ponder never visited still has a rank.
    const GGS_Board unvisited = make_ggs_board(".68.1", child_of(root, results[3]), WHITE);
    require_equal(
        ggs_ponder_check_text(unvisited, summaries, 2),
        "ponder check game .68.1 discs 15 relation child rank 4/" + std::to_string(results.size()) +
            " visits 0 depth 0@0% kind m elapsed 1234 ponder_game .68.1 board " + board_key(unvisited),
        "unvisited child relation"
    );

    // Before the synchro games diverge, our position in the other game equals the root.
    const GGS_Board same = make_ggs_board(".68.0", root, BLACK);
    require_equal(
        ggs_ponder_check_text(same, summaries, 2),
        "ponder check game .68.0 discs 14 relation root visits 10 top " + idx_to_coord(results[0].flip.pos) +
            " elapsed 1234 ponder_game .68.1 board " + START_BOARD,
        "root relation"
    );

    // The same discs with the other side to move are a different position.
    const GGS_Board other_side = make_ggs_board(".68.0", root, WHITE);
    require_equal(ggs_ponder_check_text(other_side, summaries, 2), "ponder check game .68.0 discs 14 relation none board " + board_key(other_side), "side to move matters");

    GGS_Ponder_Summary empty[2];
    require_equal(ggs_ponder_check_text(child, empty, 2), "ponder check game .68.1 discs 15 relation none board " + board_key(child), "no finished Ponder");
}

void test_newest_summary_wins() {
    const Board root = convert_board_from_str(START_BOARD).first;
    const std::vector<Ponder_elem> results = make_results(root);
    const Board child_board = child_of(root, results[0]);
    GGS_Ponder_Summary summaries[2];
    // Slot 0: an older Ponder whose root was our position's parent.
    summaries[0] = ggs_make_ponder_summary(".68.0", 0, root, BLACK, 100, results);
    summaries[0].sequence = 1;
    // Slot 1: a newer Ponder whose root is our position itself.
    summaries[1] = ggs_make_ponder_summary(".68.1", 1, child_board, WHITE, 200, make_results(child_board));
    summaries[1].sequence = 2;
    const GGS_Board ggs_board = make_ggs_board(".68.1", child_board, WHITE);
    const std::string text = ggs_ponder_check_text(ggs_board, summaries, 2);
    require(text.find("relation root") != std::string::npos && text.find("ponder_game .68.1") != std::string::npos, "the newest Ponder is reported: " + text);
    summaries[0].sequence = 3;
    const std::string older_text = ggs_ponder_check_text(ggs_board, summaries, 2);
    require(older_text.find("relation child rank 1/") != std::string::npos && older_text.find("ponder_game .68.0") != std::string::npos, "sequence decides which Ponder is newest: " + older_text);
}

// A position where the side to move must pass but the other side can move,
// found by a fixed random playout. Returns the side to move.
int find_pass_position(Board *board) {
    std::mt19937 rng(20261008);
    for (int game = 0; game < 1000; ++game) {
        Board b = convert_board_from_str(START_BOARD).first;
        int player = BLACK;
        while (true) {
            if (b.get_legal() == 0ULL) {
                Board passed = b;
                passed.pass();
                if (passed.get_legal() == 0ULL) {
                    break;
                }
                *board = b;
                return player;
            }
            const std::vector<int> moves = legal_moves(b);
            Flip flip;
            calc_flip(&flip, &b, moves[rng() % moves.size()]);
            b.move_board(&flip);
            player ^= 1;
        }
    }
    throw std::runtime_error("no pass position found");
}

void test_pass_root() {
    Board root;
    const int player = find_pass_position(&root);
    Board passed = root;
    passed.pass();
    const std::vector<int> moves = legal_moves(passed);
    std::vector<Ponder_elem> results;
    results.emplace_back(make_elem(passed, moves[0], 4, 20, MPC_100_LEVEL, true, 6));
    GGS_Ponder_Summary summaries[2];
    summaries[0] = ggs_make_ponder_summary(".70.0", 0, root, player, 50, results);
    summaries[0].sequence = 1;
    require(summaries[0].root_key == root.to_str(player), "the root is logged before the pass");
    // After the other side's move, the side that passed is to move again.
    const GGS_Board child = make_ggs_board(".70.0", child_of(passed, results[0]), player);
    const std::string text = ggs_ponder_check_text(child, summaries, 2);
    require(text.find("relation child rank 1/1 visits 4 depth 20@100% kind c") != std::string::npos, "child after a pass: " + text);
}

}

int main() {
    try {
        bit_init();
        mobility_init();
        flip_init();
        test_summary_text();
        test_check_relations();
        test_newest_summary_wins();
        test_pass_root();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
