/*
    Focused regression tests for midgame NWS cancellation.

    Build and run from bin/ so that the evaluation resources are available:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            ../src/tools/probcut/test_midsearch_nws_cancellation.cpp \
            -o test_midsearch_nws_cancellation.exe
*/

#define EGAROUCID_TEST_MIDSEARCH_NWS_CANCELLATION

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../engine/engine_all.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Search_snapshot {
    uint64_t player;
    uint64_t opponent;
    int n_discs;
    uint_fast8_t parity;
    uint_fast8_t eval_feature_idx;
    int evaluation;
};

Search_snapshot take_snapshot(Search *search) {
    return Search_snapshot{
        search->board.player,
        search->board.opponent,
        search->n_discs,
        search->parity,
        search->eval.feature_idx,
        mid_evaluate_diff(search)
    };
}

void require_restored(Search *search, const Search_snapshot &before, const std::string &message) {
    require(search->board.player == before.player, message + ": player bitboard");
    require(search->board.opponent == before.opponent, message + ": opponent bitboard");
    require(search->n_discs == before.n_discs, message + ": disc count");
    require(search->parity == before.parity, message + ": parity");
    require(search->eval.feature_idx == before.eval_feature_idx, message + ": evaluation stack index");
    require(mid_evaluate_diff(search) == before.evaluation, message + ": evaluation value");
}

bool initialize_engine() {
    thread_pool.resize(0);
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
    if (!transposition_table.resize(10)) {
        return false;
    }
    hash_init_rand(10);
    global_hash_level = 10;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
    stability_init();
    return evaluate_init(
        "./resources/eval.egev2",
        "./resources/eval_move_ordering_end.egev",
        false
    );
}

void test_cancelled_at_entry() {
    Board board;
    board.reset();

    for (int depth = 0; depth <= MID_SIMPLE_ORDERING_DEPTH; ++depth) {
        transposition_table.init();
        global_searching = true;
        bool searching = false;
        Search search(&board, MPC_100_LEVEL, false, false);
        const Search_snapshot before = take_snapshot(&search);
        const int value = nega_alpha_ordering_nws_simple(
            &search, 0, depth, Nws_node_hint::no_static_eval(), board.get_legal(), &searching
        );
        require(value == SCORE_UNDEFINED, "false local flag at depth " + std::to_string(depth));
        require(search.n_nodes == 0, "false local flag visits no nodes at depth " + std::to_string(depth));
        require_restored(&search, before, "false local flag restores depth " + std::to_string(depth));

        transposition_table.init();
        global_searching = false;
        searching = true;
        Search global_search(&board, MPC_100_LEVEL, false, false);
        const Search_snapshot global_before = take_snapshot(&global_search);
        const int global_value = nega_alpha_ordering_nws_simple(
            &global_search, 0, depth, Nws_node_hint::no_static_eval(), board.get_legal(), &searching
        );
        global_searching = true;
        require(global_value == SCORE_UNDEFINED, "false global flag at depth " + std::to_string(depth));
        require(global_search.n_nodes == 0, "false global flag visits no nodes at depth " + std::to_string(depth));
        require_restored(&global_search, global_before, "false global flag restores depth " + std::to_string(depth));
    }
}

void cancel_after_pass(bool *searching) {
    search_cancellation_store(searching, false);
}

Board make_pass_board() {
    Board board;
    const std::string board_text = ".OXXXXXX" + std::string(56, '-') + " O";
    require(board.from_str(board_text), "pass board parses");
    require(board.get_legal() == 0ULL, "side to move has no legal move");
    Board passed = board.copy();
    passed.pass();
    require(passed.get_legal() != 0ULL, "opponent has a legal move after pass");
    return board;
}

void test_pass_cancellation_propagates() {
    const Board board = make_pass_board();
    for (const int depth: {2, 3, 4}) {
        transposition_table.init();
        global_searching = true;
        bool searching = true;
        Search search(&board, MPC_100_LEVEL, false, false);
        const Search_snapshot before = take_snapshot(&search);

        midsearch_nws_after_pass_test_hook = cancel_after_pass;
        const int value = nega_alpha_ordering_nws_simple(
            &search, 0, depth, Nws_node_hint::no_static_eval(), LEGAL_UNDEFINED, &searching
        );
        midsearch_nws_after_pass_test_hook = nullptr;

        require(!searching, "pass hook cancels depth " + std::to_string(depth));
        require(value == SCORE_UNDEFINED, "pass propagates undefined at depth " + std::to_string(depth));
        require(search.n_nodes == 1, "cancelled pass visits only its parent at depth " + std::to_string(depth));
        require_restored(&search, before, "cancelled pass restores depth " + std::to_string(depth));
    }
}

int eval1_children_completed = 0;

void cancel_after_first_eval1_child(bool *searching) {
    ++eval1_children_completed;
    if (eval1_children_completed == 1) {
        search_cancellation_store(searching, false);
    }
}

void test_eval2_post_child_polling() {
    Board board;
    board.reset();
    const int n_root_children = pop_count_ull(board.get_legal());
    require(n_root_children > 1, "batch test has multiple root children");

    transposition_table.init();
    global_searching = true;
    bool searching = true;
    Search search(&board, MPC_100_LEVEL, false, false);
    const Search_snapshot before = take_snapshot(&search);
    eval1_children_completed = 0;
    midsearch_nws_after_eval1_test_hook = cancel_after_first_eval1_child;
    const int value = nega_alpha_eval2_nws(
        &search, SCORE_MAX - 1, Nws_node_hint::no_static_eval(), board.get_legal(), &searching
    );
    midsearch_nws_after_eval1_test_hook = nullptr;

    require(
        !searching,
        "eval2 child hook cancels local search (children=" +
            std::to_string(eval1_children_completed) +
            ", nodes=" + std::to_string(search.n_nodes) +
            ", value=" + std::to_string(value) + ")"
    );
    require(value == SCORE_UNDEFINED, "eval2 post-child poll propagates undefined");
    require(
        eval1_children_completed == 1,
        "eval2 polls immediately after the first child"
    );
    require_restored(&search, before, "cancelled eval2 child restores state");
}

void test_completed_pass_restores_state() {
    const Board board = make_pass_board();
    for (const int depth: {2, 3, 4}) {
        transposition_table.init();
        global_searching = true;
        bool searching = true;
        Search search(&board, MPC_100_LEVEL, false, false);
        const Search_snapshot before = take_snapshot(&search);
        const int value = nega_alpha_ordering_nws_simple(
            &search, 0, depth, Nws_node_hint::no_static_eval(), LEGAL_UNDEFINED, &searching
        );
        require(searching, "pass search completes at depth " + std::to_string(depth));
        require(value != SCORE_UNDEFINED, "completed pass has a value at depth " + std::to_string(depth));
        require_restored(&search, before, "completed pass restores depth " + std::to_string(depth));
    }
}

} // namespace

int main() {
    try {
        require(initialize_engine(), "engine initialization");
        test_cancelled_at_entry();
        test_pass_cancellation_propagates();
        test_eval2_post_child_polling();
        test_completed_pass_restores_state();
    } catch (const std::exception &error) {
        midsearch_nws_after_pass_test_hook = nullptr;
        midsearch_nws_after_eval1_test_hook = nullptr;
        global_searching = true;
        std::cerr << "Midsearch NWS cancellation test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Midsearch NWS cancellation tests passed" << std::endl;
    return 0;
}
