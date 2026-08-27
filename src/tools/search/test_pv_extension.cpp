/*
    Focused regression tests for Edax-style PV extension.

    Example (run the executable from bin/):
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            ../src/tools/search/test_pv_extension.cpp \
            -o test_pv_extension.exe
*/

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(int actual, int expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
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
    if (!hash_resize(DEFAULT_HASH_LEVEL, 20, "./", false)) {
        return false;
    }
    stability_init();
    return evaluate_init(
        "./resources/eval.egev2",
        "./resources/eval_move_ordering_end.egev",
        false
    );
}

void test_schedule() {
    require_equal(get_pv_extension_empties(9, 20), PV_EXTENSION_DISABLED, "depth 9");
    require_equal(get_pv_extension_empties(10, 18), 10, "depth 10");
    require_equal(get_pv_extension_empties(12, 20), 10, "depth 12");
    require_equal(get_pv_extension_empties(13, 23), 12, "depth 13");
    require_equal(get_pv_extension_empties(18, 28), 12, "depth 18");
    require_equal(get_pv_extension_empties(19, 31), 14, "depth 19");
    require_equal(get_pv_extension_empties(24, 36), 14, "depth 24");
    require_equal(get_pv_extension_empties(25, 39), 16, "depth 25");
    require_equal(get_pv_extension_empties(20, 20), PV_EXTENSION_DISABLED, "full-depth root");
}

void test_node_gate() {
    require(
        should_extend_pv_to_end(2, 10, 10, false, SEARCH_NODE_PV),
        "PV node at the trigger must extend"
    );
    require(
        !should_extend_pv_to_end(1, 10, 10, false, SEARCH_NODE_PV),
        "one nominal ply left must not extend"
    );
    require(
        !should_extend_pv_to_end(2, 11, 10, false, SEARCH_NODE_PV),
        "node above the trigger must not extend"
    );
    require(
        !should_extend_pv_to_end(2, 10, 10, true, SEARCH_NODE_PV),
        "an endgame search must not re-enter PV extension"
    );
    require(
        !should_extend_pv_to_end(2, 10, 10, false, SEARCH_NODE_NONPV),
        "a non-PV node must not extend"
    );
}

void test_real_search_enters_extension() {
    Board board;
    require(
        board.from_str("--XXXXX---XXXX---OOOXX---OOXXXX--OOXXXO-OOOOXOO----XOX----XXXXX- O"),
        "regression board parses"
    );
    require_equal(HW2 - board.n_discs(), 23, "regression board empty squares");

    transposition_table.init();
    global_searching = true;
    bool searching = true;
    Search search(&board, MPC_100_LEVEL, false, false);
    const auto result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        13,
        false,
        std::vector<Clog_result>(),
        board.get_legal(),
        tim(),
        &searching
    );
    require(searching && global_searching, "regression search completes");
    require(result.second != MOVE_UNDEFINED, "regression search returns a move");
    require(search.n_pv_extensions > 0, "regression search enters PV extension");

    transposition_table.init();
    searching = true;
    Search exact_search(&board, MPC_100_LEVEL, false, false);
    first_nega_scout_legal(
        &exact_search,
        -SCORE_MAX,
        SCORE_MAX,
        HW2 - board.n_discs(),
        true,
        std::vector<Clog_result>(),
        board.get_legal(),
        tim(),
        &searching
    );
    require_equal(exact_search.n_pv_extensions, 0, "full-depth endgame search extension count");
}

} // namespace

int main() {
    try {
        test_schedule();
        test_node_gate();
        require(initialize_engine(), "engine initialization");
        test_real_search_enters_extension();
    } catch (const std::exception &error) {
        std::cerr << "PV extension test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PV extension tests passed" << std::endl;
    return 0;
}
