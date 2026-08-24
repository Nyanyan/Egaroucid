/*
    Egaroucid Project

    Regression test for the selective endgame MPC safety reserve.
    Run the executable from bin/ so the evaluation resources are available.
*/

#include <iostream>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

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

bool check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

} // namespace

int main() {
    bool ok = true;
    ok &= check(mpc_error_margin<true>(MPC_74_LEVEL, 24) == 2, "74% depth-24 margin");
    ok &= check(mpc_error_margin<true>(MPC_74_LEVEL, 25) == 0, "74% depth-25 margin");
    ok &= check(mpc_error_margin<true>(MPC_88_LEVEL, 24) == 0, "88% is unchanged");
    ok &= check(mpc_error_margin<false>(MPC_74_LEVEL, 24) == 0, "midgame MPC is unchanged");
    if (!ok) {
        return 1;
    }
    if (!initialize_engine()) {
        std::cerr << "FAIL: engine initialization\n";
        return 2;
    }

    Board board;
    if (!board.from_str("-XXXX-----OXX-O--OXXX-OO-O-XXXOOXOOOOOOOXOOXOOOOOOOOXXOOXO-OX-OO X")) {
        std::cerr << "FAIL: regression board\n";
        return 2;
    }
    bool searching = true;
    global_searching = true;
    const uint64_t start = tim();
    Search search(&board, MPC_74_LEVEL, false, false);
    const std::pair<int, int> result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        HW2 - board.n_discs(),
        true,
        std::vector<Clog_result>(),
        board.get_legal(),
        start,
        &searching
    );
    ok &= check(searching, "search completed");
    ok &= check(result.first == -18, "regression score is -18");
    ok &= check(result.second == get_coord_from_chars('c', '8'), "regression move is c8");
    if (!ok) {
        std::cerr << "got " << result.first << ' ' << idx_to_coord(result.second) << '\n';
        return 1;
    }
    std::cout << "PASS end MPC safety regression "
              << result.first << ' ' << idx_to_coord(result.second)
              << " nodes " << search.n_nodes << '\n';
    return 0;
}
