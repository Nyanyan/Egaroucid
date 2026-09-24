/*
    Egaroucid Project

    A small, fixed-condition driver for comparing full-depth endgame searches
    at a specified MPC level. Run the executable from bin/ so that the
    evaluation files are found in bin/resources/.

    Example:
        end_mpc_search_benchmark.exe 1 20 20 "<64 cells> X"

    Arguments:
        MPC level (0=74%, 1=88%, ..., 6=100%)
        thread count
        hash level
        board string
        optional --diagnostics (engine iteration and YBWC statistics)
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

bool initialize_engine(int n_threads, int hash_level) {
    thread_pool.resize(std::max(0, n_threads - 1));
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
    if (!hash_resize(DEFAULT_HASH_LEVEL, hash_level, "./", false)) {
        return false;
    }
    stability_init();
    return evaluate_init(
        "./resources/eval.egev2",
        "./resources/eval_move_ordering_end.egev",
        false
    );
}

} // namespace

int main(int argc, char **argv) {
    const bool diagnostics = argc == 6 && std::string(argv[5]) == "--diagnostics";
    if (argc != 5 && !diagnostics) {
        std::cerr << "usage: " << argv[0]
                  << " <mpc-level:0..6> <threads> <hash-level> \"<board> <side>\""
                  << " [--diagnostics]\n";
        return 2;
    }

    const int mpc_level = std::atoi(argv[1]);
    const int n_threads = std::atoi(argv[2]);
    const int hash_level = std::atoi(argv[3]);
    if (mpc_level < 0 || mpc_level >= N_SELECTIVITY_LEVEL ||
        n_threads <= 0 || hash_level < 0 || hash_level >= N_HASH_LEVEL) {
        std::cerr << "invalid numeric argument\n";
        return 2;
    }

    Board board;
    if (!board.from_str(argv[4])) {
        std::cerr << "invalid board\n";
        return 2;
    }
    if (!initialize_engine(n_threads, hash_level)) {
        std::cerr << "engine initialization failed\n";
        return 3;
    }

    bool searching = true;
    global_searching = true;
    const int depth = HW2 - board.n_discs();
    const uint64_t start = tim();
    const Search_result result = tree_search_legal(
        board,
        -SCORE_MAX,
        SCORE_MAX,
        depth,
        (uint_fast8_t)mpc_level,
        diagnostics,
        board.get_legal(),
        n_threads > 1,
        TIME_LIMIT_INF,
        THREAD_ID_NONE,
        &searching
    );
    const uint64_t elapsed = tim() - start;
    const uint64_t nodes = result.nodes + result.clog_nodes;

    std::cout << "value\tmove\tdepth\tmpc_level\tnodes\ttime_ms\tnps\n"
              << result.value << '\t'
              << idx_to_coord(result.policy) << '\t'
              << depth << '\t'
              << mpc_level << '\t'
              << nodes << '\t'
              << elapsed << '\t'
              << calc_nps(nodes, elapsed) << '\n';
    return searching ? 0 : 4;
}
