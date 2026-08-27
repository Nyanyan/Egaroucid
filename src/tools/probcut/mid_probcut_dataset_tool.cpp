/*
    Egaroucid Project

    Collect cold-TT static/shallow/deep score differences for midgame MPC.

    Run from bin/:
        mid_probcut_dataset_tool.exe <positions> <deep-depth> <shallow-depths>
            <threads> <hash-level> [position-limit]
*/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

struct Score {
    int value = SCORE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed = 0;
    bool complete = false;
};

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

std::vector<int> parse_depths(const std::string &text, int deep_depth) {
    std::vector<int> result;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        const int depth = std::atoi(field.c_str());
        if (0 <= depth && depth < deep_depth && (depth == 0 || (depth & 1) == (deep_depth & 1))) {
            result.emplace_back(depth);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

Score search_score(const Board &board, int depth, int n_threads) {
    transposition_table.init();
    global_searching = true;
    bool searching = true;
    const uint64_t start = tim();
    Search search(&board, MPC_100_LEVEL, n_threads > 1, false);
    search.thread_id = THREAD_ID_NONE;
    const std::pair<int, int> result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        depth,
        false,
        std::vector<Clog_result>(),
        board.get_legal(),
        start,
        &searching
    );
    return Score{result.first, search.n_nodes, tim() - start, searching && global_searching};
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr
            << "usage: " << argv[0]
            << " <positions> <deep-depth> <shallow-depths> <threads> <hash-level>"
               " [position-limit]\n";
        return 2;
    }
    const std::string positions_path = argv[1];
    const int deep_depth = std::atoi(argv[2]);
    const std::vector<int> shallow_depths = parse_depths(argv[3], deep_depth);
    const int n_threads = std::atoi(argv[4]);
    const int hash_level = std::atoi(argv[5]);
    const int position_limit = argc == 7 ? std::atoi(argv[6]) : 0;
    if (
        deep_depth <= 1 || shallow_depths.empty() || n_threads <= 0 ||
        hash_level < 0 || hash_level >= N_HASH_LEVEL
    ) {
        std::cerr << "invalid argument\n";
        return 2;
    }
    if (!initialize_engine(n_threads, hash_level)) {
        std::cerr << "engine initialization failed\n";
        return 3;
    }

    std::ifstream input(positions_path);
    std::string line;
    int index = 0;
    std::cout
        << "index\tboard\tn_discs\tdeep_depth\tshallow_depth\tstatic_value"
           "\tshallow_value\tdeep_value\terror\tshallow_nodes\tdeep_nodes"
           "\tshallow_time_ms\tdeep_time_ms\tlegal_count\n";
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        ++index;
        if (position_limit > 0 && index > position_limit) {
            break;
        }
        Board board;
        if (!board.from_str(line) || deep_depth > HW2 - board.n_discs() || board.get_legal() == 0ULL) {
            std::cerr << "invalid position at line " << index << '\n';
            return 2;
        }
        Search eval_search(&board, MPC_100_LEVEL, false, false);
        const int static_value = mid_evaluate_diff(&eval_search);
        const Score deep = search_score(board, deep_depth, n_threads);
        if (!deep.complete) {
            std::cerr << "deep search terminated at line " << index << '\n';
            return 4;
        }
        for (const int shallow_depth : shallow_depths) {
            Score shallow;
            if (shallow_depth == 0) {
                shallow = Score{static_value, 0, 0, true};
            } else {
                shallow = search_score(board, shallow_depth, n_threads);
            }
            if (!shallow.complete) {
                std::cerr << "shallow search terminated at line " << index << '\n';
                return 4;
            }
            std::cout
                << index << '\t'
                << line << '\t'
                << static_cast<int>(board.n_discs()) << '\t'
                << deep_depth << '\t'
                << shallow_depth << '\t'
                << static_value << '\t'
                << shallow.value << '\t'
                << deep.value << '\t'
                << deep.value - shallow.value << '\t'
                << shallow.nodes << '\t'
                << deep.nodes << '\t'
                << shallow.elapsed << '\t'
                << deep.elapsed << '\t'
                << pop_count_ull(board.get_legal()) << '\n';
        }
    }
    return 0;
}
