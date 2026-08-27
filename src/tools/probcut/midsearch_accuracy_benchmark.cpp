/*
    Egaroucid Project

    Cold-TT midgame accuracy benchmark.

    For every input position this tool compares a selective search with a
    100%-selectivity reference search.  If the policies differ, the selected
    policy is searched again with the reference settings so that policy regret
    is measured in the reference search, rather than by comparing scores from
    different selectivity levels.

    Run from bin/ so that resources/eval.egev2 is available.

    Usage:
        midsearch_accuracy_benchmark.exe <positions> <depth> <mpc-level:0..6>
            <reference-depth> <threads> <hash-level> [position-limit]
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

struct FixedSearchResult {
    int value = SCORE_UNDEFINED;
    int policy = MOVE_UNDEFINED;
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

FixedSearchResult run_fixed_search(
    const Board &board,
    int depth,
    uint_fast8_t mpc_level,
    int n_threads,
    uint64_t use_legal
) {
    transposition_table.init();
    global_searching = true;
    bool searching = true;
    const uint64_t start = tim();
    Search search(&board, mpc_level, n_threads > 1, false);
    search.thread_id = THREAD_ID_NONE;
    const std::pair<int, int> result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        depth,
        false,
        std::vector<Clog_result>(),
        use_legal,
        start,
        &searching
    );
    FixedSearchResult out;
    out.value = result.first;
    out.policy = result.second;
    out.nodes = search.n_nodes;
    out.elapsed = tim() - start;
    out.complete = searching && global_searching;
    return out;
}

std::vector<std::string> read_positions(const std::string &path, int limit) {
    std::ifstream input(path);
    std::vector<std::string> positions;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        positions.emplace_back(line);
        if (limit > 0 && static_cast<int>(positions.size()) >= limit) {
            break;
        }
    }
    return positions;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        std::cerr
            << "usage: " << argv[0]
            << " <positions> <depth> <mpc-level:0..6> <reference-depth>"
               " <threads> <hash-level> [position-limit]\n";
        return 2;
    }

    const std::string positions_path = argv[1];
    const int depth = std::atoi(argv[2]);
    const int mpc_level = std::atoi(argv[3]);
    const int reference_depth = std::atoi(argv[4]);
    const int n_threads = std::atoi(argv[5]);
    const int hash_level = std::atoi(argv[6]);
    const int position_limit = argc == 8 ? std::atoi(argv[7]) : 0;
    if (
        depth <= 0 || reference_depth < depth ||
        mpc_level < 0 || mpc_level >= N_SELECTIVITY_LEVEL ||
        n_threads <= 0 || hash_level < 0 || hash_level >= N_HASH_LEVEL
    ) {
        std::cerr << "invalid numeric argument\n";
        return 2;
    }

    const std::vector<std::string> positions = read_positions(positions_path, position_limit);
    if (positions.empty()) {
        std::cerr << "no positions read\n";
        return 2;
    }
    if (!initialize_engine(n_threads, hash_level)) {
        std::cerr << "engine initialization failed\n";
        return 3;
    }

    int complete_count = 0;
    int policy_agreement = 0;
    int regret_ge_2 = 0;
    int regret_ge_4 = 0;
    int64_t regret_sum = 0;
    int64_t static_abs_error_sum = 0;
    int64_t static_error_sum = 0;
    uint64_t candidate_nodes_sum = 0;
    uint64_t reference_nodes_sum = 0;
    uint64_t candidate_time_sum = 0;
    uint64_t reference_time_sum = 0;

    std::cout
        << "index\tempties\tstatic_value\tcandidate_value\tcandidate_move"
           "\tcandidate_nodes\tcandidate_time_ms\treference_value\treference_move"
           "\treference_nodes\treference_time_ms\tcandidate_reference_value"
           "\tregret\tagree\tcomplete\n";

    for (size_t i = 0; i < positions.size(); ++i) {
        Board board;
        if (!board.from_str(positions[i])) {
            std::cerr << "invalid board at input line " << (i + 1) << '\n';
            return 2;
        }
        const int empties = HW2 - board.n_discs();
        if (reference_depth > empties) {
            std::cerr << "reference depth exceeds empties at input line " << (i + 1) << '\n';
            return 2;
        }
        const uint64_t legal = board.get_legal();
        if (legal == 0ULL) {
            std::cerr << "pass position is unsupported at input line " << (i + 1) << '\n';
            return 2;
        }

        Search eval_search(&board, MPC_100_LEVEL, false, false);
        const int static_value = mid_evaluate_diff(&eval_search);
        const FixedSearchResult candidate = run_fixed_search(
            board, depth, static_cast<uint_fast8_t>(mpc_level), n_threads, legal
        );
        const FixedSearchResult reference = run_fixed_search(
            board, reference_depth, MPC_100_LEVEL, n_threads, legal
        );

        int candidate_reference_value = reference.value;
        FixedSearchResult forced;
        forced.complete = true;
        if (candidate.policy != reference.policy) {
            forced = run_fixed_search(
                board,
                reference_depth,
                MPC_100_LEVEL,
                n_threads,
                1ULL << candidate.policy
            );
            candidate_reference_value = forced.value;
        }

        const bool complete = candidate.complete && reference.complete && forced.complete;
        const bool agree = candidate.policy == reference.policy;
        const int regret = reference.value - candidate_reference_value;
        if (complete) {
            ++complete_count;
            policy_agreement += agree;
            regret_sum += regret;
            regret_ge_2 += regret >= 2;
            regret_ge_4 += regret >= 4;
            static_error_sum += static_value - reference.value;
            static_abs_error_sum += std::abs(static_value - reference.value);
            candidate_nodes_sum += candidate.nodes;
            reference_nodes_sum += reference.nodes;
            candidate_time_sum += candidate.elapsed;
            reference_time_sum += reference.elapsed;
        }

        std::cout
            << (i + 1) << '\t'
            << empties << '\t'
            << static_value << '\t'
            << candidate.value << '\t'
            << idx_to_coord(candidate.policy) << '\t'
            << candidate.nodes << '\t'
            << candidate.elapsed << '\t'
            << reference.value << '\t'
            << idx_to_coord(reference.policy) << '\t'
            << reference.nodes << '\t'
            << reference.elapsed << '\t'
            << candidate_reference_value << '\t'
            << regret << '\t'
            << static_cast<int>(agree) << '\t'
            << static_cast<int>(complete) << '\n';
    }

    const double denominator = std::max(1, complete_count);
    std::cerr << std::fixed << std::setprecision(3)
              << "SUMMARY"
              << " positions=" << positions.size()
              << " complete=" << complete_count
              << " agreement=" << (policy_agreement / denominator)
              << " mean_regret=" << (regret_sum / denominator)
              << " regret_ge_2=" << regret_ge_2
              << " regret_ge_4=" << regret_ge_4
              << " static_bias=" << (static_error_sum / denominator)
              << " static_mae=" << (static_abs_error_sum / denominator)
              << " candidate_nodes=" << candidate_nodes_sum
              << " reference_nodes=" << reference_nodes_sum
              << " candidate_time_ms=" << candidate_time_sum
              << " reference_time_ms=" << reference_time_sum
              << '\n';
    return complete_count == static_cast<int>(positions.size()) ? 0 : 4;
}
