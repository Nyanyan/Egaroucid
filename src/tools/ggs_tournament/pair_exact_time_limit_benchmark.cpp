/*
    Exercise the GGS pair-outcome search under a real time limit.

    Build once with the default early exact pair NWS and once with
    -DEGAROUCID_GGS_EARLY_PAIR_EXACT_NWS=0 for an A/B comparison.
    Run from bin/ so evaluation resources are available.
*/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "../../engine/engine_all.hpp"

namespace {

struct Exact_Result {
    int value = SCORE_UNDEFINED;
    int policy = MOVE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed = 0;
};

bool initialize_engine(int n_threads, int hash_level) {
    thread_pool.resize(std::max(0, n_threads - 1));
    bit_init(); mobility_init(); flip_init(); last_flip_init(); endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
    if (!hash_resize(DEFAULT_HASH_LEVEL, hash_level, "./", false)) return false;
    stability_init();
    return evaluate_init("./resources/eval.egev2", "./resources/eval_move_ordering_end.egev", false);
}

Exact_Result exact_search(const Board &board, int n_threads, uint64_t legal) {
    transposition_table.init();
    global_searching = true;
    bool searching = true;
    const uint64_t start = tim();
    Search search(&board, MPC_100_LEVEL, n_threads > 1, false);
    search.thread_id = THREAD_ID_NONE;
    const std::pair<int, int> raw = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        HW2 - board.n_discs(),
        true,
        std::vector<Clog_result>(),
        legal,
        start,
        &searching
    );
    if (!searching) return Exact_Result();
    return Exact_Result{raw.first, raw.second, search.n_nodes, tim() - start};
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " <positions> <time-ms> <threads> <hash-level> <pair-value> <limit> [exact-reference:0|1]\n";
        return 2;
    }
    const std::string path = argv[1];
    const uint64_t time_limit = std::strtoull(argv[2], nullptr, 10);
    const int n_threads = std::atoi(argv[3]);
    const int hash_level = std::atoi(argv[4]);
    const int pair_value = std::atoi(argv[5]);
    const int limit = std::atoi(argv[6]);
    const bool run_exact_reference = argc == 8 && std::atoi(argv[7]) != 0;
    if (time_limit == 0 || n_threads <= 0 || limit < 0) return 2;
    if (!initialize_engine(n_threads, hash_level)) return 3;

    std::ifstream input(path);
    if (!input) return 2;
    std::string line;
    int index = 0;
    int attempted = 0;
    int early_win = 0;
    int early_draw = 0;
    int exact_value = 0;
    int proven_lower_bound = 0;
    int pair_win = 0;
    int pair_nonloss = 0;
    int pair_outcome_degradation = 0;
    uint64_t total_nodes = 0;
    uint64_t total_time = 0;
    uint64_t reference_nodes = 0;
    uint64_t reference_time = 0;
    std::cout << "index\tempties\tmove\tvalue\tdepth\tprobability\texact_value"
                 "\tproven_lower_bound\tpair_outcome\tnodes\ttime_ms\tearly_attempted"
                 "\tearly_budget\tearly_proved_outcome\treference_value\treference_move"
                 "\tcandidate_exact_value\treference_pair_outcome\tcandidate_pair_outcome"
                 "\tpair_outcome_degradation\treference_nodes\treference_time_ms\n";
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (limit > 0 && index >= limit) break;
        Board board;
        if (!board.from_str(line)) return 2;
        ++index;

        transposition_table.init();
        global_searching = true;
        bool searching = true;
        AI_TL_Iteration_Diagnostics diagnostics;
        diagnostics.enable_pair_outcome_nws = true;
        diagnostics.pair_value = pair_value;
        Search_result result;
        iterative_deepening_search_time_limit(
            board,
            -SCORE_MAX,
            SCORE_MAX,
            false,
            std::vector<Clog_result>(),
            board.get_legal(),
            n_threads > 1,
            THREAD_ID_NONE,
            &result,
            time_limit,
            &searching,
            &diagnostics
        );

        const bool has_exact = ai_search_result_has_exact_value(board.n_discs(), result);
        const bool has_lower = ai_search_result_has_proven_lower_bound(board.n_discs(), result);
        int outcome = -2;
        if (has_exact) {
            outcome = ai_tl_ggs_match_outcome(pair_value + result.value);
        } else if (has_lower && pair_value + result.value > 0) {
            outcome = 1;
        }
        attempted += diagnostics.early_pair_exact_attempted;
        early_win += diagnostics.early_pair_exact_proved_outcome == AI_TL_GGS_EARLY_PAIR_PROVED_WIN;
        early_draw += diagnostics.early_pair_exact_proved_outcome == AI_TL_GGS_EARLY_PAIR_PROVED_DRAW;
        exact_value += has_exact;
        proven_lower_bound += has_lower;
        pair_win += outcome > 0;
        pair_nonloss += outcome >= 0;
        total_nodes += result.nodes;
        total_time += result.time;

        Exact_Result reference;
        Exact_Result candidate_exact;
        int reference_outcome = -2;
        int candidate_outcome = -2;
        bool degraded = false;
        if (run_exact_reference) {
            reference = exact_search(board, n_threads, board.get_legal());
            if (reference.value == SCORE_UNDEFINED) return 4;
            if (result.policy == reference.policy) {
                candidate_exact = reference;
                candidate_exact.nodes = 0;
                candidate_exact.elapsed = 0;
            } else {
                candidate_exact = exact_search(board, n_threads, 1ULL << result.policy);
                if (candidate_exact.value == SCORE_UNDEFINED) return 4;
            }
            reference_outcome = ai_tl_ggs_match_outcome(pair_value + reference.value);
            candidate_outcome = ai_tl_ggs_match_outcome(pair_value + candidate_exact.value);
            degraded = candidate_outcome < reference_outcome;
            pair_outcome_degradation += degraded;
            reference_nodes += reference.nodes + candidate_exact.nodes;
            reference_time += reference.elapsed + candidate_exact.elapsed;
        }
        std::cout << index << '\t' << HW2 - board.n_discs() << '\t'
                  << (is_valid_policy(result.policy) ? idx_to_coord(result.policy) : "undefined") << '\t'
                  << result.value << '\t' << result.depth << '\t' << result.probability << '\t'
                  << has_exact << '\t' << has_lower << '\t' << outcome << '\t'
                  << result.nodes << '\t' << result.time << '\t'
                  << diagnostics.early_pair_exact_attempted << '\t'
                  << diagnostics.early_pair_exact_budget << '\t'
                  << diagnostics.early_pair_exact_proved_outcome << '\t'
                  << reference.value << '\t'
                  << (is_valid_policy(reference.policy) ? idx_to_coord(reference.policy) : "undefined") << '\t'
                  << candidate_exact.value << '\t' << reference_outcome << '\t'
                  << candidate_outcome << '\t' << degraded << '\t'
                  << reference.nodes + candidate_exact.nodes << '\t'
                  << reference.elapsed + candidate_exact.elapsed << '\n';
    }
    std::cerr << std::fixed << std::setprecision(3)
              << "SUMMARY positions=" << index
              << " early_attempted=" << attempted
              << " early_win=" << early_win
              << " early_draw=" << early_draw
              << " exact_value=" << exact_value
              << " proven_lower_bound=" << proven_lower_bound
              << " pair_win=" << pair_win
              << " pair_nonloss=" << pair_nonloss
              << " pair_outcome_degradation=" << pair_outcome_degradation
              << " nodes=" << total_nodes
              << " time_ms=" << total_time
              << " avg_time_ms=" << (double)total_time / std::max(1, index)
              << " reference_nodes=" << reference_nodes
              << " reference_time_ms=" << reference_time
              << '\n';
    return 0;
}
