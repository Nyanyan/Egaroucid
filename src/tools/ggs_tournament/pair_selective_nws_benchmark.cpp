/*
    Compare a selective full-window root search with pair-outcome NWS probes.

    The benchmark derives its targets from the full-window result:
      - direct: target == full value (one fail-high probe)
      - win-draw: win target == full value + 1 (fail-low), followed by a draw
        target == full value (fail-high)

    Run from bin/ so evaluation resources are available.
*/

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

struct Probe {
    int value = SCORE_UNDEFINED;
    int policy = MOVE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed = 0;
    bool complete = false;
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

Probe run(
    const Board &board,
    int alpha,
    int beta,
    int depth,
    uint_fast8_t mpc_level,
    int n_threads,
    bool clear_tt
) {
    if (clear_tt) transposition_table.init();
    bool searching = true;
    global_searching = true;
    const uint64_t start = tim();
    Search search(&board, mpc_level, n_threads > 1, false);
    const auto result = first_nega_scout_legal(
        &search, alpha, beta, depth, false, {}, board.get_legal(), start, &searching
    );
    return Probe{result.first, result.second, search.n_nodes, tim() - start, searching};
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " <positions> <depth> <mpc-level> <threads> <hash-level>"
                     " <direct|win-draw> [limit]\n";
        return 2;
    }
    const std::string path = argv[1];
    const int depth = std::atoi(argv[2]);
    const int mpc_level = std::atoi(argv[3]);
    const int n_threads = std::atoi(argv[4]);
    const int hash_level = std::atoi(argv[5]);
    const std::string mode = argv[6];
    const int limit = argc == 8 ? std::atoi(argv[7]) : 0;
    if (depth <= 0 || mpc_level < 0 || mpc_level >= N_SELECTIVITY_LEVEL ||
        n_threads <= 0 || (mode != "direct" && mode != "win-draw")) {
        return 2;
    }
    if (!initialize_engine(n_threads, hash_level)) return 3;

    std::ifstream input(path);
    std::string line;
    int index = 0;
    int positions = 0;
    int correct = 0;
    uint64_t full_nodes = 0, probe_nodes = 0, full_time = 0, probe_time = 0;
    std::cout << "index\tempties\tfull_value\tfull_move\tfull_nodes\tfull_ms"
                 "\tprobe_value\tprobe_move\tprobe_nodes\tprobe_ms\tcorrect\n";
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        ++index;
        if (limit > 0 && index > limit) break;
        ++positions;
        Board board;
        if (!board.from_str(line) || depth > HW2 - board.n_discs()) return 2;
        const Probe full = run(board, -SCORE_MAX, SCORE_MAX, depth, mpc_level, n_threads, true);
        if (!full.complete) return 4;

        Probe selected;
        bool outcome_consistent = true;
        uint64_t selected_nodes = 0, selected_time = 0;
        if (mode == "win-draw") {
            const int win_target = full.value + 1;
            Probe win;
            if (win_target <= SCORE_MAX) {
                win = run(board, win_target - 1, win_target, depth, mpc_level, n_threads, true);
                selected_nodes += win.nodes;
                selected_time += win.elapsed;
                if (!win.complete) return 4;
            } else {
                transposition_table.init();
            }
            if (win_target <= SCORE_MAX && win.value >= win_target) {
                outcome_consistent = false;
                selected = win;
                // The nodes/time of this probe were already accumulated above.
                selected.nodes = 0;
                selected.elapsed = 0;
            } else {
                selected = run(board, full.value - 1, full.value, depth, mpc_level, n_threads, false);
            }
        } else {
            selected = run(board, full.value - 1, full.value, depth, mpc_level, n_threads, true);
        }
        selected_nodes += selected.nodes;
        selected_time += selected.elapsed;
        const bool ok = outcome_consistent && selected.complete && selected.value >= full.value;
        correct += ok;
        full_nodes += full.nodes;
        full_time += full.elapsed;
        probe_nodes += selected_nodes;
        probe_time += selected_time;
        std::cout << index << '\t' << HW2 - board.n_discs() << '\t'
                  << full.value << '\t' << idx_to_coord(full.policy) << '\t'
                  << full.nodes << '\t' << full.elapsed << '\t'
                  << selected.value << '\t' << idx_to_coord(selected.policy) << '\t'
                  << selected_nodes << '\t' << selected_time << '\t' << ok << '\n';
    }
    std::cerr << std::fixed << std::setprecision(3)
              << "SUMMARY positions=" << positions << " correct=" << correct
              << " full_nodes=" << full_nodes << " probe_nodes=" << probe_nodes
              << " node_ratio=" << (double)probe_nodes / std::max<uint64_t>(1, full_nodes)
              << " full_ms=" << full_time << " probe_ms=" << probe_time
              << " time_ratio=" << (double)probe_time / std::max<uint64_t>(1, full_time)
              << '\n';
    return correct == positions ? 0 : 5;
}
