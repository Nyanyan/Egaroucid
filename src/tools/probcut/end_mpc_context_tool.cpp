/*
    Egaroucid Project

    Diagnostic helper for endgame MPC.  This tool is built with
    END_PROBCUT_CONTEXT_TRACE and is not part of the production engine.
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

int trace(
    const std::string &board_text,
    int mpc_level,
    int contexts_per_depth,
    int min_depth,
    int max_depth
) {
    Board board;
    if (!board.from_str(board_text)) {
        std::cerr << "invalid board\n";
        return 2;
    }
    bool searching = true;
    global_searching = true;
    const uint64_t start = tim();
    Search search(&board, (uint_fast8_t)mpc_level, false, false);
    end_probcut_context_trace_min_depth = std::max(0, min_depth);
    end_probcut_context_trace_max_depth = std::min(HW2, max_depth);
    end_probcut_context_trace_remaining =
        std::max(0, contexts_per_depth) *
        std::max(0, end_probcut_context_trace_max_depth - end_probcut_context_trace_min_depth + 1);
    for (int d = 0; d <= HW2; ++d) {
        end_probcut_context_trace_per_depth_remaining[d] =
            end_probcut_context_trace_min_depth <= d && d <= end_probcut_context_trace_max_depth
                ? std::max(0, contexts_per_depth)
                : 0;
    }
    const int depth = HW2 - board.n_discs();
    const std::pair<int, int> result = first_nega_scout_legal(
        &search,
        -SCORE_MAX,
        SCORE_MAX,
        depth,
        true,
        std::vector<Clog_result>(),
        board.get_legal(),
        start,
        &searching
    );
    std::cout << "TRACE_RESULT\t" << result.first << '\t'
              << idx_to_coord(result.second) << '\t' << search.n_nodes << '\t'
              << tim() - start << '\n';
    return 0;
}

std::vector<int> parse_depths(const std::string &text) {
    std::vector<int> result;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, ',')) {
        const int depth = std::atoi(field.c_str());
        if (0 <= depth && depth < HW2) {
            result.emplace_back(depth);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int score_grid(const std::string &board_text, int deep_depth, const std::string &depth_text) {
    Board board;
    if (!board.from_str(board_text)) {
        std::cerr << "invalid board\n";
        return 2;
    }
    const uint64_t legal = board.get_legal();
    if (legal == 0 || deep_depth <= 0 || deep_depth > HW2 - board.n_discs()) {
        std::cerr << "invalid scoring context\n";
        return 2;
    }
    const std::vector<int> shallow_depths = parse_depths(depth_text);
    if (shallow_depths.empty()) {
        std::cerr << "no shallow depths\n";
        return 2;
    }

    Search eval_search(&board, MPC_100_LEVEL, false, false);
    const int d0_value = mid_evaluate_diff(&eval_search);
    std::vector<int> shallow_values;
    std::vector<uint64_t> shallow_nodes;
    std::vector<uint64_t> shallow_times;
    for (const int shallow_depth : shallow_depths) {
        if (shallow_depth == 0) {
            shallow_values.emplace_back(d0_value);
            shallow_nodes.emplace_back(0);
            shallow_times.emplace_back(0);
            continue;
        }
        transposition_table.init();
        bool searching = true;
        global_searching = true;
        const uint64_t start = tim();
        Search shallow_search(&board, MPC_100_LEVEL, false, false);
        const std::pair<int, int> result = first_nega_scout_legal(
            &shallow_search,
            -SCORE_MAX,
            SCORE_MAX,
            shallow_depth,
            false,
            std::vector<Clog_result>(),
            legal,
            start,
            &searching
        );
        if (!searching) {
            return 4;
        }
        shallow_values.emplace_back(result.first);
        shallow_nodes.emplace_back(shallow_search.n_nodes);
        shallow_times.emplace_back(tim() - start);
    }

    transposition_table.init();
    bool searching = true;
    global_searching = true;
    const uint64_t deep_start = tim();
    Search deep_search(&board, MPC_100_LEVEL, false, false);
    const std::pair<int, int> deep_result = first_nega_scout_legal(
        &deep_search,
        -SCORE_MAX,
        SCORE_MAX,
        deep_depth,
        true,
        std::vector<Clog_result>(),
        legal,
        deep_start,
        &searching
    );
    if (!searching) {
        return 4;
    }
    const uint64_t deep_time = tim() - deep_start;
    for (size_t i = 0; i < shallow_depths.size(); ++i) {
        std::cout
            << "END_PROBCUT_SCORE_V2\t"
            << deep_depth << '\t'
            << shallow_depths[i] << '\t'
            << shallow_values[i] << '\t'
            << deep_result.first << '\t'
            << shallow_nodes[i] << '\t'
            << deep_search.n_nodes << '\t'
            << shallow_times[i] << '\t'
            << deep_time << '\t'
            << d0_value << '\t'
            << pop_count_ull(legal)
            << '\n';
    }
    return 0;
}

int shallow_grid(const std::string &board_text, int deep_depth, const std::string &depth_text) {
    Board board;
    if (!board.from_str(board_text)) {
        std::cerr << "invalid board\n";
        return 2;
    }
    const uint64_t legal = board.get_legal();
    if (legal == 0 || deep_depth <= 0 || deep_depth > HW2 - board.n_discs()) {
        std::cerr << "invalid scoring context\n";
        return 2;
    }
    const std::vector<int> shallow_depths = parse_depths(depth_text);
    if (shallow_depths.empty()) {
        std::cerr << "no shallow depths\n";
        return 2;
    }

    Search eval_search(&board, MPC_100_LEVEL, false, false);
    const int d0_value = mid_evaluate_diff(&eval_search);
    for (const int shallow_depth : shallow_depths) {
        int shallow_value = d0_value;
        uint64_t shallow_nodes = 0;
        uint64_t shallow_time = 0;
        if (shallow_depth > 0) {
            transposition_table.init();
            bool searching = true;
            global_searching = true;
            const uint64_t start = tim();
            Search shallow_search(&board, MPC_100_LEVEL, false, false);
            const std::pair<int, int> result = first_nega_scout_legal(
                &shallow_search,
                -SCORE_MAX,
                SCORE_MAX,
                shallow_depth,
                false,
                std::vector<Clog_result>(),
                legal,
                start,
                &searching
            );
            if (!searching) {
                return 4;
            }
            shallow_value = result.first;
            shallow_nodes = shallow_search.n_nodes;
            shallow_time = tim() - start;
        }
        std::cout
            << "END_PROBCUT_SHALLOW_V2\t"
            << deep_depth << '\t'
            << shallow_depth << '\t'
            << shallow_value << '\t'
            << shallow_nodes << '\t'
            << shallow_time << '\t'
            << d0_value << '\t'
            << pop_count_ull(legal)
            << '\n';
    }
    return 0;
}

std::pair<int, int> cold_root_search(const Board &board, int mpc_level, bool use_multi_thread) {
    transposition_table.init();
    bool searching = true;
    global_searching = true;
    const uint64_t start = tim();
    Search search(&board, (uint_fast8_t)mpc_level, use_multi_thread, false);
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
    std::cout << result.first << '\t' << idx_to_coord(result.second)
              << '\t' << search.n_nodes << '\t' << tim() - start;
    return result;
}

int compare_pv(const std::string &board_text, int mpc_level, int n_plies, int n_threads) {
    Board board;
    if (!board.from_str(board_text)) {
        std::cerr << "invalid board\n";
        return 2;
    }
    std::cout << "ply\tempty\tmpc_value\tmpc_move\tmpc_nodes\tmpc_ms"
              << "\texact_value\texact_move\texact_nodes\texact_ms\tboard\n";
    for (int ply = 0; ply < n_plies && board.get_legal(); ++ply) {
        const std::string current_board = board.to_str();
        std::cout << ply << '\t' << (HW2 - board.n_discs()) << '\t';
        const std::pair<int, int> mpc_result = cold_root_search(board, mpc_level, n_threads > 1);
        std::cout << '\t';
        const std::pair<int, int> exact_result = cold_root_search(board, MPC_100_LEVEL, n_threads > 1);
        std::cout << '\t' << current_board << '\n';

        const int move = mpc_result.second;
        if ((board.get_legal() & (1ULL << move)) == 0) {
            std::cerr << "invalid MPC move at ply " << ply << '\n';
            return 4;
        }
        Flip flip;
        calc_flip(&flip, &board, move);
        board.move_board(&flip);
        if (board.get_legal() == 0) {
            board.pass();
            if (board.get_legal() == 0) {
                break;
            }
        }
    }
    return 0;
}

int label(
    const std::string &board_text,
    int deep_depth,
    int shallow_depth,
    int boundary,
    int threshold,
    const std::string &direction,
    int n_threads
) {
    Board board;
    if (!board.from_str(board_text)) {
        std::cerr << "invalid board\n";
        return 2;
    }
    const uint64_t legal = board.get_legal();
    if (legal == 0) {
        std::cerr << "context has no legal move\n";
        return 2;
    }

    bool searching = true;
    global_searching = true;
    const uint64_t shallow_start = tim();
    Search shallow_search(&board, MPC_100_LEVEL, false, false);
    const std::pair<int, int> shallow_result = first_nega_scout_legal(
        &shallow_search,
        -SCORE_MAX,
        SCORE_MAX,
        shallow_depth,
        false,
        std::vector<Clog_result>(),
        legal,
        shallow_start,
        &searching
    );
    const uint64_t shallow_time = tim() - shallow_start;

    const bool candidate = direction == "high"
        ? shallow_result.first >= threshold
        : shallow_result.first <= threshold;
    if (!candidate) {
        std::cout
            << "shallow_value\tshallow_nodes\tshallow_ms\tcandidate\tdeep_bound"
            << "\tcorrect\tdeep_nodes\tdeep_ms\n"
            << shallow_result.first << '\t'
            << shallow_search.n_nodes << '\t'
            << shallow_time << "\t0\t0\t1\t0\t0\n";
        return 0;
    }
    if (n_threads == 0) {
        std::cout
            << "shallow_value\tshallow_nodes\tshallow_ms\tcandidate\tdeep_bound"
            << "\tcorrect\tdeep_nodes\tdeep_ms\n"
            << shallow_result.first << '\t'
            << shallow_search.n_nodes << '\t'
            << shallow_time << "\t1\t0\t0\t0\t0\n";
        return 0;
    }

    const uint64_t deep_start = tim();
    Search deep_search(&board, MPC_100_LEVEL, n_threads > 1, false);
    int deep_bound;
    bool correct;
    if (direction == "high") {
        deep_bound = nega_alpha_ordering_nws(
            &deep_search,
            boundary - 1,
            deep_depth,
            false,
            legal,
            true,
            &searching
        );
        correct = deep_bound >= boundary;
    } else if (direction == "low") {
        deep_bound = nega_alpha_ordering_nws(
            &deep_search,
            boundary,
            deep_depth,
            false,
            legal,
            true,
            &searching
        );
        correct = deep_bound <= boundary;
    } else {
        std::cerr << "direction must be high or low\n";
        return 2;
    }
    const uint64_t deep_time = tim() - deep_start;

    std::cout
        << "shallow_value\tshallow_nodes\tshallow_ms\tcandidate\tdeep_bound\tcorrect"
        << "\tdeep_nodes\tdeep_ms\n"
        << shallow_result.first << '\t'
        << shallow_search.n_nodes << '\t'
        << shallow_time << "\t1\t"
        << deep_bound << '\t'
        << (correct ? 1 : 0) << '\t'
        << deep_search.n_nodes << '\t'
        << deep_time << '\n';
    return searching ? 0 : 4;
}

int batch_probe(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "cannot open context file\n";
        return 2;
    }
    std::cout
        << "index\tboard\tdeep_depth\tshallow_depth\troot_distance\tdirection"
        << "\tboundary\tthreshold\tshallow_value\tcandidate\tshallow_nodes\tshallow_ms\n";
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        if (!line.starts_with("END_PROBCUT_CONTEXT_V2\t")) {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, '\t')) {
            fields.emplace_back(field);
        }
        if (fields.size() != 15) {
            continue;
        }
        Board board;
        if (!board.from_str(fields[1])) {
            continue;
        }
        const int deep_depth = std::stoi(fields[2]);
        const int shallow_depth = std::stoi(fields[3]);
        const int root_distance = std::stoi(fields[5]);
        const int alpha = std::stoi(fields[6]);
        const int beta = std::stoi(fields[7]);
        const std::string &direction = fields[8];
        const int threshold = std::stoi(fields[12]);
        const int boundary = direction == "high" ? beta : alpha;
        bool searching = true;
        const uint64_t start = tim();
        Search search(&board, MPC_100_LEVEL, false, false);
        const std::pair<int, int> result = first_nega_scout_legal(
            &search,
            -SCORE_MAX,
            SCORE_MAX,
            shallow_depth,
            false,
            std::vector<Clog_result>(),
            board.get_legal(),
            start,
            &searching
        );
        const bool candidate = direction == "high"
            ? result.first >= threshold
            : result.first <= threshold;
        std::cout
            << index++ << '\t' << fields[1]
            << '\t' << deep_depth
            << '\t' << shallow_depth
            << '\t' << root_distance
            << '\t' << direction
            << '\t' << boundary
            << '\t' << threshold
            << '\t' << result.first
            << '\t' << (candidate ? 1 : 0)
            << '\t' << search.n_nodes
            << '\t' << tim() - start
            << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: end_mpc_context_tool <trace|score-grid|shallow-grid|label> ...\n";
        return 2;
    }
    const std::string command = argv[1];
    int n_threads = 1;
    if (command == "label" && argc == 9) {
        n_threads = std::atoi(argv[8]);
    } else if (command == "compare-pv" && argc == 6) {
        n_threads = std::atoi(argv[5]);
    }
    if (!initialize_engine(n_threads, 20)) {
        std::cerr << "engine initialization failed\n";
        return 3;
    }
    if (command == "trace" && 4 <= argc && argc <= 7) {
        return trace(
            argv[2],
            std::atoi(argv[3]),
            argc >= 5 ? std::atoi(argv[4]) : 16,
            argc >= 6 ? std::atoi(argv[5]) : 3,
            argc >= 7 ? std::atoi(argv[6]) : HW2
        );
    }
    if (command == "score-grid" && argc == 5) {
        return score_grid(argv[2], std::atoi(argv[3]), argv[4]);
    }
    if (command == "shallow-grid" && argc == 5) {
        return shallow_grid(argv[2], std::atoi(argv[3]), argv[4]);
    }
    if (command == "compare-pv" && argc == 6) {
        return compare_pv(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), n_threads);
    }
    if (command == "batch-probe" && argc == 3) {
        return batch_probe(argv[2]);
    }
    if (command == "label" && argc == 9) {
        return label(
            argv[2],
            std::atoi(argv[3]),
            std::atoi(argv[4]),
            std::atoi(argv[5]),
            std::atoi(argv[6]),
            argv[7],
            n_threads
        );
    }
    std::cerr << "invalid arguments\n";
    return 2;
}
