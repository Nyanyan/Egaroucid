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
            [reference-cache] [forced-cache] [summary-only:0|1]
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

struct FixedSearchResult {
    int value = SCORE_UNDEFINED;
    int policy = MOVE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed = 0;
    uint64_t pv_extensions = 0;
    bool complete = false;
};

#if defined(EGAROUCID_MPC_RUNTIME_TUNING)
bool parse_depth_values(const char *name, int *values) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return true;
    }
    std::string source(text);
    size_t begin = 0;
    while (begin < source.size()) {
        const size_t end = source.find(',', begin);
        const std::string field = source.substr(begin, end - begin);
        const size_t separator = field.find(':');
        if (separator == std::string::npos) {
            return false;
        }
        const int depth = std::atoi(field.substr(0, separator).c_str());
        const int value = std::atoi(field.substr(separator + 1).c_str());
        if (depth < 0 || depth >= HW2 - 2 || value < 0) {
            return false;
        }
        values[depth] = value;
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return true;
}

bool parse_coefficients(const char *name, double *coefficients) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return true;
    }
    std::string source(text);
    size_t begin = 0;
    for (int index = 0; index < 7; ++index) {
        const size_t end = source.find(',', begin);
        if (begin >= source.size() || (index < 6 && end == std::string::npos)) {
            return false;
        }
        coefficients[index] = std::strtod(
            source.substr(begin, end - begin).c_str(), nullptr
        );
        if (!std::isfinite(coefficients[index])) {
            return false;
        }
        begin = end == std::string::npos ? source.size() : end + 1;
    }
    return begin == source.size();
}

bool configure_runtime_mpc() {
    mid_mpc_runtime_tuning_reset();
    return
        parse_depth_values(
            "EGAROUCID_MID_MPC_SHALLOW_DEPTHS",
            mid_mpc_runtime_shallow_depth
        ) &&
        parse_depth_values(
            "EGAROUCID_MID_MPC_HIGH_GATE_SLACK",
            mid_mpc_runtime_high_gate_slack
        ) &&
        parse_depth_values(
            "EGAROUCID_MID_MPC_LOW_GATE_SLACK",
            mid_mpc_runtime_low_gate_slack
        ) &&
        parse_coefficients(
            "EGAROUCID_MID_MPC_HIGH_COEFFICIENTS",
            mid_mpc_runtime_high_coefficients
        ) &&
        parse_coefficients(
            "EGAROUCID_MID_MPC_LOW_COEFFICIENTS",
            mid_mpc_runtime_low_coefficients
        );
}
#endif

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
    out.pv_extensions = search.n_pv_extensions;
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

std::unordered_map<std::string, FixedSearchResult> read_reference_cache(
    const std::string &path,
    int reference_depth
) {
    std::unordered_map<std::string, FixedSearchResult> result;
    std::ifstream input(path);
    std::string line;
    std::getline(input, line); // header
    while (std::getline(input, line)) {
        const size_t board_end = line.find('\t');
        if (board_end == std::string::npos) continue;
        const std::string board = line.substr(0, board_end);
        std::vector<std::string> fields;
        size_t begin = board_end + 1;
        while (begin <= line.size()) {
            const size_t end = line.find('\t', begin);
            fields.emplace_back(line.substr(begin, end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        if (fields.size() != 5 || std::atoi(fields[0].c_str()) != reference_depth) {
            continue;
        }
        FixedSearchResult cached;
        cached.value = std::atoi(fields[1].c_str());
        cached.policy = std::atoi(fields[2].c_str());
        cached.nodes = std::strtoull(fields[3].c_str(), nullptr, 10);
        cached.elapsed = std::strtoull(fields[4].c_str(), nullptr, 10);
        cached.complete = true;
        result.emplace(board, cached);
    }
    return result;
}

std::string forced_cache_key(const std::string &board, int reference_depth, int policy) {
    return board + '\t' + std::to_string(reference_depth) + '\t' + std::to_string(policy);
}

std::unordered_map<std::string, FixedSearchResult> read_forced_cache(
    const std::string &path
) {
    std::unordered_map<std::string, FixedSearchResult> result;
    std::ifstream input(path);
    std::string line;
    std::getline(input, line); // header
    while (std::getline(input, line)) {
        std::vector<std::string> fields;
        size_t begin = 0;
        while (begin <= line.size()) {
            const size_t end = line.find('\t', begin);
            fields.emplace_back(line.substr(begin, end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        if (fields.size() != 6) continue;
        const int reference_depth = std::atoi(fields[1].c_str());
        const int policy = std::atoi(fields[2].c_str());
        FixedSearchResult cached;
        cached.policy = policy;
        cached.value = std::atoi(fields[3].c_str());
        cached.nodes = std::strtoull(fields[4].c_str(), nullptr, 10);
        cached.elapsed = std::strtoull(fields[5].c_str(), nullptr, 10);
        cached.complete = true;
        result.emplace(forced_cache_key(fields[0], reference_depth, policy), cached);
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 7 || argc > 11) {
        std::cerr
            << "usage: " << argv[0]
            << " <positions> <depth> <mpc-level:0..6> <reference-depth>"
               " <threads> <hash-level> [position-limit] [reference-cache]"
               " [forced-cache] [summary-only:0|1]\n";
        return 2;
    }

    const std::string positions_path = argv[1];
    const int depth = std::atoi(argv[2]);
    const int mpc_level = std::atoi(argv[3]);
    const int reference_depth = std::atoi(argv[4]);
    const int n_threads = std::atoi(argv[5]);
    const int hash_level = std::atoi(argv[6]);
    const int position_limit = argc >= 8 ? std::atoi(argv[7]) : 0;
    const std::string reference_cache_path = argc >= 9 ? argv[8] : "";
    const std::string forced_cache_path = argc >= 10 ? argv[9] : "";
    const bool summary_only = argc >= 11 && std::atoi(argv[10]) != 0;
    const bool reference_cache_exists =
        !reference_cache_path.empty() && std::filesystem::exists(reference_cache_path);
    std::unordered_map<std::string, FixedSearchResult> reference_cache;
    if (reference_cache_exists) {
        reference_cache = read_reference_cache(reference_cache_path, reference_depth);
    }
    std::ofstream reference_cache_output;
    if (!reference_cache_path.empty() && !reference_cache_exists) {
        reference_cache_output.open(reference_cache_path);
        if (!reference_cache_output) {
            std::cerr << "cannot create reference cache\n";
            return 2;
        }
        reference_cache_output << "board\treference_depth\tvalue\tpolicy\tnodes\ttime_ms\n";
    }
    const bool forced_cache_exists =
        !forced_cache_path.empty() && std::filesystem::exists(forced_cache_path);
    std::unordered_map<std::string, FixedSearchResult> forced_cache;
    if (forced_cache_exists) {
        forced_cache = read_forced_cache(forced_cache_path);
    }
    std::ofstream forced_cache_output;
    if (!forced_cache_path.empty()) {
        forced_cache_output.open(forced_cache_path, std::ios::app);
        if (!forced_cache_output) {
            std::cerr << "cannot open forced-move cache\n";
            return 2;
        }
        if (!forced_cache_exists) {
            forced_cache_output << "board\treference_depth\tpolicy\tvalue\tnodes\ttime_ms\n";
        }
    }
    if (
        depth <= 0 || reference_depth < depth ||
        mpc_level < 0 || mpc_level >= N_SELECTIVITY_LEVEL ||
        n_threads <= 0 || hash_level < 0 || hash_level >= N_HASH_LEVEL
    ) {
        std::cerr << "invalid numeric argument\n";
        return 2;
    }

#if defined(EGAROUCID_MPC_RUNTIME_TUNING)
    if (!configure_runtime_mpc()) {
        std::cerr << "invalid MPC runtime tuning environment variable\n";
        return 2;
    }
#endif

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
    uint64_t candidate_pv_extensions_sum = 0;

    if (!summary_only) {
        std::cout
            << "index\tempties\tstatic_value\tcandidate_value\tcandidate_move"
               "\tcandidate_nodes\tcandidate_time_ms\tcandidate_pv_extensions"
               "\treference_value\treference_move"
               "\treference_nodes\treference_time_ms\tcandidate_reference_value"
               "\tregret\tagree\tcomplete\n";
    }

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
        FixedSearchResult reference;
        const auto cached = reference_cache.find(positions[i]);
        if (cached != reference_cache.end()) {
            reference = cached->second;
        } else {
            if (reference_cache_exists) {
                std::cerr << "missing board in reference cache at input line " << (i + 1) << '\n';
                return 2;
            }
            reference = run_fixed_search(
                board, reference_depth, MPC_100_LEVEL, n_threads, legal
            );
            if (reference_cache_output) {
                reference_cache_output
                    << positions[i] << '\t' << reference_depth << '\t'
                    << reference.value << '\t' << reference.policy << '\t'
                    << reference.nodes << '\t' << reference.elapsed << '\n';
                reference_cache_output.flush();
            }
        }

        int candidate_reference_value = reference.value;
        FixedSearchResult forced;
        forced.complete = true;
        if (candidate.policy != reference.policy) {
            const std::string key = forced_cache_key(
                positions[i], reference_depth, candidate.policy
            );
            const auto cached_forced = forced_cache.find(key);
            if (cached_forced != forced_cache.end()) {
                forced = cached_forced->second;
            } else {
                forced = run_fixed_search(
                    board,
                    reference_depth,
                    MPC_100_LEVEL,
                    n_threads,
                    1ULL << candidate.policy
                );
                forced_cache.emplace(key, forced);
                if (forced_cache_output) {
                    forced_cache_output
                        << positions[i] << '\t' << reference_depth << '\t'
                        << candidate.policy << '\t' << forced.value << '\t'
                        << forced.nodes << '\t' << forced.elapsed << '\n';
                    forced_cache_output.flush();
                }
            }
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
            candidate_pv_extensions_sum += candidate.pv_extensions;
            reference_time_sum += reference.elapsed;
        }

        if (!summary_only) {
            std::cout
                << (i + 1) << '\t'
                << empties << '\t'
                << static_value << '\t'
                << candidate.value << '\t'
                << idx_to_coord(candidate.policy) << '\t'
                << candidate.nodes << '\t'
                << candidate.elapsed << '\t'
                << candidate.pv_extensions << '\t'
                << reference.value << '\t'
                << idx_to_coord(reference.policy) << '\t'
                << reference.nodes << '\t'
                << reference.elapsed << '\t'
                << candidate_reference_value << '\t'
                << regret << '\t'
                << static_cast<int>(agree) << '\t'
                << static_cast<int>(complete) << '\n';
        }
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
              << " candidate_pv_extensions=" << candidate_pv_extensions_sum
              << " reference_time_ms=" << reference_time_sum
              << '\n';
    return complete_count == static_cast<int>(positions.size()) ? 0 : 4;
}
