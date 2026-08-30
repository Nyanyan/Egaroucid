// 独立holdoutで不足している浅い探索値だけを収集する補助ツール。
// 深い探索値は既存の100%探索TSVから読み、再計算しない。

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../src/engine/engine_all.hpp"

namespace {

struct Reference {
    int n_discs = 0;
    int static_value = 0;
    int deep_value = 0;
    uint64_t deep_nodes = 0;
    uint64_t deep_time_ms = 0;
    int legal_count = 0;
};

struct Score {
    int value = SCORE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed = 0;
};

std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.emplace_back(field);
    }
    return fields;
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

bool initialize_engine(int hash_level) {
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
    if (!hash_resize(DEFAULT_HASH_LEVEL, hash_level, "./", false)) {
        return false;
    }
    stability_init();
    return evaluate_init("./resources/eval.egev2", "./resources/eval_move_ordering_end.egev", false);
}

Score search_score(const Board &board, int depth) {
    transposition_table.init();
    global_searching = true;
    bool searching = true;
    const uint64_t start = tim();
    Search search(&board, MPC_100_LEVEL, false, false);
    search.thread_id = THREAD_ID_NONE;
    const std::pair<int, int> result = first_nega_scout_legal(
        &search, -SCORE_MAX, SCORE_MAX, depth, false,
        std::vector<Clog_result>(), board.get_legal(), start, &searching
    );
    if (!searching || !global_searching) {
        return Score{};
    }
    return Score{result.first, search.n_nodes, tim() - start};
}

std::unordered_map<std::string, Reference> load_references(const std::string &path, int deep_depth) {
    std::ifstream input(path);
    std::string line;
    std::unordered_map<std::string, Reference> result;
    std::getline(input, line); // header
    while (std::getline(input, line)) {
        const std::vector<std::string> fields = split(line);
        if (fields.size() < 14 || std::atoi(fields[3].c_str()) != deep_depth || std::atoi(fields[4].c_str()) != 0) {
            continue;
        }
        result[fields[1]] = Reference{
            std::atoi(fields[2].c_str()), std::atoi(fields[5].c_str()), std::atoi(fields[7].c_str()),
            std::strtoull(fields[10].c_str(), nullptr, 10), std::strtoull(fields[12].c_str(), nullptr, 10),
            std::atoi(fields[13].c_str())
        };
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: collect_shallow_only <positions> <reference-tsv> <deep-depth> <shallow-depths> <hash-level> <limit>\n";
        return 2;
    }
    const int deep_depth = std::atoi(argv[3]);
    const std::vector<int> depths = parse_depths(argv[4], deep_depth);
    const int hash_level = std::atoi(argv[5]);
    const int limit = std::atoi(argv[6]);
    const auto references = load_references(argv[2], deep_depth);
    if (depths.empty() || references.empty() || !initialize_engine(hash_level)) {
        return 3;
    }
    std::ifstream input(argv[1]);
    std::string line;
    int index = 0;
    std::cout << "index\tboard\tn_discs\tdeep_depth\tshallow_depth\tstatic_value"
                 "\tshallow_value\tdeep_value\terror\tshallow_nodes\tdeep_nodes"
                 "\tshallow_time_ms\tdeep_time_ms\tlegal_count\n";
    while (std::getline(input, line) && index < limit) {
        if (line.empty()) {
            continue;
        }
        ++index;
        const auto iterator = references.find(line);
        if (iterator == references.end()) {
            std::cerr << "missing reference at line " << index << '\n';
            return 4;
        }
        Board board;
        if (!board.from_str(line)) {
            return 4;
        }
        const Reference &reference = iterator->second;
        for (const int depth : depths) {
            const Score shallow = depth == 0 ? Score{reference.static_value, 0, 0} : search_score(board, depth);
            std::cout << index << '\t' << line << '\t' << reference.n_discs << '\t' << deep_depth << '\t'
                      << depth << '\t' << reference.static_value << '\t' << shallow.value << '\t'
                      << reference.deep_value << '\t' << reference.deep_value - shallow.value << '\t'
                      << shallow.nodes << '\t' << reference.deep_nodes << '\t' << shallow.elapsed << '\t'
                      << reference.deep_time_ms << '\t' << reference.legal_count << '\n';
        }
    }
    return 0;
}
