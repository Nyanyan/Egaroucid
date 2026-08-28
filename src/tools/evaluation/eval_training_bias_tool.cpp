/*
    Egaroucid Project

    Reproducible diagnostics for evaluation-training bias.

    Input is a headered TSV.  It must contain `sample_id` and `board`; every
    other input column is copied to both output files with an `input_` prefix.
    `board` uses Egaroucid's usual 64 cells plus side-to-move notation, e.g.
        ---------------------------OX------XO--------------------------- X

    Build (from bin/ with MinGW g++):
        g++ -O3 -march=native -mtune=native -std=c++20 -pthread \
            ../src/tools/evaluation/eval_training_bias_tool.cpp \
            -o eval_training_bias_tool.exe

    Run (from bin/, whose resources/ directory supplies the weights/hashes):
        eval_training_bias_tool.exe input.tsv root.tsv children.tsv \
            MAX_EXACT_EMPTIES [HASH_LEVEL=20] [POSITION_LIMIT=0] [ENGINE_ROOT=./] \
            [EXACT_CHILDREN=1]

    The root TSV contains static evaluation, forced phase-1/phase/phase+1
    diagnostics, fixed depths 0,1,2,4,5,6,8,9,10, and an exact result when
    n_empties <= MAX_EXACT_EMPTIES.  The children TSV contains every legal
    child in the parent's point of view, including exact sibling ranks when
    exact search was requested and completed.  A forced phase is an artificial
    diagnostic; it is not a position that the runtime evaluator normally uses.
    Each fixed-depth PV is reconstructed immediately after the cold root search
    by repeating the same full-window selection at successive children while
    retaining that root's TT.  A tied continuation can therefore differ from
    the TT path retained by the first root search while remaining an equally
    selected PV under the stated conditions.  PV-only searches use a separate
    memoization cache and never replace the cold root node/time measurement.

    Search controls are intentionally fixed here:
      * one thread (no root/YBWC parallel search)
      * no opening book (the low-level search API is called directly)
      * MPC_100_LEVEL (MPC disabled; this also makes mid_nws_lmr_reduction 0)
      * USE_PV_EXTENSION=false, set before including the engine
      * fixed depth, full root window [-SCORE_MAX, SCORE_MAX]
      * cold global TT and cold endgame local TT before every actual search

    Searches are memoized by exact (player, opponent, depth) inside this
    process.  Duplicate samples still get their own rows, but an identical
    search is run only once; cache_hit columns identify reused measurements.
    Cached node/time fields are the first cold-search measurement, not the cost
    of the cache lookup.
    If resources/hash/hashN.eghs is absent, the engine's random-hash fallback is
    seeded with 0x00E6A202 so ordering remains reproducible.

    There is deliberately no time limit.  A result whose searching flag or
    score reports interruption is written with complete=0 and a blank value,
    and is never included implicitly as a numeric score.  Exact searches above
    MAX_EXACT_EMPTIES are likewise marked skipped with blank numeric fields.
*/

#ifndef USE_PV_EXTENSION
#define USE_PV_EXTENSION false
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../engine/engine_all.hpp"

static_assert(!USE_PV_EXTENSION, "this diagnostic must be built without PV extension");

namespace {

constexpr std::array<int, 9> FIXED_DEPTHS = {0, 1, 2, 4, 5, 6, 8, 9, 10};
constexpr uint32_t HASH_FALLBACK_SEED = 0x00E6A202U;

constexpr uint64_t bit_at(int coord) {
    return 1ULL << coord;
}

constexpr uint64_t CORNER_MASK =
    bit_at(COORD_A1) | bit_at(COORD_H1) |
    bit_at(COORD_A8) | bit_at(COORD_H8);
constexpr uint64_t X_MASK =
    bit_at(COORD_B2) | bit_at(COORD_G2) |
    bit_at(COORD_B7) | bit_at(COORD_G7);
constexpr uint64_t C_MASK =
    bit_at(COORD_B1) | bit_at(COORD_A2) |
    bit_at(COORD_G1) | bit_at(COORD_H2) |
    bit_at(COORD_A7) | bit_at(COORD_B8) |
    bit_at(COORD_H7) | bit_at(COORD_G8);

struct PositionKey {
    uint64_t player = 0;
    uint64_t opponent = 0;

    bool operator==(const PositionKey &other) const {
        return player == other.player && opponent == other.opponent;
    }
};

struct PositionKeyHash {
    static uint64_t mix(uint64_t x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    size_t operator()(const PositionKey &key) const {
        return static_cast<size_t>(mix(key.player) ^ (mix(key.opponent) << 1));
    }
};

struct SearchKey {
    PositionKey position;
    int depth = 0;

    bool operator==(const SearchKey &other) const {
        return depth == other.depth && position == other.position;
    }
};

struct SearchKeyHash {
    size_t operator()(const SearchKey &key) const {
        const size_t board_hash = PositionKeyHash{}(key.position);
        return board_hash ^ (static_cast<size_t>(key.depth + 0x9e37) * 0x9e3779b1U);
    }
};

struct ForcedEvaluation {
    int phase = -1;
    int raw = 0;
    int value = 0;
    bool valid = false;
};

struct StaticDiagnostics {
    std::string status;
    bool available = false;
    int value = 0;
    ForcedEvaluation previous;
    ForcedEvaluation actual;
    ForcedEvaluation next;
    bool actual_matches_runtime = false;
};

struct SearchMeasurement {
    std::string status;
    bool complete = false;
    bool is_end_search = false;
    int value = SCORE_UNDEFINED;
    int best_move = MOVE_UNDEFINED;
    uint64_t nodes = 0;
    uint64_t elapsed_ms = 0;
    uint64_t pv_extensions = 0;
};

struct CachedSearchMeasurement {
    SearchMeasurement measurement;
    bool cache_hit = false;
};

struct CachedStaticDiagnostics {
    StaticDiagnostics diagnostics;
    bool cache_hit = false;
};

struct PvTrace {
    std::string status;
    bool complete = false;
    std::vector<int> moves;
    int pass_count = 0;
    int legal_move_count = 0;
    int cache_hits = 0;
    int searches = 0;
    Board leaf;
    int leaf_view_sign = 1;
    int leaf_static_value = 0;
    int leaf_static_value_root = 0;
    std::string leaf_static_status;
    bool leaf_static_available = false;
    bool leaf_static_cache_hit = false;
    bool leaf_terminal = false;
};

struct InputSchema {
    std::vector<std::string> header;
    size_t sample_id_idx = 0;
    size_t board_idx = 0;
    std::vector<size_t> extra_indices;
};

struct InputRow {
    size_t input_line = 0;
    std::string sample_id;
    std::string board_text;
    std::vector<std::string> extra_values;
};

struct ChildDiagnostics {
    std::string edge_kind;
    int move = MOVE_UNDEFINED;
    int flip_count = 0;
    Board board;
    std::string static_status;
    bool static_available = false;
    int static_value_parent = 0;
    bool static_cache_hit = false;
    int static_rank = 0;
    int static_loss = 0;
    bool static_best = false;
    std::string exact_status;
    bool exact_complete = false;
    bool exact_cache_hit = false;
    int exact_value_parent = SCORE_UNDEFINED;
    uint64_t exact_nodes = 0;
    uint64_t exact_elapsed_ms = 0;
    int exact_rank = 0;
    int exact_loss = 0;
    bool exact_best = false;
};

std::unordered_map<PositionKey, StaticDiagnostics, PositionKeyHash> static_cache;
std::unordered_map<SearchKey, SearchMeasurement, SearchKeyHash> search_cache;
std::unordered_map<SearchKey, SearchMeasurement, SearchKeyHash> pv_search_cache;
std::unordered_map<SearchKey, PvTrace, SearchKeyHash> fixed_pv_cache;

PositionKey position_key(const Board &board) {
    return PositionKey{board.player, board.opponent};
}

std::vector<std::string> split_tsv(const std::string &line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    return fields;
}

void remove_trailing_cr(std::string *text) {
    if (!text->empty() && text->back() == '\r') {
        text->pop_back();
    }
}

void remove_utf8_bom(std::string *text) {
    if (text->size() >= 3 &&
        static_cast<unsigned char>((*text)[0]) == 0xef &&
        static_cast<unsigned char>((*text)[1]) == 0xbb &&
        static_cast<unsigned char>((*text)[2]) == 0xbf) {
        text->erase(0, 3);
    }
}

int parse_int(const char *text, const std::string &name) {
    try {
        size_t consumed = 0;
        const std::string value_text(text);
        const long long value = std::stoll(value_text, &consumed, 10);
        if (consumed != value_text.size() ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("range");
        }
        return static_cast<int>(value);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid " + name + ": " + text);
    }
}

std::string normalize_engine_root(const std::string &text) {
    std::filesystem::path root(text);
    std::string result = root.generic_string();
    if (result.empty()) {
        result = ".";
    }
    if (result.back() != '/') {
        result += '/';
    }
    return result;
}

std::string hex64(uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << value;
    return stream.str();
}

std::string move_text(int move) {
    if (move == MOVE_PASS) {
        return "pass";
    }
    if (0 <= move && move < HW2) {
        return idx_to_coord(move);
    }
    return "";
}

std::string int_field(int value) {
    return std::to_string(value);
}

std::string uint_field(uint64_t value) {
    return std::to_string(value);
}

std::string bool_field(bool value) {
    return value ? "1" : "0";
}

void write_tsv_row(std::ofstream *output, const std::vector<std::string> &fields) {
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            *output << '\t';
        }
        *output << fields[i];
    }
    *output << '\n';
}

int board_phase(const Board &board) {
    return (board.n_discs() - 4) / PHASE_N_DISCS;
}

ForcedEvaluation evaluate_at_forced_phase(const Board &board, int phase) {
    ForcedEvaluation result;
    result.phase = phase;
    if (phase < 0 || N_PHASES <= phase) {
        return result;
    }
    Board copy = board.copy();
    Search search(&copy, MPC_100_LEVEL, false, false);
    const int player_discs = pop_count_ull(search.board.player);
#if USE_SIMD_EVALUATION
    int raw = calc_pattern(phase, &search.eval.features[search.eval.feature_idx])
        + eval_num_arr[phase][player_discs];
    #if !USE_DIM0_ONLY_EVALUATION
    raw += eval_fm_calc(phase, &search.eval.features[search.eval.feature_idx]);
    #endif
#else
    int raw = calc_pattern(phase, &search.eval) + eval_num_arr[phase][player_discs];
    #if !USE_DIM0_ONLY_EVALUATION
    raw += eval_fm_calc(phase, &search.eval);
    #endif
#endif
    int value = raw + (raw >= 0 ? STEP_2 : -STEP_2);
    value /= STEP;
    value = std::clamp(value, -SCORE_MAX, SCORE_MAX);
    result.raw = raw;
    result.value = value;
    result.valid = true;
    return result;
}

StaticDiagnostics calculate_static_diagnostics(const Board &board) {
    const int phase = board_phase(board);
    StaticDiagnostics result;
    if (phase < 0 || N_PHASES <= phase) {
        result.status = "unavailable_phase_out_of_range";
        return result;
    }
    Board copy = board.copy();
    Search search(&copy, MPC_100_LEVEL, false, false);
    result.value = mid_evaluate_diff(&search);
    result.previous = evaluate_at_forced_phase(board, phase - 1);
    result.actual = evaluate_at_forced_phase(board, phase);
    result.next = evaluate_at_forced_phase(board, phase + 1);
    result.actual_matches_runtime = result.actual.valid && result.actual.value == result.value;
    result.status = "available";
    result.available = true;
    return result;
}

CachedStaticDiagnostics get_static_diagnostics(const Board &board) {
    const PositionKey key = position_key(board);
    const auto found = static_cache.find(key);
    if (found != static_cache.end()) {
        return CachedStaticDiagnostics{found->second, true};
    }
    StaticDiagnostics diagnostics = calculate_static_diagnostics(board);
    static_cache.emplace(key, diagnostics);
    return CachedStaticDiagnostics{diagnostics, false};
}

void clear_search_tables() {
    transposition_table.init();
    std::memset(lttable, 0, sizeof(lttable));
}

SearchMeasurement calculate_search(const Board &board, int depth, bool cold_tables = true) {
    SearchMeasurement measurement;
    const int n_empties = HW2 - board.n_discs();
    measurement.is_end_search = depth == n_empties;
    if (cold_tables) {
        clear_search_tables();
    }
    global_searching = true;
    bool searching = true;
    const uint64_t start = tim();

    Search search(&board, MPC_100_LEVEL, false, false);
    search.thread_id = THREAD_ID_NONE;
    const uint64_t legal = board.get_legal();
    int value = SCORE_UNDEFINED;
    int best_move = MOVE_UNDEFINED;

    if (legal != 0ULL) {
        const std::pair<int, int> result = first_nega_scout_legal(
            &search,
            -SCORE_MAX,
            SCORE_MAX,
            depth,
            measurement.is_end_search,
            std::vector<Clog_result>(),
            legal,
            start,
            &searching
        );
        value = result.first;
        best_move = result.second;
        measurement.status = "complete";
    } else {
        // first_nega_scout_legal rejects an empty legal mask.  Entering the
        // ordinary node function keeps a pass at zero consumed plies and also
        // handles two consecutive passes (including non-full terminal boards).
        search.configure_pv_extension(depth, measurement.is_end_search);
        value = nega_scout_node(
            &search,
            -SCORE_MAX,
            SCORE_MAX,
            depth,
            false,
            LEGAL_UNDEFINED,
            measurement.is_end_search,
            SEARCH_NODE_PV,
            &searching
        );
        if (calc_legal(board.opponent, board.player) == 0ULL) {
            measurement.status = "complete_terminal";
        } else {
            measurement.status = "complete_pass";
            best_move = MOVE_PASS;
        }
    }

    measurement.elapsed_ms = tim() - start;
    measurement.nodes = search.n_nodes;
    measurement.pv_extensions = search.n_pv_extensions;
    measurement.complete = searching && global_searching && value != SCORE_UNDEFINED;
    if (measurement.complete) {
        measurement.value = value;
        measurement.best_move = best_move;
    } else {
        measurement.status = "incomplete";
        measurement.value = SCORE_UNDEFINED;
        measurement.best_move = MOVE_UNDEFINED;
    }
    return measurement;
}

CachedSearchMeasurement get_search_measurement(const Board &board, int depth) {
    const SearchKey key{position_key(board), depth};
    const auto found = search_cache.find(key);
    if (found != search_cache.end()) {
        return CachedSearchMeasurement{found->second, true};
    }
    SearchMeasurement measurement = calculate_search(board, depth);
    search_cache.emplace(key, measurement);
    return CachedSearchMeasurement{measurement, false};
}

CachedSearchMeasurement get_pv_search_measurement(const Board &board, int depth) {
    const SearchKey key{position_key(board), depth};
    const auto cold_found = search_cache.find(key);
    if (cold_found != search_cache.end()) {
        return CachedSearchMeasurement{cold_found->second, true};
    }
    const auto pv_found = pv_search_cache.find(key);
    if (pv_found != pv_search_cache.end()) {
        return CachedSearchMeasurement{pv_found->second, true};
    }
    SearchMeasurement measurement = calculate_search(board, depth, false);
    pv_search_cache.emplace(key, measurement);
    return CachedSearchMeasurement{measurement, false};
}

bool initialize_engine(int hash_level, const std::string &engine_root) {
    // hash_init_rand() otherwise seeds from wall-clock time through
    // raw_myrandom.  A missing lower-level hash file is common in resources/;
    // fixing the fallback seed keeps tie-breaking/order reproducible.
    raw_myrandom.seed(HASH_FALLBACK_SEED);
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
    if (!hash_resize(DEFAULT_HASH_LEVEL, hash_level, engine_root, false)) {
        return false;
    }
    stability_init();
    return evaluate_init(
        engine_root + "resources/eval.egev2",
        engine_root + "resources/eval_move_ordering_end.egev",
        false
    );
}

InputSchema read_schema(std::ifstream *input) {
    std::string line;
    if (!std::getline(*input, line)) {
        throw std::runtime_error("input TSV is empty");
    }
    remove_trailing_cr(&line);
    InputSchema schema;
    schema.header = split_tsv(line);
    if (!schema.header.empty()) {
        remove_utf8_bom(&schema.header[0]);
    }
    bool found_sample_id = false;
    bool found_board = false;
    for (size_t i = 0; i < schema.header.size(); ++i) {
        if (schema.header[i] == "sample_id") {
            if (found_sample_id) {
                throw std::runtime_error("duplicate sample_id column");
            }
            schema.sample_id_idx = i;
            found_sample_id = true;
        } else if (schema.header[i] == "board") {
            if (found_board) {
                throw std::runtime_error("duplicate board column");
            }
            schema.board_idx = i;
            found_board = true;
        } else {
            schema.extra_indices.emplace_back(i);
        }
    }
    if (!found_sample_id || !found_board) {
        throw std::runtime_error("input TSV header must contain sample_id and board");
    }
    return schema;
}

bool read_input_row(
    std::ifstream *input,
    const InputSchema &schema,
    size_t *line_number,
    InputRow *row
) {
    std::string line;
    while (std::getline(*input, line)) {
        ++*line_number;
        remove_trailing_cr(&line);
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tsv(line);
        if (fields.size() != schema.header.size()) {
            throw std::runtime_error(
                "TSV field count mismatch at line " + std::to_string(*line_number) +
                ": got " + std::to_string(fields.size()) +
                ", expected " + std::to_string(schema.header.size())
            );
        }
        row->input_line = *line_number;
        row->sample_id = fields[schema.sample_id_idx];
        row->board_text = fields[schema.board_idx];
        row->extra_values.clear();
        for (const size_t idx : schema.extra_indices) {
            row->extra_values.emplace_back(std::move(fields[idx]));
        }
        return true;
    }
    return false;
}

void append_input_fields(std::vector<std::string> *fields, const InputRow &row) {
    fields->emplace_back(row.sample_id);
    fields->emplace_back(row.board_text);
    fields->emplace_back(std::to_string(row.input_line));
    fields->insert(fields->end(), row.extra_values.begin(), row.extra_values.end());
}

void append_input_header(std::vector<std::string> *fields, const InputSchema &schema) {
    fields->emplace_back("sample_id");
    fields->emplace_back("board");
    fields->emplace_back("input_line");
    for (const size_t idx : schema.extra_indices) {
        const std::string name = schema.header[idx].empty()
            ? "column_" + std::to_string(idx + 1)
            : schema.header[idx];
        fields->emplace_back("input_" + name);
    }
}

void append_forced_evaluation(
    std::vector<std::string> *fields,
    const ForcedEvaluation &evaluation
) {
    if (!evaluation.valid) {
        fields->insert(fields->end(), {"", "", ""});
        return;
    }
    fields->emplace_back(int_field(evaluation.phase));
    fields->emplace_back(int_field(evaluation.raw));
    fields->emplace_back(int_field(evaluation.value));
}

void append_measurement_header(std::vector<std::string> *fields, const std::string &prefix) {
    fields->emplace_back(prefix + "_status");
    fields->emplace_back(prefix + "_complete");
    fields->emplace_back(prefix + "_cache_hit");
    fields->emplace_back(prefix + "_is_end_search");
    fields->emplace_back(prefix + "_value");
    fields->emplace_back(prefix + "_best_move");
    fields->emplace_back(prefix + "_best_move_idx");
    fields->emplace_back(prefix + "_nodes");
    fields->emplace_back(prefix + "_elapsed_ms");
    fields->emplace_back(prefix + "_pv_extensions");
}

void append_pv_header(std::vector<std::string> *fields, const std::string &prefix) {
    fields->emplace_back(prefix + "_pv_status");
    fields->emplace_back(prefix + "_pv_complete");
    fields->emplace_back(prefix + "_pv_moves");
    fields->emplace_back(prefix + "_pv_legal_move_count");
    fields->emplace_back(prefix + "_pv_pass_count");
    fields->emplace_back(prefix + "_pv_searches");
    fields->emplace_back(prefix + "_pv_cache_hits");
    fields->emplace_back(prefix + "_pv_leaf_board");
    fields->emplace_back(prefix + "_pv_leaf_player_bits_hex");
    fields->emplace_back(prefix + "_pv_leaf_opponent_bits_hex");
    fields->emplace_back(prefix + "_pv_leaf_n_empties");
    fields->emplace_back(prefix + "_pv_leaf_phase");
    fields->emplace_back(prefix + "_pv_leaf_static_status");
    fields->emplace_back(prefix + "_pv_leaf_static_available");
    fields->emplace_back(prefix + "_pv_leaf_static_value");
    fields->emplace_back(prefix + "_pv_leaf_static_value_root");
    fields->emplace_back(prefix + "_pv_leaf_static_cache_hit");
    fields->emplace_back(prefix + "_pv_leaf_terminal");
}

std::string pv_moves_text(const std::vector<int> &moves) {
    std::string result;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i != 0) {
            result += ' ';
        }
        result += move_text(moves[i]);
    }
    return result;
}

void append_pv(std::vector<std::string> *fields, const PvTrace &trace) {
    fields->emplace_back(trace.status);
    fields->emplace_back(bool_field(trace.complete));
    fields->emplace_back(pv_moves_text(trace.moves));
    fields->emplace_back(int_field(trace.legal_move_count));
    fields->emplace_back(int_field(trace.pass_count));
    fields->emplace_back(int_field(trace.searches));
    fields->emplace_back(int_field(trace.cache_hits));
    if (!trace.complete) {
        fields->insert(
            fields->end(),
            {"", "", "", "", "", "", "", "", "", "", ""}
        );
        return;
    }
    fields->emplace_back(trace.leaf.to_str());
    fields->emplace_back(hex64(trace.leaf.player));
    fields->emplace_back(hex64(trace.leaf.opponent));
    fields->emplace_back(int_field(HW2 - trace.leaf.n_discs()));
    fields->emplace_back(int_field(board_phase(trace.leaf)));
    fields->emplace_back(trace.leaf_static_status);
    fields->emplace_back(bool_field(trace.leaf_static_available));
    if (trace.leaf_static_available) {
        fields->emplace_back(int_field(trace.leaf_static_value));
        fields->emplace_back(int_field(trace.leaf_static_value_root));
    } else {
        fields->insert(fields->end(), {"", ""});
    }
    fields->emplace_back(bool_field(trace.leaf_static_cache_hit));
    fields->emplace_back(bool_field(trace.leaf_terminal));
}

void append_measurement(
    std::vector<std::string> *fields,
    const SearchMeasurement &measurement,
    bool cache_hit
) {
    fields->emplace_back(measurement.status);
    fields->emplace_back(bool_field(measurement.complete));
    fields->emplace_back(bool_field(cache_hit));
    fields->emplace_back(bool_field(measurement.is_end_search));
    if (measurement.complete) {
        fields->emplace_back(int_field(measurement.value));
        fields->emplace_back(move_text(measurement.best_move));
        fields->emplace_back(
            measurement.best_move == MOVE_UNDEFINED
                ? ""
                : int_field(measurement.best_move)
        );
    } else {
        fields->insert(fields->end(), {"", "", ""});
    }
    fields->emplace_back(uint_field(measurement.nodes));
    fields->emplace_back(uint_field(measurement.elapsed_ms));
    fields->emplace_back(uint_field(measurement.pv_extensions));
}

SearchMeasurement skipped_measurement(const std::string &status, bool is_end_search) {
    SearchMeasurement result;
    result.status = status;
    result.is_end_search = is_end_search;
    return result;
}

void append_static_depth_measurement(
    std::vector<std::string> *fields,
    const StaticDiagnostics &diagnostics,
    bool cache_hit
) {
    SearchMeasurement measurement;
    measurement.status = "complete_static";
    measurement.complete = true;
    measurement.value = diagnostics.value;
    append_measurement(fields, measurement, cache_hit);
}

int count_mask(uint64_t discs, uint64_t mask) {
    return pop_count_ull(discs & mask);
}

int opponent_legal_count(const Board &board) {
    return pop_count_ull(calc_legal(board.opponent, board.player));
}

void append_position_features(std::vector<std::string> *fields, const Board &board) {
    fields->emplace_back(board.to_str());
    fields->emplace_back(hex64(board.player));
    fields->emplace_back(hex64(board.opponent));
    fields->emplace_back(int_field(HW2 - board.n_discs()));
    fields->emplace_back(int_field(board_phase(board)));
    fields->emplace_back(int_field(pop_count_ull(board.get_legal())));
    fields->emplace_back(int_field(opponent_legal_count(board)));
    fields->emplace_back(int_field(board.count_player()));
    fields->emplace_back(int_field(board.count_opponent()));
    fields->emplace_back(int_field(count_mask(board.player, CORNER_MASK)));
    fields->emplace_back(int_field(count_mask(board.opponent, CORNER_MASK)));
    fields->emplace_back(int_field(count_mask(board.player, X_MASK)));
    fields->emplace_back(int_field(count_mask(board.opponent, X_MASK)));
    fields->emplace_back(int_field(count_mask(board.player, C_MASK)));
    fields->emplace_back(int_field(count_mask(board.opponent, C_MASK)));
    const bool terminal = board.get_legal() == 0ULL &&
        calc_legal(board.opponent, board.player) == 0ULL;
    fields->emplace_back(bool_field(board.get_legal() == 0ULL && !terminal));
    fields->emplace_back(bool_field(terminal));
}

void append_position_feature_header(std::vector<std::string> *fields, const std::string &prefix) {
    fields->emplace_back(prefix + "normalized_board");
    fields->emplace_back(prefix + "player_bits_hex");
    fields->emplace_back(prefix + "opponent_bits_hex");
    fields->emplace_back(prefix + "n_empties");
    fields->emplace_back(prefix + "phase");
    fields->emplace_back(prefix + "legal_count");
    fields->emplace_back(prefix + "opponent_legal_count");
    fields->emplace_back(prefix + "player_discs");
    fields->emplace_back(prefix + "opponent_discs");
    fields->emplace_back(prefix + "player_corner_count");
    fields->emplace_back(prefix + "opponent_corner_count");
    fields->emplace_back(prefix + "player_x_count");
    fields->emplace_back(prefix + "opponent_x_count");
    fields->emplace_back(prefix + "player_c_count");
    fields->emplace_back(prefix + "opponent_c_count");
    fields->emplace_back(prefix + "is_pass");
    fields->emplace_back(prefix + "is_terminal");
}

std::vector<ChildDiagnostics> make_children(
    const Board &parent,
    bool exact_requested,
    bool exact_children_enabled,
    bool parent_exact_complete
) {
    std::vector<ChildDiagnostics> children;
    uint64_t legal = parent.get_legal();
    if (legal == 0ULL) {
        if (calc_legal(parent.opponent, parent.player) != 0ULL) {
            ChildDiagnostics child;
            child.edge_kind = "pass";
            child.move = MOVE_PASS;
            child.board = parent.copy();
            child.board.pass();
            const CachedStaticDiagnostics static_result = get_static_diagnostics(child.board);
            child.static_status = static_result.diagnostics.status;
            child.static_available = static_result.diagnostics.available;
            if (child.static_available) {
                child.static_value_parent = -static_result.diagnostics.value;
            }
            child.static_cache_hit = static_result.cache_hit;
            if (!exact_requested) {
                child.exact_status = "skipped_parent_n_empties_above_limit";
            } else if (!exact_children_enabled) {
                child.exact_status = "skipped_exact_children_disabled";
            } else if (!parent_exact_complete) {
                child.exact_status = "skipped_parent_exact_incomplete";
            } else {
                const int depth = HW2 - child.board.n_discs();
                const CachedSearchMeasurement exact = get_search_measurement(child.board, depth);
                child.exact_status = exact.measurement.status;
                child.exact_complete = exact.measurement.complete;
                child.exact_cache_hit = exact.cache_hit;
                child.exact_nodes = exact.measurement.nodes;
                child.exact_elapsed_ms = exact.measurement.elapsed_ms;
                if (child.exact_complete) {
                    child.exact_value_parent = -exact.measurement.value;
                }
            }
            children.emplace_back(std::move(child));
        }
        return children;
    }

    for (uint_fast8_t move = first_bit(&legal); legal; move = next_bit(&legal)) {
        Flip flip;
        calc_flip(&flip, const_cast<Board *>(&parent), move);
        ChildDiagnostics child;
        child.edge_kind = "legal";
        child.move = move;
        child.flip_count = pop_count_ull(flip.flip);
        child.board = parent.copy();
        child.board.move_board(&flip);
        const CachedStaticDiagnostics static_result = get_static_diagnostics(child.board);
        child.static_status = static_result.diagnostics.status;
        child.static_available = static_result.diagnostics.available;
        if (child.static_available) {
            child.static_value_parent = -static_result.diagnostics.value;
        }
        child.static_cache_hit = static_result.cache_hit;

        if (!exact_requested) {
            child.exact_status = "skipped_parent_n_empties_above_limit";
        } else if (!exact_children_enabled) {
            child.exact_status = "skipped_exact_children_disabled";
        } else if (!parent_exact_complete) {
            child.exact_status = "skipped_parent_exact_incomplete";
        } else {
            const int child_depth = HW2 - child.board.n_discs();
            const CachedSearchMeasurement exact = get_search_measurement(child.board, child_depth);
            child.exact_status = exact.measurement.status;
            child.exact_complete = exact.measurement.complete;
            child.exact_cache_hit = exact.cache_hit;
            child.exact_nodes = exact.measurement.nodes;
            child.exact_elapsed_ms = exact.measurement.elapsed_ms;
            if (child.exact_complete) {
                child.exact_value_parent = -exact.measurement.value;
            }
        }
        children.emplace_back(std::move(child));
    }
    return children;
}

void rank_children(std::vector<ChildDiagnostics> *children) {
    if (children->empty()) {
        return;
    }
    int best_static = std::numeric_limits<int>::min();
    for (const ChildDiagnostics &child : *children) {
        if (child.static_available) {
            best_static = std::max(best_static, child.static_value_parent);
        }
    }
    for (ChildDiagnostics &child : *children) {
        if (!child.static_available) {
            continue;
        }
        child.static_rank = 1;
        for (const ChildDiagnostics &other : *children) {
            if (other.static_available &&
                other.static_value_parent > child.static_value_parent) {
                ++child.static_rank;
            }
        }
        child.static_loss = best_static - child.static_value_parent;
        child.static_best = child.static_value_parent == best_static;
    }

    const bool all_exact = std::all_of(
        children->begin(),
        children->end(),
        [](const ChildDiagnostics &child) { return child.exact_complete; }
    );
    if (!all_exact) {
        return;
    }
    int best_exact = std::numeric_limits<int>::min();
    for (const ChildDiagnostics &child : *children) {
        best_exact = std::max(best_exact, child.exact_value_parent);
    }
    for (ChildDiagnostics &child : *children) {
        child.exact_rank = 1;
        for (const ChildDiagnostics &other : *children) {
            if (other.exact_value_parent > child.exact_value_parent) {
                ++child.exact_rank;
            }
        }
        child.exact_loss = best_exact - child.exact_value_parent;
        child.exact_best = child.exact_value_parent == best_exact;
    }
}

PvTrace trace_fixed_depth_pv(
    const Board &root,
    int depth,
    const CachedSearchMeasurement *root_measurement
) {
    const SearchKey trace_key{position_key(root), depth};
    const auto cached_trace = fixed_pv_cache.find(trace_key);
    if (cached_trace != fixed_pv_cache.end()) {
        return cached_trace->second;
    }
    PvTrace trace;
    trace.leaf = root.copy();
    int remaining_depth = depth;
    int view_sign = 1;
    CachedSearchMeasurement current;

    if (depth == 0) {
        trace.status = "complete_static_leaf";
    } else {
        if (root_measurement == nullptr || !root_measurement->measurement.complete) {
            trace.status = root_measurement == nullptr
                ? "incomplete_missing_root_measurement"
                : "incomplete_root_search";
            return trace;
        }
        current = *root_measurement;
        trace.searches = 1;
        trace.cache_hits = current.cache_hit ? 1 : 0;
        if (current.cache_hit) {
            // A cached root no longer guarantees that the global TT contains
            // only this fixed-depth tree.  Clear once, then keep the table warm
            // along the reconstructed PV.  The root itself is not re-run.
            clear_search_tables();
        }
    }

    // This is reconstructed with a full-window search at each selected child,
    // retaining the cold root search's TT.  If several moves tie, the selected
    // tied continuation can differ from the root search's retained TT path, but
    // every recorded move is selected under the same deterministic evaluator,
    // fixed-depth, MPC, LMR, PV-extension, and threading controls.
    while (remaining_depth > 0) {
        const uint64_t legal = trace.leaf.get_legal();
        const uint64_t opponent_legal = calc_legal(trace.leaf.opponent, trace.leaf.player);
        if (legal == 0ULL && opponent_legal == 0ULL) {
            trace.status = "complete_terminal_before_requested_depth";
            break;
        }
        if (!current.measurement.complete) {
            trace.status = "incomplete_continuation_search";
            return trace;
        }

        if (legal == 0ULL) {
            trace.moves.emplace_back(MOVE_PASS);
            ++trace.pass_count;
            trace.leaf.pass();
            view_sign = -view_sign;
            current = get_pv_search_measurement(trace.leaf, remaining_depth);
            ++trace.searches;
            trace.cache_hits += current.cache_hit ? 1 : 0;
            continue;
        }

        const int move = current.measurement.best_move;
        if (move < 0 || HW2 <= move || (legal & (1ULL << move)) == 0ULL) {
            trace.status = "incomplete_invalid_best_move";
            return trace;
        }
        Flip flip;
        calc_flip(&flip, &trace.leaf, move);
        trace.moves.emplace_back(move);
        trace.leaf.move_board(&flip);
        view_sign = -view_sign;
        ++trace.legal_move_count;
        --remaining_depth;

        if (remaining_depth > 0) {
            const uint64_t child_legal = trace.leaf.get_legal();
            const uint64_t child_opponent_legal =
                calc_legal(trace.leaf.opponent, trace.leaf.player);
            if (child_legal == 0ULL && child_opponent_legal == 0ULL) {
                trace.status = "complete_terminal_before_requested_depth";
                break;
            }
            current = get_pv_search_measurement(trace.leaf, remaining_depth);
            ++trace.searches;
            trace.cache_hits += current.cache_hit ? 1 : 0;
        }
    }

    if (trace.status.empty()) {
        trace.status = "complete";
    }
    const CachedStaticDiagnostics leaf_static = get_static_diagnostics(trace.leaf);
    trace.leaf_view_sign = view_sign;
    trace.leaf_static_status = leaf_static.diagnostics.status;
    trace.leaf_static_available = leaf_static.diagnostics.available;
    if (trace.leaf_static_available) {
        trace.leaf_static_value = leaf_static.diagnostics.value;
        trace.leaf_static_value_root = view_sign * trace.leaf_static_value;
    }
    trace.leaf_static_cache_hit = leaf_static.cache_hit;
    trace.leaf_terminal = trace.leaf.get_legal() == 0ULL &&
        calc_legal(trace.leaf.opponent, trace.leaf.player) == 0ULL;
    trace.complete = true;
    fixed_pv_cache.emplace(trace_key, trace);
    return trace;
}

std::vector<std::string> make_root_header(const InputSchema &schema) {
    std::vector<std::string> header;
    append_input_header(&header, schema);
    append_position_feature_header(&header, "root_");
    header.insert(header.end(), {
        "static_value",
        "forced_phase_minus1", "forced_raw_minus1", "forced_value_minus1",
        "forced_phase_actual", "forced_raw_actual", "forced_value_actual",
        "forced_phase_plus1", "forced_raw_plus1", "forced_value_plus1",
        "forced_actual_matches_static",
        "search_threads", "book_enabled", "mpc_enabled", "lmr_enabled",
        "pv_extension_enabled", "root_full_window", "hash_level", "hash_source",
        "hash_fallback_seed", "max_exact_empties", "exact_children_enabled"
    });
    for (const int depth : FIXED_DEPTHS) {
        const std::string prefix = "depth_" + std::to_string(depth);
        append_measurement_header(&header, prefix);
        append_pv_header(&header, prefix);
    }
    header.emplace_back("exact_depth");
    append_measurement_header(&header, "exact");
    return header;
}

std::vector<std::string> make_child_header(const InputSchema &schema) {
    std::vector<std::string> header;
    append_input_header(&header, schema);
    header.insert(header.end(), {
        "edge_kind", "move", "move_idx", "flip_count"
    });
    append_position_feature_header(&header, "child_");
    header.insert(header.end(), {
        "static_status", "static_available", "static_value_parent",
        "static_cache_hit", "static_rank", "static_loss", "static_best",
        "exact_status", "exact_complete", "exact_cache_hit",
        "exact_value_parent", "exact_nodes", "exact_elapsed_ms", "exact_rank",
        "exact_loss", "exact_best", "selected_by_exact"
    });
    for (const int depth : FIXED_DEPTHS) {
        header.emplace_back("selected_by_depth_" + std::to_string(depth));
    }
    return header;
}

void validate_board_for_evaluation(const Board &board, size_t input_line) {
    const int phase = board_phase(board);
    if (board.player & board.opponent) {
        throw std::runtime_error("overlapping bitboards at input line " + std::to_string(input_line));
    }
    if (board.n_discs() < 4 || phase < 0 || N_PHASES <= phase) {
        throw std::runtime_error(
            "board is outside the 4..63-disc static-evaluation phase range at input line " +
            std::to_string(input_line)
        );
    }
}

void validate_board_text(const std::string &board_text, size_t input_line) {
    std::string compact;
    compact.reserve(board_text.size());
    for (const unsigned char c : board_text) {
        if (!std::isspace(c)) {
            compact += static_cast<char>(c);
        }
    }
    if (compact.size() != HW2 + 1) {
        throw std::runtime_error(
            "board text must contain 64 cells and a side-to-move marker at input line " +
            std::to_string(input_line)
        );
    }
    for (int i = 0; i < HW2; ++i) {
        if (!is_black_like_char(compact[i]) &&
            !is_white_like_char(compact[i]) &&
            !is_vacant_like_char(compact[i])) {
            throw std::runtime_error(
                "invalid board cell character at input line " +
                std::to_string(input_line) + ", cell " + std::to_string(i)
            );
        }
    }
    if (!is_black_like_char(compact[HW2]) &&
        !is_white_like_char(compact[HW2])) {
        throw std::runtime_error(
            "invalid side-to-move marker at input line " + std::to_string(input_line)
        );
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 5 || argc > 9) {
        std::cerr
            << "usage: " << argv[0]
            << " <input.tsv> <root.tsv> <children.tsv> <max-exact-empties>"
               " [hash-level=20] [position-limit=0] [engine-root=./]"
               " [exact-children=1]\n";
        return 2;
    }

    try {
        const std::string input_path = argv[1];
        const std::string root_output_path = argv[2];
        const std::string child_output_path = argv[3];
        const int max_exact_empties = parse_int(argv[4], "max-exact-empties");
        const int hash_level = argc >= 6 ? parse_int(argv[5], "hash-level") : 20;
        const int position_limit = argc >= 7 ? parse_int(argv[6], "position-limit") : 0;
        const std::string engine_root = normalize_engine_root(argc >= 8 ? argv[7] : "./");
        const int exact_children_arg = argc >= 9 ? parse_int(argv[8], "exact-children") : 1;
        const bool exact_children_enabled = exact_children_arg != 0;
        const std::filesystem::path hash_path = std::filesystem::path(engine_root) /
            "resources" / "hash" / ("hash" + std::to_string(hash_level) + ".eghs");
        std::error_code hash_size_error;
        const uintmax_t expected_hash_bytes = 8ULL * 65536ULL * sizeof(uint32_t);
        const bool valid_hash_file = std::filesystem::is_regular_file(hash_path) &&
            std::filesystem::file_size(hash_path, hash_size_error) == expected_hash_bytes &&
            !hash_size_error;

        if (max_exact_empties < 0 || HW2 - 4 < max_exact_empties) {
            throw std::runtime_error("max-exact-empties must be in [0, 60]");
        }
        if (hash_level < 0 || N_HASH_LEVEL <= hash_level) {
            throw std::runtime_error(
                "hash-level must be in [0, " + std::to_string(N_HASH_LEVEL - 1) + "]"
            );
        }
        if (position_limit < 0) {
            throw std::runtime_error("position-limit must be non-negative");
        }
        if (exact_children_arg != 0 && exact_children_arg != 1) {
            throw std::runtime_error("exact-children must be 0 or 1");
        }
        if (input_path == root_output_path || input_path == child_output_path ||
            root_output_path == child_output_path) {
            throw std::runtime_error("input, root output, and child output paths must differ");
        }

        std::ifstream input(input_path);
        if (!input) {
            throw std::runtime_error("cannot open input TSV: " + input_path);
        }
        const InputSchema schema = read_schema(&input);

        std::ofstream root_output(root_output_path, std::ios::out | std::ios::trunc);
        std::ofstream child_output(child_output_path, std::ios::out | std::ios::trunc);
        if (!root_output) {
            throw std::runtime_error("cannot open root output TSV: " + root_output_path);
        }
        if (!child_output) {
            throw std::runtime_error("cannot open child output TSV: " + child_output_path);
        }

        if (!initialize_engine(hash_level, engine_root)) {
            throw std::runtime_error("engine initialization failed for root: " + engine_root);
        }

        write_tsv_row(&root_output, make_root_header(schema));
        write_tsv_row(&child_output, make_child_header(schema));

        size_t line_number = 1;
        int n_positions = 0;
        InputRow input_row;
        while (read_input_row(&input, schema, &line_number, &input_row)) {
            if (position_limit > 0 && n_positions >= position_limit) {
                break;
            }
            ++n_positions;

            validate_board_text(input_row.board_text, input_row.input_line);
            Board board;
            if (!board.from_str(input_row.board_text)) {
                throw std::runtime_error(
                    "invalid board at input line " + std::to_string(input_row.input_line)
                );
            }
            validate_board_for_evaluation(board, input_row.input_line);
            const int n_empties = HW2 - board.n_discs();

            const CachedStaticDiagnostics static_result = get_static_diagnostics(board);
            if (!static_result.diagnostics.actual_matches_runtime) {
                throw std::runtime_error(
                    "forced actual phase disagrees with runtime static evaluation at input line " +
                    std::to_string(input_row.input_line)
                );
            }

            std::unordered_map<int, CachedSearchMeasurement> fixed_results;
            std::unordered_map<int, PvTrace> fixed_pv_traces;
            for (const int depth : FIXED_DEPTHS) {
                if (depth == 0) {
                    fixed_pv_traces.emplace(
                        depth,
                        trace_fixed_depth_pv(board, depth, nullptr)
                    );
                    continue;
                }
                if (n_empties < depth) {
                    fixed_results.emplace(
                        depth,
                        CachedSearchMeasurement{
                            skipped_measurement("skipped_depth_exceeds_n_empties", false),
                            false
                        }
                    );
                } else {
                    fixed_results.emplace(depth, get_search_measurement(board, depth));
                }
                // Trace now, before another cold root search clears this
                // depth's TT.  Duplicate roots reuse fixed_pv_cache.
                fixed_pv_traces.emplace(
                    depth,
                    trace_fixed_depth_pv(board, depth, &fixed_results.at(depth))
                );
            }

            const bool exact_requested = n_empties <= max_exact_empties;
            CachedSearchMeasurement exact_result;
            if (exact_requested) {
                exact_result = get_search_measurement(board, n_empties);
            } else {
                exact_result.measurement = skipped_measurement(
                    "skipped_n_empties_above_limit",
                    true
                );
            }


            std::vector<std::string> root_fields;
            append_input_fields(&root_fields, input_row);
            append_position_features(&root_fields, board);
            root_fields.emplace_back(int_field(static_result.diagnostics.value));
            append_forced_evaluation(&root_fields, static_result.diagnostics.previous);
            append_forced_evaluation(&root_fields, static_result.diagnostics.actual);
            append_forced_evaluation(&root_fields, static_result.diagnostics.next);
            root_fields.emplace_back(bool_field(static_result.diagnostics.actual_matches_runtime));
            root_fields.insert(root_fields.end(), {
                "1", "0", "0", "0", "0", "1", int_field(hash_level),
                valid_hash_file ? "file" : "deterministic_fallback",
                "0x00E6A202", int_field(max_exact_empties),
                bool_field(exact_children_enabled)
            });
            for (const int depth : FIXED_DEPTHS) {
                if (depth == 0) {
                    append_static_depth_measurement(
                        &root_fields,
                        static_result.diagnostics,
                        static_result.cache_hit
                    );
                } else {
                    const CachedSearchMeasurement &result = fixed_results.at(depth);
                    append_measurement(&root_fields, result.measurement, result.cache_hit);
                }
                append_pv(&root_fields, fixed_pv_traces.at(depth));
            }
            root_fields.emplace_back(int_field(n_empties));
            append_measurement(
                &root_fields,
                exact_result.measurement,
                exact_result.cache_hit
            );
            write_tsv_row(&root_output, root_fields);

            std::vector<ChildDiagnostics> children = make_children(
                board,
                exact_requested,
                exact_children_enabled,
                exact_result.measurement.complete
            );
            rank_children(&children);
            for (const ChildDiagnostics &child : children) {
                std::vector<std::string> child_fields;
                append_input_fields(&child_fields, input_row);
                child_fields.emplace_back(child.edge_kind);
                child_fields.emplace_back(move_text(child.move));
                child_fields.emplace_back(int_field(child.move));
                child_fields.emplace_back(int_field(child.flip_count));
                append_position_features(&child_fields, child.board);
                child_fields.emplace_back(child.static_status);
                child_fields.emplace_back(bool_field(child.static_available));
                child_fields.emplace_back(
                    child.static_available ? int_field(child.static_value_parent) : ""
                );
                child_fields.emplace_back(bool_field(child.static_cache_hit));
                child_fields.emplace_back(
                    child.static_available ? int_field(child.static_rank) : ""
                );
                child_fields.emplace_back(
                    child.static_available ? int_field(child.static_loss) : ""
                );
                child_fields.emplace_back(
                    child.static_available ? bool_field(child.static_best) : ""
                );
                child_fields.emplace_back(child.exact_status);
                child_fields.emplace_back(bool_field(child.exact_complete));
                child_fields.emplace_back(bool_field(child.exact_cache_hit));
                if (child.exact_complete) {
                    child_fields.emplace_back(int_field(child.exact_value_parent));
                } else {
                    child_fields.emplace_back("");
                }
                child_fields.emplace_back(
                    child.exact_complete ? uint_field(child.exact_nodes) : ""
                );
                child_fields.emplace_back(
                    child.exact_complete ? uint_field(child.exact_elapsed_ms) : ""
                );
                child_fields.emplace_back(child.exact_complete ? int_field(child.exact_rank) : "");
                child_fields.emplace_back(child.exact_complete ? int_field(child.exact_loss) : "");
                child_fields.emplace_back(child.exact_complete ? bool_field(child.exact_best) : "");
                child_fields.emplace_back(
                    exact_result.measurement.complete
                        ? bool_field(exact_result.measurement.best_move == child.move)
                        : ""
                );
                for (const int depth : FIXED_DEPTHS) {
                    if (depth == 0) {
                        child_fields.emplace_back(
                            child.static_available ? bool_field(child.static_best) : ""
                        );
                    } else {
                        const SearchMeasurement &measurement = fixed_results.at(depth).measurement;
                        child_fields.emplace_back(
                            measurement.complete
                                ? bool_field(measurement.best_move == child.move)
                                : ""
                        );
                    }
                }
                write_tsv_row(&child_output, child_fields);
            }

            if ((n_positions % 10) == 0) {
                std::cerr
                    << "processed " << n_positions
                    << " positions; static cache=" << static_cache.size()
                    << ", search cache=" << search_cache.size() << '\n';
            }
        }

        root_output.flush();
        child_output.flush();
        if (!root_output || !child_output) {
            throw std::runtime_error("failed while writing output TSV");
        }
        std::cerr
            << "done: positions=" << n_positions
            << ", unique static positions=" << static_cache.size()
            << ", unique searches=" << search_cache.size() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "eval_training_bias_tool: " << error.what() << '\n';
        return 1;
    }
}
