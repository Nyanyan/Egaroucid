/*
    Egaroucid Project

    Optional .egev2 evaluation for move ordering and shallow auxiliary
    ordering/MPC searches. Features are shared with the main evaluation.
    Auxiliary results may use the ordinary TT without a model tag, or skip
    TT bounds for comparison; normal main-search leaves retain their model.
*/

#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>
#include "move_ordering_policy.hpp"

inline int eval_move_ordering_mid_shallow_max_depth = 0;
inline bool eval_move_ordering_mid_shallow_share_tt = false;
inline bool eval_move_ordering_mid_shallow_end_only = false;

#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
inline bool eval_move_ordering_mid_enabled = false;
inline std::vector<int16_t> eval_move_ordering_mid_patterns;
inline int16_t eval_move_ordering_mid_num[N_PHASES][MAX_STONE_NUM];
inline bool eval_move_ordering_aux_enabled = false;
inline std::vector<int16_t> eval_move_ordering_aux_patterns;
inline int16_t eval_move_ordering_aux_num[N_PHASES][MAX_STONE_NUM];
#if !USE_SIMD_EVALUATION
inline const int16_t *eval_move_ordering_mid_tables[2][N_PHASES][N_PATTERNS];
inline const int16_t *eval_move_ordering_aux_tables[2][N_PHASES][N_PATTERNS];
#endif

// Validate the compressed and decoded lengths before publishing any table.
// In particular, .egev2 has no feature-definition metadata of its own.
inline bool load_eval_move_ordering_model_file(const char *file, bool show_log, bool auxiliary) {
    auto &model_patterns = auxiliary ? eval_move_ordering_aux_patterns : eval_move_ordering_mid_patterns;
    auto &model_num = auxiliary ? eval_move_ordering_aux_num : eval_move_ordering_mid_num;
#if !USE_SIMD_EVALUATION
    auto &model_tables = auxiliary ? eval_move_ordering_aux_tables : eval_move_ordering_mid_tables;
#endif
    if (show_log) {
        std::cerr << (auxiliary ? "auxiliary" : "static") << " move ordering evaluation file " << file << std::endl;
    }
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "[ERROR] can't open move ordering evaluation " << file << std::endl;
        return false;
    }
    const std::streamoff file_size = input.tellg();
    input.seekg(0);
    int32_t compressed_size = 0;
    input.read(reinterpret_cast<char *>(&compressed_size), sizeof(compressed_size));
    const size_t expected_size = size_t(N_PHASES) * (N_PATTERN_PARAMS_RAW + MAX_STONE_NUM);
    if (!input || compressed_size <= 0 || size_t(compressed_size) > expected_size ||
        file_size != std::streamoff(sizeof(compressed_size)) + std::streamoff(compressed_size) * 2) {
        std::cerr << "[ERROR] invalid move ordering evaluation length" << std::endl;
        return false;
    }
    std::vector<int16_t> compressed(compressed_size);
    input.read(reinterpret_cast<char *>(compressed.data()), compressed.size() * sizeof(int16_t));
    if (!input) {
        return false;
    }
    std::vector<int16_t> parameters;
    parameters.reserve(expected_size);
    for (int16_t value : compressed) {
        const size_t count = value >= N_ZEROS_PLUS ? value - N_ZEROS_PLUS : 1;
        if (count == 0 || count > expected_size - parameters.size()) {
            std::cerr << "[ERROR] invalid move ordering evaluation parameter count" << std::endl;
            return false;
        }
        parameters.insert(parameters.end(), count, value >= N_ZEROS_PLUS ? 0 : value);
    }
    if (parameters.size() != expected_size) {
        std::cerr << "[ERROR] move ordering evaluation has " << parameters.size()
                  << " parameters; expected " << expected_size << std::endl;
        return false;
    }
    size_t source_offset = 0;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        for (int i = 0; i < N_PATTERN_PARAMS_RAW; ++i) {
            if (parameters[source_offset + i] < -4092 || parameters[source_offset + i] > 4092) {
                std::cerr << "[ERROR] move ordering pattern value outside [-4092, 4092]" << std::endl;
                return false;
            }
        }
        source_offset += N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
    }
#if USE_SIMD_EVALUATION
    std::vector<int16_t> patterns(size_t(N_PHASES) * N_PATTERN_PARAMS, 0);
#else
    // Compact tables for both player orientations, without padding patterns
    // to MAX_EVALUATE_IDX. Main evaluation and FM Dim0 tables are untouched.
    std::vector<int16_t> patterns(size_t(2) * N_PHASES * N_PATTERN_PARAMS_RAW);
#endif
    source_offset = 0;
    for (int phase = 0; phase < N_PHASES; ++phase) {
#if USE_SIMD_EVALUATION
        int16_t *destination = patterns.data() + size_t(phase) * N_PATTERN_PARAMS;
        for (int i = 0; i < N_PATTERN_PARAMS_RAW; ++i) {
            destination[i + 1] = parameters[source_offset + i] + SIMD_EVAL_MAX_VALUE;
        }
#else
        size_t pattern_offset = 0;
        for (int pattern = 0; pattern < N_PATTERNS; ++pattern) {
            const int count = pow3[pattern_sizes[pattern]];
            int16_t *normal = patterns.data() + size_t(phase) * N_PATTERN_PARAMS_RAW + pattern_offset;
            int16_t *reversed = normal + size_t(N_PHASES) * N_PATTERN_PARAMS_RAW;
            for (int i = 0; i < count; ++i) {
                normal[i] = parameters[source_offset + pattern_offset + i];
                reversed[swap_player_idx(i, pattern_sizes[pattern])] = normal[i];
            }
            pattern_offset += count;
        }
#endif
        source_offset += N_PATTERN_PARAMS_RAW;
        std::copy_n(parameters.data() + source_offset, MAX_STONE_NUM, model_num[phase]);
        source_offset += MAX_STONE_NUM;
    }
    model_patterns.swap(patterns);
#if !USE_SIMD_EVALUATION
    for (int orientation = 0; orientation < 2; ++orientation) {
        for (int phase = 0; phase < N_PHASES; ++phase) {
            size_t offset = size_t(orientation * N_PHASES + phase) * N_PATTERN_PARAMS_RAW;
            for (int pattern = 0; pattern < N_PATTERNS; ++pattern) {
                model_tables[orientation][phase][pattern] = model_patterns.data() + offset;
                offset += pow3[pattern_sizes[pattern]];
            }
        }
    }
#endif
    (auxiliary ? eval_move_ordering_aux_enabled : eval_move_ordering_mid_enabled) = true;
    return true;
}

inline bool load_eval_move_ordering_mid_file(const char *file, bool show_log) {
    // Explicit model selection keeps the experimental/manual CLI semantics.
    eval_move_ordering_auto_enabled = false;
    return load_eval_move_ordering_model_file(file, show_log, false);
}

inline bool load_eval_move_ordering_aux_file(const char *file, bool show_log) {
    return load_eval_move_ordering_model_file(file, show_log, true);
}

inline int mid_evaluate_separate_move_ordering(Search *search, bool auxiliary = false) {
    const bool use_auxiliary = auxiliary && eval_move_ordering_aux_enabled;
    const auto &model_num = use_auxiliary ? eval_move_ordering_aux_num : eval_move_ordering_mid_num;
    const int phase = search->phase();
#if USE_SIMD_EVALUATION
    const auto &model_patterns = use_auxiliary ? eval_move_ordering_aux_patterns : eval_move_ordering_mid_patterns;
    const int16_t *table = model_patterns.data() + size_t(phase) * N_PATTERN_PARAMS;
    const int *start0 = reinterpret_cast<const int *>(table);
    const int *start4 = reinterpret_cast<const int *>(table + PATTERN4_START_IDX);
    const int *start6 = reinterpret_cast<const int *>(table + PATTERN6_START_IDX);
    const Eval_features *features = &search->eval.features[search->eval.feature_idx];
    __m256i sum = gather_eval(start0, _mm256_cvtepu16_epi32(features->f128[0]));
    sum = _mm256_add_epi32(sum, gather_eval(start0, _mm256_cvtepu16_epi32(features->f128[1])));
    sum = _mm256_add_epi32(sum, gather_eval(start6, _mm256_cvtepu16_epi32(features->f128[2])));
    sum = _mm256_add_epi32(sum, gather_eval(start4, _mm256_cvtepu16_epi32(features->f128[3])));
    sum = _mm256_add_epi32(sum, gather_eval(start0, calc_idx8_comp(features->f128[4], 0)));
    sum = _mm256_add_epi32(sum, gather_eval(start0, calc_idx8_comp(features->f128[5], 1)));
    sum = _mm256_add_epi32(sum, gather_eval(start0, calc_idx8_comp(features->f128[6], 2)));
    sum = _mm256_add_epi32(sum, gather_eval(start0, calc_idx8_comp(features->f128[7], 3)));
    sum = _mm256_and_si256(sum, eval_lower_mask);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int result = _mm_cvtsi128_si32(sum128) + _mm_extract_epi32(sum128, 1) - SIMD_EVAL_MAX_VALUE * N_PATTERN_FEATURES;
#else
    const int orientation = search->eval.reversed[search->eval.feature_idx];
    const auto &model_tables = use_auxiliary ? eval_move_ordering_aux_tables : eval_move_ordering_mid_tables;
    int result = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        result += model_tables[orientation][phase][feature_to_pattern[i]][search->eval.features[search->eval.feature_idx][i]];
    }
#endif
    result += model_num[phase][pop_count_ull(search->board.player)];
    result += result >= 0 ? STEP_2 : -STEP_2;
    return std::clamp(result / STEP, -SCORE_MAX, SCORE_MAX);
}
#else
inline constexpr bool eval_move_ordering_mid_enabled = false;
inline constexpr bool eval_move_ordering_aux_enabled = false;
inline bool load_eval_move_ordering_mid_file(const char *, bool) {
    std::cerr << "[ERROR] separate .egev2 move ordering evaluation is unavailable in this build" << std::endl;
    return false;
}
inline bool load_eval_move_ordering_aux_file(const char *, bool) {
    std::cerr << "[ERROR] separate auxiliary evaluation is unavailable in this build" << std::endl;
    return false;
}
#endif

inline bool initialize_auto_move_ordering_models(const char *main_eval_file, bool show_log) {
    eval_move_ordering_auto_enabled = false;
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    eval_move_ordering_mid_enabled = false;
    eval_move_ordering_aux_enabled = false;
    if (!eval_move_ordering_auto_requested) return true;
    const std::string main_path(main_eval_file);
    const auto separator = main_path.find_last_of("/\\");
    const std::string directory = separator == std::string::npos ? "" : main_path.substr(0, separator + 1);
    const std::string offset_file = directory + "offset0_16.egev2";
    const std::string decay_file = directory + "decay_half16.egev2";
    if (!load_eval_move_ordering_model_file(offset_file.c_str(), show_log, false) ||
        !load_eval_move_ordering_model_file(decay_file.c_str(), show_log, true)) {
        eval_move_ordering_mid_enabled = false;
        eval_move_ordering_aux_enabled = false;
        std::cerr << "[ERROR] [FATAL] automatic move ordering models not loaded" << std::endl;
        return false;
    }
    eval_move_ordering_auto_enabled = true;
    if (show_log) std::cerr << "automatic move ordering policy initialized (midgame >=27, endgame >74%)" << std::endl;
#else
    (void)main_eval_file;
    (void)show_log;
#endif
    return true;
}

inline Move_ordering_model search_static_ordering_model(const Search *search) {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    if (search->move_ordering_policy.automatic) return search->move_ordering_policy.static_model;
    if (!eval_move_ordering_auto_enabled && eval_move_ordering_mid_enabled) return Move_ordering_model::primary;
#endif
    return Move_ordering_model::ordinary;
}

inline bool search_uses_separate_ordering_evaluation(const Search *search) {
    return search_static_ordering_model(search) != Move_ordering_model::ordinary;
}

inline int mid_evaluate_for_ordering(Search *search) {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    const auto model = search_static_ordering_model(search);
    if (model != Move_ordering_model::ordinary) {
        return mid_evaluate_separate_move_ordering(search, model == Move_ordering_model::secondary);
    }
#endif
#if USE_DIM0_ONLY_EVALUATION
    return mid_evaluate_diff(search);
#else
    return mid_evaluate_move_ordering_dim0(search);
#endif
}

// Only callers performing an auxiliary ordering/MPC search enter this scope.
// Depth is tested once at its entry; reaching depth <= 5 in the main search
// never enables it. Passes and nested MPC probes retain the selected model.
class Move_ordering_eval_scope {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    Search &search;
    const bool saved_evaluation;
    const bool saved_tt;
    const bool saved_parallel;
    const Move_ordering_model saved_model;
#endif
public:
    Move_ordering_eval_scope(Search &s, int depth, bool is_end_search = false)
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
        : search(s), saved_evaluation(s.use_separate_shallow_evaluation),
          saved_tt(s.share_separate_shallow_eval_tt), saved_parallel(s.use_multi_thread),
          saved_model(s.separate_shallow_model)
#endif
    {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
        const auto &policy = search.move_ordering_policy;
        const int max_depth = policy.automatic ? policy.auxiliary_max_depth : eval_move_ordering_mid_shallow_max_depth;
        const bool end_only = policy.automatic || eval_move_ordering_mid_shallow_end_only;
        const bool permitted_context = !end_only || is_end_search || saved_evaluation;
        const auto model = policy.automatic ? policy.auxiliary_model
            : (eval_move_ordering_aux_enabled ? Move_ordering_model::secondary : Move_ordering_model::primary);
        if (search_uses_separate_ordering_evaluation(&search) && model != Move_ordering_model::ordinary &&
            permitted_context && depth > 0 && depth <= max_depth) {
            search.use_separate_shallow_evaluation = true;
            search.separate_shallow_model = model;
            search.share_separate_shallow_eval_tt = policy.automatic ? policy.share_tt : eval_move_ordering_mid_shallow_share_tt;
            search.use_multi_thread = false;
        }
#else
        (void)s;
        (void)depth;
        (void)is_end_search;
#endif
    }
    Move_ordering_eval_scope(const Move_ordering_eval_scope &) = delete;
    Move_ordering_eval_scope &operator=(const Move_ordering_eval_scope &) = delete;
    ~Move_ordering_eval_scope() {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
        search.use_separate_shallow_evaluation = saved_evaluation;
        search.share_separate_shallow_eval_tt = saved_tt;
        search.use_multi_thread = saved_parallel;
        search.separate_shallow_model = saved_model;
#endif
    }
};

inline bool search_uses_separate_shallow_evaluation(const Search *search) {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    return search->use_separate_shallow_evaluation;
#else
    (void)search;
    return false;
#endif
}

inline int mid_evaluate_for_search(Search *search) {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    if (search->use_separate_shallow_evaluation) {
        return mid_evaluate_separate_move_ordering(search, search->separate_shallow_model == Move_ordering_model::secondary);
    }
#endif
#if !USE_DIM0_ONLY_EVALUATION
    if (search->use_dim0_mpc_eval) {
        return mid_evaluate_dim0(search);
    }
#endif
    return mid_evaluate_diff(search);
}

inline int move_ordering_leaf_hint(Search *search, int static_value) {
#if USE_SEPARATE_MID_MOVE_ORDERING_EVALUATION && !defined(USE_NNUE_EVALUATION)
    if (search->separate_shallow_model == search_static_ordering_model(search)) return static_value;
#endif
    return mid_evaluate_for_search(search);
}
