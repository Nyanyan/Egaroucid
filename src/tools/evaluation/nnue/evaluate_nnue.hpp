/*
    Egaroucid Project

    @file evaluate_nnue.hpp
        NNUE evaluation function
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#include "../../../engine/setting.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/evaluate_common.hpp"
#include "../evaluation_definition.hpp"

constexpr int EVAL_NNUE_INPUT_FEATURES = 128;
constexpr int EVAL_NNUE_INPUT_KIND_STONE = 1;
constexpr int EVAL_NNUE_INPUT_KIND_PATTERN = 2;
constexpr int EVAL_NNUE_INPUT_KIND_PATTERN_PAIR = 3;
constexpr int EVAL_NNUE_MAX_HIDDEN1 = 64;
constexpr int EVAL_NNUE_MAX_HIDDEN2 = 64;
constexpr int EVAL_NNUE_MAX_POST_INPUT = EVAL_NNUE_MAX_FT_DIM * 2;
constexpr int EVAL_NNUE_SIMD_WIDTH = 32;

inline int eval_nnue_ft_dim = 0;
inline int eval_nnue_hidden1_dim = 0;
inline int eval_nnue_hidden2_dim = 0;
inline int eval_nnue_ft_shift = 0;
inline int eval_nnue_hidden1_shift = 0;
inline int eval_nnue_hidden2_shift = 0;
inline int eval_nnue_output_shift = 0;
inline int eval_nnue_input_kind = EVAL_NNUE_INPUT_KIND_STONE;
inline int eval_nnue_input_features = EVAL_NNUE_INPUT_FEATURES;
inline int eval_nnue_pattern_feature_columns = 0;
inline bool eval_nnue_enabled = false;

struct EvalNnueCoordFeature {
    uint8_t feature = 0;
    uint16_t weight = 0;
};

struct EvalNnueCoordFeatures {
    uint8_t n_features = 0;
    EvalNnueCoordFeature features[16];
};

inline std::array<EvalNnueCoordFeatures, HW2> eval_nnue_coord_to_feature;
inline std::array<int, ADJ_N_FEATURES> eval_nnue_pattern_feature_starts;
inline bool eval_nnue_pattern_tables_initialized = false;

// These names are referenced by shared search code. In an NNUE-only build they
// intentionally stay false, so Dim0-specific MPC logic is not used.
inline bool eval_fm_enabled = false;
inline bool eval_fm_use_dim0_mpc_search = false;

inline std::vector<int16_t> eval_nnue_ft_bias;
inline std::vector<int16_t> eval_nnue_ft_weight;
inline std::vector<int32_t> eval_nnue_hidden1_bias;
inline std::vector<int8_t> eval_nnue_hidden1_weight;
inline std::vector<int32_t> eval_nnue_hidden2_bias;
inline std::vector<int8_t> eval_nnue_hidden2_weight;
inline std::array<int32_t, N_PHASES> eval_nnue_output_bias;
inline std::vector<int8_t> eval_nnue_output_weight;
inline int eval_nnue_post_input_padded_value = 0;
inline int eval_nnue_hidden1_padded_value = 0;
inline int eval_nnue_hidden2_padded_value = 0;

inline int eval_nnue_round_up(const int n, const int base) {
    return ((n + base - 1) / base) * base;
}

inline int eval_nnue_pattern_total_input_features() {
    int res = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        res += adj_eval_sizes[i];
    }
    return res;
}

inline int eval_nnue_pattern_feature_start(const int feature_idx) {
    int start = 0;
    for (int i = 1; i <= feature_idx; ++i) {
        if (adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            start += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
    }
    return start;
}

inline void eval_nnue_init_pattern_tables() {
    if (eval_nnue_pattern_tables_initialized) {
        return;
    }
    int feature_start = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        if (i > 0 && adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            feature_start += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
        eval_nnue_pattern_feature_starts[(size_t)i] = feature_start;
    }
    for (EvalNnueCoordFeatures &entry: eval_nnue_coord_to_feature) {
        entry.n_features = 0;
    }
    for (int feature = 0; feature < ADJ_N_SYMMETRY_PATTERNS; ++feature) {
        const int n_cells = adj_feature_to_coord[feature].n_cells;
        for (int digit = 0; digit < n_cells; ++digit) {
            const int coord = adj_feature_to_coord[feature].cells[digit];
            if (coord < 0 || coord >= HW2) {
                continue;
            }
            const int bit_cell = HW2_M1 - coord;
            EvalNnueCoordFeatures &entry = eval_nnue_coord_to_feature[(size_t)bit_cell];
            if (entry.n_features >= 16) {
                std::cerr << "[ERROR] [FATAL] too many NNUE pattern features for cell " << bit_cell << std::endl;
                std::exit(1);
            }
            EvalNnueCoordFeature &target = entry.features[entry.n_features++];
            target.feature = (uint8_t)feature;
            target.weight = (uint16_t)adj_pow3[n_cells - 1 - digit];
        }
    }
    eval_nnue_pattern_tables_initialized = true;
}

inline int eval_nnue_post_input_dim() {
    return eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN ? eval_nnue_ft_dim : eval_nnue_ft_dim * 2;
}

inline int eval_nnue_post_input_padded() {
    return eval_nnue_post_input_padded_value != 0 ? eval_nnue_post_input_padded_value : eval_nnue_round_up(eval_nnue_post_input_dim(), EVAL_NNUE_SIMD_WIDTH);
}

inline int eval_nnue_hidden1_padded() {
    return eval_nnue_hidden1_padded_value != 0 ? eval_nnue_hidden1_padded_value : eval_nnue_round_up(eval_nnue_hidden1_dim, EVAL_NNUE_SIMD_WIDTH);
}

inline int eval_nnue_hidden2_padded() {
    return eval_nnue_hidden2_padded_value != 0 ? eval_nnue_hidden2_padded_value : eval_nnue_round_up(eval_nnue_hidden2_dim, EVAL_NNUE_SIMD_WIDTH);
}

template <typename T>
inline bool eval_nnue_read_scalar(std::ifstream &in, T *value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return !in.fail();
}

template <typename T>
inline bool eval_nnue_read_vector(std::ifstream &in, std::vector<T> *values, const size_t n) {
    values->resize(n);
    if (n == 0) {
        return true;
    }
    in.read(reinterpret_cast<char*>(values->data()), sizeof(T) * n);
    return !in.fail();
}

inline bool eval_nnue_load(const char *file, bool show_log) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] [FATAL] can't open NNUE eval " << file << std::endl;
        return false;
    }

    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "EGNNUE1", 7) != 0) {
        std::cerr << "[ERROR] [FATAL] unsupported NNUE eval file " << file << std::endl;
        return false;
    }

    uint32_t version = 0;
    uint32_t input_features = 0;
    uint32_t ft_dim = 0;
    uint32_t hidden1_dim = 0;
    uint32_t hidden2_dim = 0;
    uint32_t n_phases = 0;
    uint32_t ft_shift = 0;
    uint32_t hidden1_shift = 0;
    uint32_t hidden2_shift = 0;
    uint32_t output_shift = 0;
    uint32_t reserved[8] = {};

    if (!eval_nnue_read_scalar(in, &version) ||
        !eval_nnue_read_scalar(in, &input_features) ||
        !eval_nnue_read_scalar(in, &ft_dim) ||
        !eval_nnue_read_scalar(in, &hidden1_dim) ||
        !eval_nnue_read_scalar(in, &hidden2_dim) ||
        !eval_nnue_read_scalar(in, &n_phases) ||
        !eval_nnue_read_scalar(in, &ft_shift) ||
        !eval_nnue_read_scalar(in, &hidden1_shift) ||
        !eval_nnue_read_scalar(in, &hidden2_shift) ||
        !eval_nnue_read_scalar(in, &output_shift)) {
        std::cerr << "[ERROR] [FATAL] broken NNUE eval header " << file << std::endl;
        return false;
    }
    in.read(reinterpret_cast<char*>(reserved), sizeof(reserved));
    const uint32_t input_kind = version >= 2 ? reserved[0] : EVAL_NNUE_INPUT_KIND_STONE;
    const uint32_t pattern_feature_columns = version >= 2 ? reserved[1] : 0;
    const bool stone_header_ok =
        version == 1 &&
        input_kind == EVAL_NNUE_INPUT_KIND_STONE &&
        input_features == EVAL_NNUE_INPUT_FEATURES;
    const bool pattern_header_ok =
        version == 2 &&
        input_kind == EVAL_NNUE_INPUT_KIND_PATTERN &&
        pattern_feature_columns == ADJ_N_FEATURES &&
        input_features == (uint32_t)eval_nnue_pattern_total_input_features();
    const bool pattern_pair_header_ok =
        version == 3 &&
        input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR &&
        pattern_feature_columns == ADJ_N_FEATURES &&
        input_features == (uint32_t)eval_nnue_pattern_total_input_features();
    if (in.fail() ||
        (!stone_header_ok && !pattern_header_ok && !pattern_pair_header_ok) ||
        ft_dim == 0 || ft_dim > EVAL_NNUE_MAX_FT_DIM ||
        hidden1_dim == 0 || hidden1_dim > EVAL_NNUE_MAX_HIDDEN1 ||
        hidden2_dim == 0 || hidden2_dim > EVAL_NNUE_MAX_HIDDEN2 ||
        n_phases != N_PHASES) {
        std::cerr << "[ERROR] [FATAL] incompatible NNUE eval header " << file
                  << " version " << version
                  << " input_kind " << input_kind
                  << " input_features " << input_features
                  << " pattern_feature_columns " << pattern_feature_columns
                  << " ft_dim " << ft_dim
                  << " hidden1_dim " << hidden1_dim
                  << " hidden2_dim " << hidden2_dim
                  << " n_phases " << n_phases << std::endl;
        return false;
    }

    eval_nnue_ft_dim = (int)ft_dim;
    eval_nnue_hidden1_dim = (int)hidden1_dim;
    eval_nnue_hidden2_dim = (int)hidden2_dim;
    eval_nnue_ft_shift = (int)ft_shift;
    eval_nnue_hidden1_shift = (int)hidden1_shift;
    eval_nnue_hidden2_shift = (int)hidden2_shift;
    eval_nnue_output_shift = (int)output_shift;
    eval_nnue_input_kind = (int)input_kind;
    eval_nnue_input_features = (int)input_features;
    eval_nnue_pattern_feature_columns = (int)pattern_feature_columns;
    eval_nnue_post_input_padded_value = eval_nnue_round_up(eval_nnue_post_input_dim(), EVAL_NNUE_SIMD_WIDTH);
    eval_nnue_hidden1_padded_value = eval_nnue_round_up(eval_nnue_hidden1_dim, EVAL_NNUE_SIMD_WIDTH);
    eval_nnue_hidden2_padded_value = eval_nnue_round_up(eval_nnue_hidden2_dim, EVAL_NNUE_SIMD_WIDTH);

    const int post_input_padded = eval_nnue_post_input_padded_value;
    const int hidden1_padded = eval_nnue_hidden1_padded_value;
    const int hidden2_padded = eval_nnue_hidden2_padded_value;

    bool ok = true;
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_ft_bias, eval_nnue_ft_dim);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_ft_weight, (size_t)eval_nnue_input_features * (size_t)eval_nnue_ft_dim);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_hidden1_bias, eval_nnue_hidden1_dim);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_hidden1_weight, eval_nnue_hidden1_dim * post_input_padded);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_hidden2_bias, eval_nnue_hidden2_dim);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_hidden2_weight, eval_nnue_hidden2_dim * hidden1_padded);
    in.read(reinterpret_cast<char*>(eval_nnue_output_bias.data()), sizeof(int32_t) * N_PHASES);
    ok = ok && !in.fail();
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_output_weight, N_PHASES * hidden2_padded);
    if (!ok) {
        std::cerr << "[ERROR] [FATAL] broken NNUE eval payload " << file << std::endl;
        return false;
    }

    eval_nnue_enabled = true;
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN ||
        eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR) {
        eval_nnue_init_pattern_tables();
    }
    if (show_log) {
        std::cerr << "NNUE evaluation loaded " << file
                  << " input_kind " << eval_nnue_input_kind
                  << " input_features " << eval_nnue_input_features
                  << " ft_dim " << eval_nnue_ft_dim
                  << " hidden1 " << eval_nnue_hidden1_dim
                  << " hidden2 " << eval_nnue_hidden2_dim
                  << " shifts " << eval_nnue_ft_shift << " "
                  << eval_nnue_hidden1_shift << " "
                  << eval_nnue_hidden2_shift << " "
                  << eval_nnue_output_shift << std::endl;
    }
    return true;
}

inline bool evaluate_init(const char* file, const char*, bool show_log) {
    if (!eval_nnue_load(file, show_log)) {
        return false;
    }
    if (show_log) {
        std::cerr << "NNUE-only evaluation function initialized" << std::endl;
    }
    return true;
}

inline bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

inline bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval_nnue.egevnnue", "", show_log);
}

inline const int16_t* eval_nnue_feature_weight(const int feature) {
    return &eval_nnue_ft_weight[(size_t)feature * (size_t)eval_nnue_ft_dim];
}

inline int eval_nnue_pattern_global_feature(const int feature_idx, const uint16_t local_value) {
    return eval_nnue_pattern_feature_starts[(size_t)feature_idx] + (int)local_value;
}

inline void eval_nnue_copy_accumulator(int16_t *dst, const int16_t *src) {
    std::memcpy(dst, src, sizeof(int16_t) * (size_t)eval_nnue_ft_dim);
}

inline void eval_nnue_add_feature(int16_t *acc, const int feature) {
    const int16_t *w = eval_nnue_feature_weight(feature);
#if USE_SIMD
    if (eval_nnue_ft_dim == 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 16));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 16), _mm256_add_epi16(a, b));
        return;
    }
    if (eval_nnue_ft_dim == 128) {
        __m256i a;
        __m256i b;
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 16));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 16), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 32));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 32), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 48));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 48));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 48), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 64));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 64), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 80));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 80));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 80), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 96));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 96));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 96), _mm256_add_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 112));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 112));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 112), _mm256_add_epi16(a, b));
        return;
    }
    int i = 0;
    for (; i + 16 <= eval_nnue_ft_dim; i += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, b));
    }
    for (; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + w[i]);
    }
#else
    for (int i = 0; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] + w[i]);
    }
#endif
}

inline void eval_nnue_sub_feature(int16_t *acc, const int feature) {
    const int16_t *w = eval_nnue_feature_weight(feature);
#if USE_SIMD
    if (eval_nnue_ft_dim == 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 16));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 16), _mm256_sub_epi16(a, b));
        return;
    }
    if (eval_nnue_ft_dim == 128) {
        __m256i a;
        __m256i b;
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 16));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 16), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 32));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 32), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 48));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 48));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 48), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 64));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 64), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 80));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 80));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 80), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 96));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 96));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 96), _mm256_sub_epi16(a, b));
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 112));
        b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 112));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 112), _mm256_sub_epi16(a, b));
        return;
    }
    int i = 0;
    for (; i + 16 <= eval_nnue_ft_dim; i += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, b));
    }
    for (; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] - w[i]);
    }
#else
    for (int i = 0; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] - w[i]);
    }
#endif
}

inline void eval_nnue_replace_feature(int16_t *acc, const int old_feature, const int new_feature) {
    if (old_feature == new_feature) {
        return;
    }
    const int16_t *old_w = eval_nnue_feature_weight(old_feature);
    const int16_t *new_w = eval_nnue_feature_weight(new_feature);
#if USE_SIMD
    if (eval_nnue_ft_dim == 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc));
        __m256i oldv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(old_w));
        __m256i newv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(new_w));
        a = _mm256_add_epi16(_mm256_sub_epi16(a, oldv), newv);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc), a);
        a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 16));
        oldv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(old_w + 16));
        newv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(new_w + 16));
        a = _mm256_add_epi16(_mm256_sub_epi16(a, oldv), newv);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 16), a);
        return;
    }
    if (eval_nnue_ft_dim == 128) {
        for (int i = 0; i < 128; i += 16) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
            const __m256i oldv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(old_w + i));
            const __m256i newv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(new_w + i));
            a = _mm256_add_epi16(_mm256_sub_epi16(a, oldv), newv);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), a);
        }
        return;
    }
    int i = 0;
    for (; i + 16 <= eval_nnue_ft_dim; i += 16) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + i));
        const __m256i oldv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(old_w + i));
        const __m256i newv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(new_w + i));
        a = _mm256_add_epi16(_mm256_sub_epi16(a, oldv), newv);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + i), a);
    }
    for (; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] - old_w[i] + new_w[i]);
    }
#else
    for (int i = 0; i < eval_nnue_ft_dim; ++i) {
        acc[i] = (int16_t)(acc[i] - old_w[i] + new_w[i]);
    }
#endif
}

inline void eval_nnue_calc_pattern_accumulator(Board *board, Eval_search *eval, const int idx, const int perspective) {
    int16_t *acc = eval->accumulator[idx][perspective];
    std::memcpy(acc, eval_nnue_ft_bias.data(), sizeof(int16_t) * (size_t)eval_nnue_ft_dim);
    uint16_t features[ADJ_N_FEATURES];
    adj_calc_features(board, features);
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        eval->pattern_features[idx][perspective][i] = features[i];
        const int feature = eval_nnue_pattern_global_feature(i, features[i]);
        eval_nnue_add_feature(acc, feature);
    }
}

inline void eval_nnue_pattern_set_feature(
    Eval_search *eval,
    const int idx,
    const int perspective,
    const int feature_idx,
    const uint16_t local_value
) {
    const uint16_t old_local = eval->pattern_features[idx][perspective][feature_idx];
    if (old_local == local_value) {
        return;
    }
    int16_t *acc = eval->accumulator[idx][perspective];
    const int old_feature = eval_nnue_pattern_global_feature(feature_idx, old_local);
    const int new_feature = eval_nnue_pattern_global_feature(feature_idx, local_value);
    eval->pattern_features[idx][perspective][feature_idx] = local_value;
    eval_nnue_replace_feature(acc, old_feature, new_feature);
}

inline void eval_nnue_pattern_add_delta(
    Eval_search *eval,
    const int idx,
    const int perspective,
    const int feature_idx,
    const int delta
) {
    const int next_value = (int)eval->pattern_features[idx][perspective][feature_idx] + delta;
    eval_nnue_pattern_set_feature(eval, idx, perspective, feature_idx, (uint16_t)next_value);
}

inline void eval_nnue_pattern_apply_cell_delta(
    Eval_search *eval,
    const int idx,
    const int perspective,
    const int cell,
    const int sign
) {
    const EvalNnueCoordFeatures &entry = eval_nnue_coord_to_feature[(size_t)cell];
    for (int i = 0; i < entry.n_features; ++i) {
        const EvalNnueCoordFeature &feature = entry.features[i];
        eval_nnue_pattern_add_delta(eval, idx, perspective, feature.feature, sign * (int)feature.weight);
    }
}

inline void eval_nnue_pattern_touch(
    const int feature_idx,
    uint64_t *touched_lo,
    uint64_t *touched_hi,
    uint8_t *touched_features,
    int *n_touched
) {
    if (feature_idx < 64) {
        const uint64_t bit = 1ULL << feature_idx;
        if ((*touched_lo & bit) != 0) {
            return;
        }
        *touched_lo |= bit;
        touched_features[(*n_touched)++] = (uint8_t)feature_idx;
        return;
    }
    const uint64_t bit = 1ULL << (feature_idx - 64);
    if ((*touched_hi & bit) == 0) {
        *touched_hi |= bit;
        touched_features[(*n_touched)++] = (uint8_t)feature_idx;
    }
}

inline void eval_nnue_pattern_add_delta_local(
    Eval_search *eval,
    const int idx,
    const int perspective,
    const int feature_idx,
    const int delta,
    uint64_t *touched_lo,
    uint64_t *touched_hi,
    uint8_t *touched_features,
    int *n_touched
) {
    eval->pattern_features[idx][perspective][feature_idx] =
        (uint16_t)((int)eval->pattern_features[idx][perspective][feature_idx] + delta);
    eval_nnue_pattern_touch(feature_idx, touched_lo, touched_hi, touched_features, n_touched);
}

inline void eval_nnue_pattern_apply_cell_delta_local(
    Eval_search *eval,
    const int idx,
    const int perspective,
    const int cell,
    const int sign,
    uint64_t *touched_lo,
    uint64_t *touched_hi,
    uint8_t *touched_features,
    int *n_touched
) {
    const EvalNnueCoordFeatures &entry = eval_nnue_coord_to_feature[(size_t)cell];
    for (int i = 0; i < entry.n_features; ++i) {
        const EvalNnueCoordFeature &feature = entry.features[i];
        eval_nnue_pattern_add_delta_local(
            eval,
            idx,
            perspective,
            feature.feature,
            sign * (int)feature.weight,
            touched_lo,
            touched_hi,
            touched_features,
            n_touched
        );
    }
}

inline void eval_nnue_pattern_add_delta_pair_local(
    Eval_search *eval,
    const int idx,
    const int feature_idx,
    const int delta0,
    const int delta1,
    uint64_t *touched_lo,
    uint64_t *touched_hi,
    uint8_t *touched_features,
    int *n_touched
) {
    eval->pattern_features[idx][0][feature_idx] =
        (uint16_t)((int)eval->pattern_features[idx][0][feature_idx] + delta0);
    eval->pattern_features[idx][1][feature_idx] =
        (uint16_t)((int)eval->pattern_features[idx][1][feature_idx] + delta1);
    eval_nnue_pattern_touch(feature_idx, touched_lo, touched_hi, touched_features, n_touched);
}

inline void eval_nnue_pattern_apply_cell_delta_pair_local(
    Eval_search *eval,
    const int idx,
    const int cell,
    const int sign0,
    const int sign1,
    uint64_t *touched_lo,
    uint64_t *touched_hi,
    uint8_t *touched_features,
    int *n_touched
) {
    const EvalNnueCoordFeatures &entry = eval_nnue_coord_to_feature[(size_t)cell];
    for (int i = 0; i < entry.n_features; ++i) {
        const EvalNnueCoordFeature &feature = entry.features[i];
        eval_nnue_pattern_add_delta_pair_local(
            eval,
            idx,
            feature.feature,
            sign0 * (int)feature.weight,
            sign1 * (int)feature.weight,
            touched_lo,
            touched_hi,
            touched_features,
            n_touched
        );
    }
}

inline void eval_nnue_pattern_flush_touched(
    Eval_search *eval,
    const int src_idx,
    const int src_perspective,
    const int dst_idx,
    const int dst_perspective,
    const uint8_t *touched_features,
    const int n_touched
) {
    int16_t *acc = eval->accumulator[dst_idx][dst_perspective];
    for (int i = 0; i < n_touched; ++i) {
        const int feature_idx = touched_features[i];
        const uint16_t old_local = eval->pattern_features[src_idx][src_perspective][feature_idx];
        const uint16_t new_local = eval->pattern_features[dst_idx][dst_perspective][feature_idx];
        if (old_local == new_local) {
            continue;
        }
        eval_nnue_replace_feature(
            acc,
            eval_nnue_pattern_global_feature(feature_idx, old_local),
            eval_nnue_pattern_global_feature(feature_idx, new_local)
        );
    }
}

inline void calc_eval_features(Board *board, Eval_search *eval) {
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN ||
        eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR) {
        eval->feature_idx = 0;
        eval_nnue_calc_pattern_accumulator(board, eval, 0, 0);
        if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR) {
            Board opponent_view = *board;
            opponent_view.pass();
            eval_nnue_calc_pattern_accumulator(&opponent_view, eval, 0, 1);
        }
        return;
    }
    eval->feature_idx = 0;
    int16_t *stm = eval->accumulator[0][0];
    int16_t *non_stm = eval->accumulator[0][1];
    std::memcpy(stm, eval_nnue_ft_bias.data(), sizeof(int16_t) * (size_t)eval_nnue_ft_dim);
    std::memcpy(non_stm, eval_nnue_ft_bias.data(), sizeof(int16_t) * (size_t)eval_nnue_ft_dim);

    uint64_t bits = board->player;
    for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
        eval_nnue_add_feature(stm, (int)cell);
        eval_nnue_add_feature(non_stm, (int)cell + HW2);
    }
    bits = board->opponent;
    for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
        eval_nnue_add_feature(stm, (int)cell + HW2);
        eval_nnue_add_feature(non_stm, (int)cell);
    }
}

inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board) {
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN) {
        const int prev = (int)eval->feature_idx;
        const int next = (int)eval->feature_idx + 1;
        eval_nnue_copy_accumulator(eval->accumulator[next][0], eval->accumulator[prev][0]);
        std::memcpy(
            eval->pattern_features[next][0],
            eval->pattern_features[prev][0],
            sizeof(uint16_t) * (size_t)ADJ_N_FEATURES
        );

        eval_nnue_pattern_apply_cell_delta(eval, next, 0, (int)flip->pos, -1);
        uint64_t bits = board->player & ~flip->flip;
        for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
            eval_nnue_pattern_apply_cell_delta(eval, next, 0, (int)cell, 1);
        }
        bits = board->opponent & ~flip->flip;
        for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
            eval_nnue_pattern_apply_cell_delta(eval, next, 0, (int)cell, -1);
        }
        eval_nnue_pattern_set_feature(
            eval,
            next,
            0,
            ADJ_N_FEATURES - 1,
            (uint16_t)pop_count_ull(board->opponent & ~flip->flip)
        );
        eval->feature_idx = (uint_fast8_t)next;
        return;
    }
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR) {
        const int prev = (int)eval->feature_idx;
        const int next = prev + 1;
        eval_nnue_copy_accumulator(eval->accumulator[next][0], eval->accumulator[prev][1]);
        eval_nnue_copy_accumulator(eval->accumulator[next][1], eval->accumulator[prev][0]);
        std::memcpy(
            eval->pattern_features[next][0],
            eval->pattern_features[prev][1],
            sizeof(uint16_t) * (size_t)ADJ_N_FEATURES
        );
        std::memcpy(
            eval->pattern_features[next][1],
            eval->pattern_features[prev][0],
            sizeof(uint16_t) * (size_t)ADJ_N_FEATURES
        );

        uint64_t touched_lo = 0;
        uint64_t touched_hi = 0;
        uint8_t touched_features[ADJ_N_FEATURES] = {};
        int n_touched = 0;

        eval_nnue_pattern_apply_cell_delta_pair_local(
            eval, next, (int)flip->pos, -1, -2, &touched_lo, &touched_hi, touched_features, &n_touched
        );
        uint64_t bits = flip->flip;
        for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
            eval_nnue_pattern_apply_cell_delta_pair_local(
                eval, next, (int)cell, 1, -1, &touched_lo, &touched_hi, touched_features, &n_touched
            );
        }
        const int n_flipped = pop_count_ull(flip->flip);
        eval_nnue_pattern_add_delta_pair_local(
            eval, next, ADJ_N_FEATURES - 1, -n_flipped, n_flipped + 1, &touched_lo, &touched_hi, touched_features, &n_touched
        );
        eval_nnue_pattern_flush_touched(eval, prev, 1, next, 0, touched_features, n_touched);
        eval_nnue_pattern_flush_touched(eval, prev, 0, next, 1, touched_features, n_touched);
        eval->feature_idx = (uint_fast8_t)next;
        return;
    }
    const int prev = (int)eval->feature_idx;
    const int next = prev + 1;
    int16_t *next_stm = eval->accumulator[next][0];
    int16_t *next_non_stm = eval->accumulator[next][1];
    eval_nnue_copy_accumulator(next_stm, eval->accumulator[prev][1]);
    eval_nnue_copy_accumulator(next_non_stm, eval->accumulator[prev][0]);

    uint64_t bits = flip->flip;
    for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
        eval_nnue_sub_feature(next_stm, (int)cell);
        eval_nnue_add_feature(next_stm, (int)cell + HW2);
        eval_nnue_sub_feature(next_non_stm, (int)cell + HW2);
        eval_nnue_add_feature(next_non_stm, (int)cell);
    }
    eval_nnue_add_feature(next_stm, (int)flip->pos + HW2);
    eval_nnue_add_feature(next_non_stm, (int)flip->pos);
    eval->feature_idx = (uint_fast8_t)next;
}

inline void eval_move(Eval_search *eval, const Flip *flip) {
    eval_move(eval, flip, nullptr);
}

inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_pass(Eval_search *eval, const Board *board) {
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN) {
        const int idx = (int)eval->feature_idx;
        uint64_t bits = board->player;
        for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
            eval_nnue_pattern_apply_cell_delta(eval, idx, 0, (int)cell, 1);
        }
        bits = board->opponent;
        for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)) {
            eval_nnue_pattern_apply_cell_delta(eval, idx, 0, (int)cell, -1);
        }
        eval_nnue_pattern_set_feature(
            eval,
            idx,
            0,
            ADJ_N_FEATURES - 1,
            (uint16_t)pop_count_ull(board->opponent)
        );
        return;
    }
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR) {
        const int idx = (int)eval->feature_idx;
        alignas(32) int16_t tmp_acc[EVAL_NNUE_MAX_FT_DIM];
        uint16_t tmp_features[ADJ_N_FEATURES];
        eval_nnue_copy_accumulator(tmp_acc, eval->accumulator[idx][0]);
        eval_nnue_copy_accumulator(eval->accumulator[idx][0], eval->accumulator[idx][1]);
        eval_nnue_copy_accumulator(eval->accumulator[idx][1], tmp_acc);
        std::memcpy(tmp_features, eval->pattern_features[idx][0], sizeof(uint16_t) * (size_t)ADJ_N_FEATURES);
        std::memcpy(eval->pattern_features[idx][0], eval->pattern_features[idx][1], sizeof(uint16_t) * (size_t)ADJ_N_FEATURES);
        std::memcpy(eval->pattern_features[idx][1], tmp_features, sizeof(uint16_t) * (size_t)ADJ_N_FEATURES);
        return;
    }
    const int idx = (int)eval->feature_idx;
    alignas(32) int16_t tmp[EVAL_NNUE_MAX_FT_DIM];
    eval_nnue_copy_accumulator(tmp, eval->accumulator[idx][0]);
    eval_nnue_copy_accumulator(eval->accumulator[idx][0], eval->accumulator[idx][1]);
    eval_nnue_copy_accumulator(eval->accumulator[idx][1], tmp);
}

inline void eval_pass(Eval_search *eval) {
    eval_pass(eval, nullptr);
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board) {
    eval_move(eval, flip, board);
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip) {
    eval_move(eval, flip, nullptr);
}

inline void eval_undo_endsearch(Eval_search *eval) {
    eval_undo(eval);
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
    eval_pass(eval, board);
}

inline void eval_pass_endsearch(Eval_search *eval) {
    eval_pass(eval, nullptr);
}

inline uint8_t eval_nnue_clamp_u8_shifted(const int32_t value, const int shift) {
    const int32_t shifted = shift > 0 ? (value >> shift) : value;
    return (uint8_t)std::clamp<int32_t>(shifted, 0, 127);
}

inline void eval_nnue_clamp_i16_to_u8_shifted(const int16_t *in, uint8_t *out, const int n, const int shift) {
#if USE_SIMD
    const __m256i zero = _mm256_setzero_si256();
    const __m256i upper = _mm256_set1_epi16(127);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 16));
        if (shift > 0) {
            lo = _mm256_srai_epi16(lo, shift);
            hi = _mm256_srai_epi16(hi, shift);
        }
        lo = _mm256_min_epi16(_mm256_max_epi16(lo, zero), upper);
        hi = _mm256_min_epi16(_mm256_max_epi16(hi, zero), upper);
        __m256i packed = _mm256_packus_epi16(lo, hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
    }
    for (; i < n; ++i) {
        out[i] = eval_nnue_clamp_u8_shifted(in[i], shift);
    }
#else
    for (int i = 0; i < n; ++i) {
        out[i] = eval_nnue_clamp_u8_shifted(in[i], shift);
    }
#endif
}

inline void eval_nnue_make_post_input(const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM], uint8_t *out) {
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN) {
        eval_nnue_clamp_i16_to_u8_shifted(acc[0], out, eval_nnue_ft_dim, eval_nnue_ft_shift);
    } else {
        eval_nnue_clamp_i16_to_u8_shifted(acc[0], out, eval_nnue_ft_dim, eval_nnue_ft_shift);
        eval_nnue_clamp_i16_to_u8_shifted(acc[1], out + eval_nnue_ft_dim, eval_nnue_ft_dim, eval_nnue_ft_shift);
    }
    const int padded = eval_nnue_post_input_padded();
    for (int i = eval_nnue_post_input_dim(); i < padded; ++i) {
        out[i] = 0;
    }
}

#if USE_SIMD
inline int32_t eval_nnue_hsum_i32_avx2(const __m256i sum, const int32_t bias) {
    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    __m128i total = _mm_add_epi32(lo, hi);
    total = _mm_hadd_epi32(total, total);
    total = _mm_hadd_epi32(total, total);
    return bias + _mm_cvtsi128_si32(total);
}

inline int32_t eval_nnue_dot_u8s8_avx2(const uint8_t *input, const int8_t *weight, const int n_padded, const int32_t bias) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < n_padded; i += 32) {
        const __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + i));
        const __m256i prod16 = _mm256_maddubs_epi16(in, w);
        const __m256i prod32 = _mm256_madd_epi16(prod16, ones);
        sum = _mm256_add_epi32(sum, prod32);
    }
    return eval_nnue_hsum_i32_avx2(sum, bias);
}

inline void eval_nnue_accumulate_u8s8_32_avx2(__m256i *sum, const uint8_t *input, const int8_t *weight, const __m256i ones) {
    const __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight));
    const __m256i prod16 = _mm256_maddubs_epi16(in, w);
    const __m256i prod32 = _mm256_madd_epi16(prod16, ones);
    *sum = _mm256_add_epi32(*sum, prod32);
}

inline int32_t eval_nnue_dot_u8s8_32_avx2(const uint8_t *input, const int8_t *weight, const int32_t bias) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input, weight, ones);
    return eval_nnue_hsum_i32_avx2(sum, bias);
}

inline int32_t eval_nnue_dot_u8s8_64_avx2(const uint8_t *input, const int8_t *weight, const int32_t bias) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input, weight, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 32, weight + 32, ones);
    return eval_nnue_hsum_i32_avx2(sum, bias);
}

inline int32_t eval_nnue_dot_u8s8_256_avx2(const uint8_t *input, const int8_t *weight, const int32_t bias) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input, weight, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 32, weight + 32, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 64, weight + 64, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 96, weight + 96, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 128, weight + 128, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 160, weight + 160, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 192, weight + 192, ones);
    eval_nnue_accumulate_u8s8_32_avx2(&sum, input + 224, weight + 224, ones);
    return eval_nnue_hsum_i32_avx2(sum, bias);
}

inline void eval_nnue_layer_32_outputs_avx2(
    const uint8_t *input,
    const int8_t *weight,
    const int weight_stride,
    const int input_len,
    const int32_t *bias,
    uint8_t *output,
    const int shift
) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < 32; i += 4) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        for (int offset = 0; offset < input_len; offset += 32) {
            const __m256i in = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + offset));
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + (size_t)(i + 0) * weight_stride + offset));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + (size_t)(i + 1) * weight_stride + offset));
            const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + (size_t)(i + 2) * weight_stride + offset));
            const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + (size_t)(i + 3) * weight_stride + offset));
            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_maddubs_epi16(in, w0), ones));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_maddubs_epi16(in, w1), ones));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_maddubs_epi16(in, w2), ones));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_maddubs_epi16(in, w3), ones));
        }
        output[i + 0] = eval_nnue_clamp_u8_shifted(eval_nnue_hsum_i32_avx2(sum0, bias[i + 0]), shift);
        output[i + 1] = eval_nnue_clamp_u8_shifted(eval_nnue_hsum_i32_avx2(sum1, bias[i + 1]), shift);
        output[i + 2] = eval_nnue_clamp_u8_shifted(eval_nnue_hsum_i32_avx2(sum2, bias[i + 2]), shift);
        output[i + 3] = eval_nnue_clamp_u8_shifted(eval_nnue_hsum_i32_avx2(sum3, bias[i + 3]), shift);
    }
}
#endif

inline int32_t eval_nnue_dot_u8s8(const uint8_t *input, const int8_t *weight, const int n_padded, const int32_t bias) {
#if USE_SIMD
    return eval_nnue_dot_u8s8_avx2(input, weight, n_padded, bias);
#else
    int32_t sum = bias;
    for (int i = 0; i < n_padded; ++i) {
        sum += (int32_t)input[i] * (int32_t)weight[i];
    }
    return sum;
#endif
}

inline int eval_nnue_finalize_output(int32_t raw) {
    if (eval_nnue_output_shift > 0) {
        raw = raw >= 0 ? ((raw + (1 << (eval_nnue_output_shift - 1))) >> eval_nnue_output_shift)
                       : -(((-raw) + (1 << (eval_nnue_output_shift - 1))) >> eval_nnue_output_shift);
    }
    raw += raw >= 0 ? STEP_2 : -STEP_2;
    int value = raw / STEP;
    return std::clamp(value, -SCORE_MAX, SCORE_MAX);
}

#if USE_SIMD
inline int eval_nnue_forward_128_32_32_avx2(const int phase_idx, const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM]) {
    alignas(32) uint8_t post_input[256];
    alignas(32) uint8_t hidden1[32];
    alignas(32) uint8_t hidden2[32];

    eval_nnue_clamp_i16_to_u8_shifted(acc[0], post_input, 128, eval_nnue_ft_shift);
    eval_nnue_clamp_i16_to_u8_shifted(acc[1], post_input + 128, 128, eval_nnue_ft_shift);

    for (int i = 0; i < 32; ++i) {
        const int32_t value = eval_nnue_dot_u8s8_256_avx2(
            post_input,
            &eval_nnue_hidden1_weight[(size_t)i * 256],
            eval_nnue_hidden1_bias[(size_t)i]
        );
        hidden1[i] = eval_nnue_clamp_u8_shifted(value, eval_nnue_hidden1_shift);
    }
    for (int i = 0; i < 32; ++i) {
        const int32_t value = eval_nnue_dot_u8s8_32_avx2(
            hidden1,
            &eval_nnue_hidden2_weight[(size_t)i * 32],
            eval_nnue_hidden2_bias[(size_t)i]
        );
        hidden2[i] = eval_nnue_clamp_u8_shifted(value, eval_nnue_hidden2_shift);
    }
    const int32_t raw = eval_nnue_dot_u8s8_32_avx2(
        hidden2,
        &eval_nnue_output_weight[(size_t)phase_idx * 32],
        eval_nnue_output_bias[(size_t)phase_idx]
    );
    return eval_nnue_finalize_output(raw);
}

inline int eval_nnue_forward_32_32_32_avx2(const int phase_idx, const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM]) {
    alignas(32) uint8_t post_input[32];
    alignas(32) uint8_t hidden1[32];
    alignas(32) uint8_t hidden2[32];

    eval_nnue_clamp_i16_to_u8_shifted(acc[0], post_input, 32, eval_nnue_ft_shift);

    eval_nnue_layer_32_outputs_avx2(
        post_input,
        eval_nnue_hidden1_weight.data(),
        32,
        32,
        eval_nnue_hidden1_bias.data(),
        hidden1,
        eval_nnue_hidden1_shift
    );
    eval_nnue_layer_32_outputs_avx2(
        hidden1,
        eval_nnue_hidden2_weight.data(),
        32,
        32,
        eval_nnue_hidden2_bias.data(),
        hidden2,
        eval_nnue_hidden2_shift
    );
    const int32_t raw = eval_nnue_dot_u8s8_32_avx2(
        hidden2,
        &eval_nnue_output_weight[(size_t)phase_idx * 32],
        eval_nnue_output_bias[(size_t)phase_idx]
    );
    return eval_nnue_finalize_output(raw);
}

inline int eval_nnue_forward_64_32_32_avx2(const int phase_idx, const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM]) {
    alignas(32) uint8_t post_input[64];
    alignas(32) uint8_t hidden1[32];
    alignas(32) uint8_t hidden2[32];

    eval_nnue_clamp_i16_to_u8_shifted(acc[0], post_input, 32, eval_nnue_ft_shift);
    eval_nnue_clamp_i16_to_u8_shifted(acc[1], post_input + 32, 32, eval_nnue_ft_shift);

    eval_nnue_layer_32_outputs_avx2(
        post_input,
        eval_nnue_hidden1_weight.data(),
        64,
        64,
        eval_nnue_hidden1_bias.data(),
        hidden1,
        eval_nnue_hidden1_shift
    );
    eval_nnue_layer_32_outputs_avx2(
        hidden1,
        eval_nnue_hidden2_weight.data(),
        32,
        32,
        eval_nnue_hidden2_bias.data(),
        hidden2,
        eval_nnue_hidden2_shift
    );
    const int32_t raw = eval_nnue_dot_u8s8_32_avx2(
        hidden2,
        &eval_nnue_output_weight[(size_t)phase_idx * 32],
        eval_nnue_output_bias[(size_t)phase_idx]
    );
    return eval_nnue_finalize_output(raw);
}
#endif

inline int eval_nnue_forward_from_accumulator(const int phase_idx, const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM]) {
#if USE_SIMD
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_STONE &&
        eval_nnue_ft_dim == 128 && eval_nnue_hidden1_dim == 32 && eval_nnue_hidden2_dim == 32) {
        return eval_nnue_forward_128_32_32_avx2(phase_idx, acc);
    }
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN &&
        eval_nnue_ft_dim == 32 && eval_nnue_hidden1_dim == 32 && eval_nnue_hidden2_dim == 32) {
        return eval_nnue_forward_32_32_32_avx2(phase_idx, acc);
    }
    if (eval_nnue_input_kind == EVAL_NNUE_INPUT_KIND_PATTERN_PAIR &&
        eval_nnue_ft_dim == 32 && eval_nnue_hidden1_dim == 32 && eval_nnue_hidden2_dim == 32) {
        return eval_nnue_forward_64_32_32_avx2(phase_idx, acc);
    }
#endif
    alignas(32) uint8_t post_input[EVAL_NNUE_MAX_POST_INPUT];
    alignas(32) uint8_t hidden1[EVAL_NNUE_MAX_HIDDEN1];
    alignas(32) uint8_t hidden2[EVAL_NNUE_MAX_HIDDEN2];

    eval_nnue_make_post_input(acc, post_input);
    const int post_input_padded = eval_nnue_post_input_padded();
    for (int i = 0; i < eval_nnue_hidden1_dim; ++i) {
        const int32_t value = eval_nnue_dot_u8s8(
            post_input,
            &eval_nnue_hidden1_weight[(size_t)i * (size_t)post_input_padded],
            post_input_padded,
            eval_nnue_hidden1_bias[(size_t)i]
        );
        hidden1[i] = eval_nnue_clamp_u8_shifted(value, eval_nnue_hidden1_shift);
    }
    const int hidden1_padded = eval_nnue_hidden1_padded();
    for (int i = eval_nnue_hidden1_dim; i < hidden1_padded; ++i) {
        hidden1[i] = 0;
    }
    for (int i = 0; i < eval_nnue_hidden2_dim; ++i) {
        const int32_t value = eval_nnue_dot_u8s8(
            hidden1,
            &eval_nnue_hidden2_weight[(size_t)i * (size_t)hidden1_padded],
            hidden1_padded,
            eval_nnue_hidden2_bias[(size_t)i]
        );
        hidden2[i] = eval_nnue_clamp_u8_shifted(value, eval_nnue_hidden2_shift);
    }
    const int hidden2_padded = eval_nnue_hidden2_padded();
    for (int i = eval_nnue_hidden2_dim; i < hidden2_padded; ++i) {
        hidden2[i] = 0;
    }
    int32_t raw = eval_nnue_dot_u8s8(
        hidden2,
        &eval_nnue_output_weight[(size_t)phase_idx * (size_t)hidden2_padded],
        hidden2_padded,
        eval_nnue_output_bias[(size_t)phase_idx]
    );
    return eval_nnue_finalize_output(raw);
}

inline int mid_evaluate(Board *board) {
    Eval_search eval;
    calc_eval_features(board, &eval);
    const int phase_idx = (board->n_discs() - 4) / PHASE_N_DISCS;
    return eval_nnue_forward_from_accumulator(phase_idx, eval.accumulator[eval.feature_idx]);
}

class Search;
inline int mid_evaluate_diff(Search *search);
inline int mid_evaluate_move_ordering_dim0(Search *search);
inline int mid_evaluate_dim0(Search *search);
inline int mid_evaluate_move_ordering_end(Search *search);
