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

constexpr int EVAL_NNUE_INPUT_FEATURES = 128;
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
inline bool eval_nnue_enabled = false;

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

inline int eval_nnue_post_input_padded() {
    return eval_nnue_post_input_padded_value != 0 ? eval_nnue_post_input_padded_value : eval_nnue_round_up(eval_nnue_ft_dim * 2, EVAL_NNUE_SIMD_WIDTH);
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
    if (in.fail() ||
        version != 1 ||
        input_features != EVAL_NNUE_INPUT_FEATURES ||
        ft_dim == 0 || ft_dim > EVAL_NNUE_MAX_FT_DIM ||
        hidden1_dim == 0 || hidden1_dim > EVAL_NNUE_MAX_HIDDEN1 ||
        hidden2_dim == 0 || hidden2_dim > EVAL_NNUE_MAX_HIDDEN2 ||
        n_phases != N_PHASES) {
        std::cerr << "[ERROR] [FATAL] incompatible NNUE eval header " << file
                  << " version " << version
                  << " input_features " << input_features
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
    eval_nnue_post_input_padded_value = eval_nnue_round_up(eval_nnue_ft_dim * 2, EVAL_NNUE_SIMD_WIDTH);
    eval_nnue_hidden1_padded_value = eval_nnue_round_up(eval_nnue_hidden1_dim, EVAL_NNUE_SIMD_WIDTH);
    eval_nnue_hidden2_padded_value = eval_nnue_round_up(eval_nnue_hidden2_dim, EVAL_NNUE_SIMD_WIDTH);

    const int post_input_padded = eval_nnue_post_input_padded_value;
    const int hidden1_padded = eval_nnue_hidden1_padded_value;
    const int hidden2_padded = eval_nnue_hidden2_padded_value;

    bool ok = true;
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_ft_bias, eval_nnue_ft_dim);
    ok = ok && eval_nnue_read_vector(in, &eval_nnue_ft_weight, EVAL_NNUE_INPUT_FEATURES * eval_nnue_ft_dim);
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
    if (show_log) {
        std::cerr << "NNUE evaluation loaded " << file
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

inline void eval_nnue_copy_accumulator(int16_t *dst, const int16_t *src) {
    std::memcpy(dst, src, sizeof(int16_t) * (size_t)eval_nnue_ft_dim);
}

inline void eval_nnue_add_feature(int16_t *acc, const int feature) {
    const int16_t *w = eval_nnue_feature_weight(feature);
#if USE_SIMD
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

inline void calc_eval_features(Board *board, Eval_search *eval) {
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

inline void eval_move(Eval_search *eval, const Flip *flip, const Board*) {
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

inline void eval_pass(Eval_search *eval, const Board*) {
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
    eval_nnue_clamp_i16_to_u8_shifted(acc[0], out, eval_nnue_ft_dim, eval_nnue_ft_shift);
    eval_nnue_clamp_i16_to_u8_shifted(acc[1], out + eval_nnue_ft_dim, eval_nnue_ft_dim, eval_nnue_ft_shift);
    const int padded = eval_nnue_post_input_padded();
    for (int i = eval_nnue_ft_dim * 2; i < padded; ++i) {
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
#endif

inline int eval_nnue_forward_from_accumulator(const int phase_idx, const int16_t acc[2][EVAL_NNUE_MAX_FT_DIM]) {
#if USE_SIMD
    if (eval_nnue_ft_dim == 128 && eval_nnue_hidden1_dim == 32 && eval_nnue_hidden2_dim == 32) {
        return eval_nnue_forward_128_32_32_avx2(phase_idx, acc);
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
