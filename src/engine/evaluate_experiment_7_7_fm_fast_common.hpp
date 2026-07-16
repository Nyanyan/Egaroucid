/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_fast_common.hpp
        Fast fused scorer for the memory-mapped 7.7 beta + FM evaluation function
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include "evaluate_experiment_7_7_fm_simdopt_mmap_common.hpp"

inline bool eval77_fm_fast_can_use_phase(const int phase_idx) {
    return eval77_fm_loaded && 0 <= phase_idx && phase_idx < N_PHASES;
}

inline bool eval77_fm_fast_can_use_dim16(const int phase_idx) {
    return eval77_fm_fast_can_use_phase(phase_idx) && eval77_fm_dim == 16;
}

inline const unsigned char *eval77_fm_phase_base(const int phase_idx) {
    return eval77_fm_payload + (size_t)phase_idx * eval77_fm_phase_stride;
}

struct Eval77FmFastScalarAccum {
    int32_t linear_quant;
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum;
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum_sq;
};

inline void eval77_fm_fast_clear_scalar(Eval77FmFastScalarAccum &accum) {
    accum.linear_quant = 0;
    accum.sum.fill(0);
    accum.sum_sq.fill(0);
}

inline void eval77_fm_fast_add_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const unsigned char *phase_base,
    const int id
) {
    const unsigned char *param_ptr = phase_base + (size_t)id * eval77_fm_param_stride;
    int16_t linear_quant;
    std::memcpy(&linear_quant, param_ptr, sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const int8_t *vector_quant = (const int8_t*)(param_ptr + sizeof(int16_t));
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int32_t v = vector_quant[dim];
        accum.sum[dim] += v;
        accum.sum_sq[dim] += v * v;
    }
}

inline int eval77_fm_fast_finish_scalar(
    const int phase_idx,
    const Eval77FmFastScalarAccum &accum
) {
    int64_t interaction_quant = 0;
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int64_t s = accum.sum[dim];
        interaction_quant += s * s - accum.sum_sq[dim];
    }

    const double linear_score =
        (double)accum.linear_quant * (double)eval77_fm_linear_scale[phase_idx];
    const double vector_scale = (double)eval77_fm_vector_scale[phase_idx];
    const double fm_score = 0.5 * (double)interaction_quant * vector_scale * vector_scale;
    return std::clamp((int)std::llround(linear_score + fm_score), -SCORE_MAX, SCORE_MAX);
}

#if USE_SIMD
struct Eval77FmFastSimdAccum {
    int32_t linear_quant;
    __m256i sum16;
    __m256i sum_sq_pair32;
};

inline void eval77_fm_fast_clear_simd_dim16(Eval77FmFastSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.linear_quant = 0;
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
}

inline void eval77_fm_fast_add_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const unsigned char *phase_base,
    const int id
) {
    const unsigned char *param_ptr = phase_base + (size_t)id * eval77_fm_param_stride;
    int16_t linear_quant;
    std::memcpy(&linear_quant, param_ptr, sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const __m128i q8 = _mm_loadu_si128((const __m128i*)(param_ptr + sizeof(int16_t)));
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline int eval77_fm_fast_finish_simd_dim16(
    const int phase_idx,
    const Eval77FmFastSimdAccum &accum
) {
    const __m256i sum_sq_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = _mm256_sub_epi32(sum_sq_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = eval77_fm_reduce_epi32(pair_term);

    const double linear_score =
        (double)accum.linear_quant * (double)eval77_fm_linear_scale[phase_idx];
    const double vector_scale = (double)eval77_fm_vector_scale[phase_idx];
    const double fm_score = 0.5 * (double)interaction_quant * vector_scale * vector_scale;
    return std::clamp((int)std::llround(linear_score + fm_score), -SCORE_MAX, SCORE_MAX);
}
#endif
