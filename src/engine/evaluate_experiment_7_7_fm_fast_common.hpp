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

inline bool eval77_fm_fast_load_file(const char *file, const bool show_log) {
    return eval77_fm_load_egev4(file, show_log);
}

inline bool eval77_fm_fast_can_use_phase(const int phase_idx) {
    return eval77_fm_loaded && 0 <= phase_idx && phase_idx < N_PHASES;
}

inline bool eval77_fm_fast_can_use_dim16(const int phase_idx) {
    return eval77_fm_fast_can_use_phase(phase_idx) && eval77_fm_dim == 16;
}

inline bool eval77_fm_fast_can_use_supported_simd_dim(const int phase_idx) {
    return eval77_fm_fast_can_use_phase(phase_idx) &&
        (eval77_fm_dim == 8 || eval77_fm_dim == 12 ||
         eval77_fm_dim == 14 || eval77_fm_dim == 16);
}

inline bool eval77_fm_fast_phase_uses_interaction(const int phase_idx) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    (void)phase_idx;
    return false;
#else
    return ((eval77_fm_interaction_phase_mask >> phase_idx) & uint64_t{1}) != 0;
#endif
}

struct Eval77FmFastPhasePtrs {
    const unsigned char *linear_base;
    const unsigned char *vector_base;
    size_t linear_stride;
    size_t vector_stride;
};

struct Eval77FmFastScalarAccum {
    int32_t linear_quant;
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum;
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum_sq;
};

inline Eval77FmFastPhasePtrs eval77_fm_fast_phase_ptrs(const int phase_idx) {
    return {
        eval77_fm_linear_payload + (size_t)phase_idx * eval77_fm_linear_phase_stride,
        eval77_fm_vector_payload + (size_t)phase_idx * eval77_fm_vector_phase_stride,
        eval77_fm_linear_param_stride,
        eval77_fm_vector_param_stride
    };
}

inline const unsigned char *eval77_fm_fast_linear_ptr(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    return phase_ptrs.linear_base + (size_t)id * phase_ptrs.linear_stride;
}

inline const int8_t *eval77_fm_fast_vector_ptr(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    return (const int8_t*)(phase_ptrs.vector_base + (size_t)id * phase_ptrs.vector_stride);
}

inline void eval77_fm_fast_clear_scalar(Eval77FmFastScalarAccum &accum) {
    accum.linear_quant = 0;
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    accum.sum.fill(0);
    accum.sum_sq.fill(0);
#endif
}

inline void eval77_fm_fast_add_linear_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;
}

inline void eval77_fm_fast_add_vector_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int8_t *vector_quant = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int32_t v = vector_quant[dim];
        accum.sum[dim] += v;
        accum.sum_sq[dim] += v * v;
    }
#else
    (void)accum;
    (void)phase_ptrs;
    (void)id;
#endif
}

inline void eval77_fm_fast_add_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    eval77_fm_fast_add_linear_id_scalar(accum, phase_ptrs, id);
    eval77_fm_fast_add_vector_id_scalar(accum, phase_ptrs, id);
}

inline int eval77_fm_fast_finish_linear_quant(
    const int phase_idx,
    const int32_t linear_quant
) {
    const double score =
        static_cast<double>(linear_quant) * eval77_fm_linear_scale_double[phase_idx];
    const int rounded = score >= 0.0 ? static_cast<int>(score + 0.5) :
        static_cast<int>(score - 0.5);
    return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
}

inline int eval77_fm_fast_score_linear_ids_scalar(
    const int phase_idx,
    const int linear_ids[],
    const int n_linear
) {
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    int32_t linear_quant = 0;
    for (int i = 0; i < n_linear; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            eval77_fm_fast_linear_ptr(phase_ptrs, linear_ids[i]),
            sizeof(value)
        );
        linear_quant += value;
    }
    return eval77_fm_fast_finish_linear_quant(phase_idx, linear_quant);
}

inline int eval77_fm_fast_finish_scalar(
    const int phase_idx,
    const Eval77FmFastScalarAccum &accum
) {
    int64_t interaction_quant = 0;
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int64_t s = accum.sum[dim];
        interaction_quant += s * s - accum.sum_sq[dim];
    }
#endif

    const double score = (double)accum.linear_quant * eval77_fm_linear_scale_double[phase_idx] +
        (double)interaction_quant * eval77_fm_interaction_scale[phase_idx];
    const int rounded = score >= 0.0 ? (int)(score + 0.5) : (int)(score - 0.5);
    return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
}

inline int eval77_fm_fast_score_from_linear_and_fm_ids_scalar(
    const int phase_idx,
    const int linear_ids[],
    const int n_linear,
    const int fm_ids[],
    const int n_fm
) {
    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        return eval77_fm_fast_score_linear_ids_scalar(
            phase_idx, linear_ids, n_linear
        );
    }
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    Eval77FmFastScalarAccum accum;
    eval77_fm_fast_clear_scalar(accum);
    for (int i = 0; i < n_linear; ++i) {
        eval77_fm_fast_add_linear_id_scalar(accum, phase_ptrs, linear_ids[i]);
    }
    for (int i = 0; i < n_fm; ++i) {
        eval77_fm_fast_add_vector_id_scalar(accum, phase_ptrs, fm_ids[i]);
    }
    return eval77_fm_fast_finish_scalar(phase_idx, accum);
}

inline int eval77_fm_fast_score_from_ids_subset_filter(
    const int phase_idx,
    const int active_ids[],
    const int n_active
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    if (!eval77_fm_fast_can_use_phase(phase_idx)) {
        return 0;
    }
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    Eval77FmFastScalarAccum accum;
    eval77_fm_fast_clear_scalar(accum);
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (0 <= id && id < EVAL77_FM_N_PARAMS_PER_PHASE) {
            eval77_fm_fast_add_id_scalar(accum, phase_ptrs, id);
        }
    }
    return eval77_fm_fast_finish_scalar(phase_idx, accum);
#else
    return eval77_fm_score_from_ids_subset_filter(phase_idx, active_ids, n_active);
#endif
}

#if USE_SIMD
struct Eval77FmFastSimdAccum {
    int32_t linear_quant;
    __m256i sum16;
    __m256i sum_sq_pair32;
};

inline void eval77_fm_fast_clear_simd_dim16(Eval77FmFastSimdAccum &accum) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    accum.linear_quant = 0;
#else
    const __m256i zero = _mm256_setzero_si256();
    accum.linear_quant = 0;
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
#endif
}

inline void eval77_fm_fast_add_linear_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;
}

inline int eval77_fm_fast_score_linear_ids_simd(
    const int phase_idx,
    const int linear_ids[],
    const int n_linear
) {
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_linear; ++i) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                eval77_fm_fast_linear_ptr(phase_ptrs, linear_ids[i])
            ),
            _MM_HINT_T0
        );
    }
    int32_t linear_quant = 0;
    for (int i = 0; i < n_linear; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            eval77_fm_fast_linear_ptr(phase_ptrs, linear_ids[i]),
            sizeof(value)
        );
        linear_quant += value;
    }
    return eval77_fm_fast_finish_linear_quant(phase_idx, linear_quant);
}

inline void eval77_fm_fast_add_vector_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int8_t *vector_ptr = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    const __m128i q8 = eval77_fm_split_layout ?
        _mm_load_si128((const __m128i*)vector_ptr) :
        _mm_loadu_si128((const __m128i*)vector_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
#else
    (void)accum;
    (void)phase_ptrs;
    (void)id;
#endif
}

inline void eval77_fm_fast_add_vector_id_simd_supported_dim(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int8_t *vector_ptr = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    __m128i q8;
    if (eval77_fm_dim == 12) {
        q8 = _mm_maskload_epi32(
            (const int*)vector_ptr,
            _mm_set_epi32(0, -1, -1, -1)
        );
    } else if (eval77_fm_dim == 14) {
        q8 = _mm_srli_si128(
            _mm_loadu_si128(
                (const __m128i*)(vector_ptr - sizeof(int16_t))
            ),
            sizeof(int16_t)
        );
    } else {
        q8 = _mm_loadl_epi64((const __m128i*)vector_ptr);
    }
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
#else
    (void)accum;
    (void)phase_ptrs;
    (void)id;
#endif
}

inline void eval77_fm_fast_add_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    eval77_fm_fast_add_linear_id_simd_dim16(accum, phase_ptrs, id);
    eval77_fm_fast_add_vector_id_simd_dim16(accum, phase_ptrs, id);
}

inline void eval77_fm_fast_add_id_simd_supported_dim(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    eval77_fm_fast_add_linear_id_simd_dim16(accum, phase_ptrs, id);
    eval77_fm_fast_add_vector_id_simd_supported_dim(accum, phase_ptrs, id);
}

inline void eval77_fm_fast_prefetch_id_simd_dim16(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, id), _MM_HINT_T0);
#else
    _mm_prefetch((const char*)eval77_fm_fast_vector_ptr(phase_ptrs, id), _MM_HINT_T0);
    if (!eval77_fm_split_layout) {
        _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, id), _MM_HINT_T0);
    }
#endif
}

inline int eval77_fm_fast_finish_simd_dim16(
    const int phase_idx,
    const Eval77FmFastSimdAccum &accum
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int64_t interaction_quant = 0;
#else
    const __m256i sum_sq_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = _mm256_sub_epi32(sum_sq_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = eval77_fm_reduce_epi32(pair_term);
#endif

    const double score = (double)accum.linear_quant * eval77_fm_linear_scale_double[phase_idx] +
        (double)interaction_quant * eval77_fm_interaction_scale[phase_idx];
    const int rounded = score >= 0.0 ? (int)(score + 0.5) : (int)(score - 0.5);
    return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
}

inline int eval77_fm_fast_score_from_linear_and_fm_ids_simd_dim16(
    const int phase_idx,
    const int linear_ids[],
    const int n_linear,
    const int fm_ids[],
    const int n_fm
) {
    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        return eval77_fm_fast_score_linear_ids_simd(
            phase_idx, linear_ids, n_linear
        );
    }
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_linear; ++i) {
        _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, linear_ids[i]), _MM_HINT_T0);
    }
    for (int i = 0; i < n_fm; ++i) {
        _mm_prefetch((const char*)eval77_fm_fast_vector_ptr(phase_ptrs, fm_ids[i]), _MM_HINT_T0);
    }

    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);
    for (int i = 0; i < n_linear; ++i) {
        eval77_fm_fast_add_linear_id_simd_dim16(accum, phase_ptrs, linear_ids[i]);
    }
    if (eval77_fm_dim == 16) {
        for (int i = 0; i < n_fm; ++i) {
            eval77_fm_fast_add_vector_id_simd_dim16(accum, phase_ptrs, fm_ids[i]);
        }
    } else {
        for (int i = 0; i < n_fm; ++i) {
            eval77_fm_fast_add_vector_id_simd_supported_dim(
                accum, phase_ptrs, fm_ids[i]
            );
        }
    }
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}
#endif
