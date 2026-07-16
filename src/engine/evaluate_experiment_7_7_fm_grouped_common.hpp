/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_grouped_common.hpp
        Fast fused scorer for the 7.7 beta + FM grouped-vector experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include "evaluate_experiment_7_7_fm_simdopt_mmap_common.hpp"

constexpr int EVAL77_FM_GROUPED_VERSION = 10;
constexpr int EVAL77_FM_GROUPED_MAX_GROUPS = N_PHASES;
constexpr size_t EVAL77_FM_GROUPED_ALIGNMENT = 64;

int eval77_fm_group_count = 0;
std::array<uint8_t, N_PHASES> eval77_fm_phase_to_fm_group{};
std::array<float, EVAL77_FM_GROUPED_MAX_GROUPS> eval77_fm_group_vector_scale{};

inline bool eval77_fm_grouped_load_file(const char *file, const bool show_log) {
    if (show_log) {
        std::cerr << "7.7 beta + FM grouped experiment evaluation file " << file << std::endl;
    }

    eval77_fm_loaded = false;
    eval77_fm_payload = nullptr;
    eval77_fm_linear_payload = nullptr;
    eval77_fm_vector_payload = nullptr;
    eval77_fm_param_stride = 0;
    eval77_fm_phase_stride = 0;
    eval77_fm_linear_param_stride = 0;
    eval77_fm_linear_phase_stride = 0;
    eval77_fm_vector_param_stride = 0;
    eval77_fm_vector_phase_stride = 0;
    eval77_fm_dim = 0;
    eval77_fm_file_version = 0;
    eval77_fm_split_layout = false;
    eval77_fm_group_count = 0;
    eval77_fm_phase_to_fm_group.fill(0);
    eval77_fm_group_vector_scale.fill(0.0f);

    if (!eval77_fm_mapped_file.open_readonly(file)) {
        std::cerr << "[ERROR] [FATAL] can't memory-map grouped eval " << file << std::endl;
        return false;
    }

    constexpr size_t header_base_size = 14 + 4 + sizeof(int32_t) * 5 + N_PHASES;
    if (eval77_fm_mapped_file.size < header_base_size + sizeof(float) * N_PHASES) {
        std::cerr << "[ERROR] [FATAL] grouped 7.7-FM file is too short" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    const unsigned char *fixed_header = eval77_fm_mapped_file.data;
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] grouped 7.7-FM experiment expects an EGEV-style file" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    const int32_t version = eval77_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = eval77_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = eval77_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = eval77_fm_read_i32_le(fixed_header + 30);
    const int32_t n_fm_group = eval77_fm_read_i32_le(fixed_header + 34);
    if (version != EVAL77_FM_GROUPED_VERSION) {
        std::cerr << "[ERROR] [FATAL] grouped 7.7-FM supports EGEV version "
                  << EVAL77_FM_GROUPED_VERSION << ", found version " << version << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }
    if (n_phase != N_PHASES || n_param != EVAL77_FM_N_PARAMS_PER_PHASE ||
        dim <= 0 || dim > EVAL77_FM_MAX_DIM ||
        n_fm_group <= 0 || n_fm_group > EVAL77_FM_GROUPED_MAX_GROUPS) {
        std::cerr << "[ERROR] [FATAL] grouped 7.7-FM header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param
                  << " dim=" << dim
                  << " fm_groups=" << n_fm_group << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    const unsigned char *phase_to_group = fixed_header + 38;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (phase_to_group[phase] >= n_fm_group) {
            std::cerr << "[ERROR] [FATAL] grouped 7.7-FM phase_to_fm_group[" << phase
                      << "]=" << (int)phase_to_group[phase]
                      << " is out of range for " << n_fm_group << " groups" << std::endl;
            eval77_fm_mapped_file.reset();
            return false;
        }
        eval77_fm_phase_to_fm_group[phase] = phase_to_group[phase];
    }

    const size_t linear_scales_offset = header_base_size;
    const size_t vector_scales_offset = linear_scales_offset + sizeof(float) * N_PHASES;
    const size_t header_size = vector_scales_offset + sizeof(float) * (size_t)n_fm_group;
    const size_t linear_offset = eval77_fm_align_up_size(header_size, EVAL77_FM_GROUPED_ALIGNMENT);
    const size_t linear_param_stride = sizeof(int16_t);
    const size_t linear_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * linear_param_stride;
    const size_t vector_param_stride = eval77_fm_align_up_size((size_t)dim, 16);
    const size_t vector_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * vector_param_stride;
    const size_t vector_offset = eval77_fm_align_up_size(
        linear_offset + (size_t)N_PHASES * linear_phase_stride,
        EVAL77_FM_GROUPED_ALIGNMENT
    );
    const size_t expected_size = vector_offset + (size_t)n_fm_group * vector_phase_stride;
    if (eval77_fm_mapped_file.size < expected_size) {
        std::cerr << "[ERROR] [FATAL] grouped 7.7-FM payload is truncated" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    std::memcpy(eval77_fm_linear_scale.data(), fixed_header + linear_scales_offset, sizeof(float) * N_PHASES);
    std::memcpy(eval77_fm_group_vector_scale.data(), fixed_header + vector_scales_offset,
                sizeof(float) * (size_t)n_fm_group);

    eval77_fm_dim = dim;
    eval77_fm_file_version = version;
    eval77_fm_group_count = n_fm_group;
    eval77_fm_param_stride = sizeof(int16_t) + (size_t)eval77_fm_dim;
    eval77_fm_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * eval77_fm_param_stride;
    eval77_fm_linear_param_stride = linear_param_stride;
    eval77_fm_linear_phase_stride = linear_phase_stride;
    eval77_fm_vector_param_stride = vector_param_stride;
    eval77_fm_vector_phase_stride = vector_phase_stride;
    eval77_fm_linear_payload = fixed_header + linear_offset;
    eval77_fm_vector_payload = fixed_header + vector_offset;
    eval77_fm_split_layout = true;

    for (int phase = 0; phase < N_PHASES; ++phase) {
        const int group = eval77_fm_phase_to_fm_group[phase];
        eval77_fm_linear_scale_double[phase] = (double)eval77_fm_linear_scale[phase];
        eval77_fm_vector_scale[phase] = eval77_fm_group_vector_scale[group];
        const double vector_scale = (double)eval77_fm_vector_scale[phase];
        eval77_fm_interaction_scale[phase] = 0.5 * vector_scale * vector_scale;
    }

    eval77_fm_loaded = true;
    if (show_log) {
        std::cerr << "7.7-FM grouped evaluation loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << eval77_fm_dim
                  << " fm_groups " << eval77_fm_group_count
                  << " layout grouped-split-aligned"
                  << " mapped_bytes " << eval77_fm_mapped_file.size << std::endl;
    }
    return true;
}

inline bool eval77_fm_fast_load_file(const char *file, const bool show_log) {
    return eval77_fm_grouped_load_file(file, show_log);
}

inline bool eval77_fm_fast_can_use_phase(const int phase_idx) {
    return eval77_fm_loaded && 0 <= phase_idx && phase_idx < N_PHASES;
}

inline bool eval77_fm_fast_can_use_dim16(const int phase_idx) {
    return eval77_fm_fast_can_use_phase(phase_idx) && eval77_fm_dim == 16;
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
    const int fm_group = eval77_fm_phase_to_fm_group[phase_idx];
    return {
        eval77_fm_linear_payload + (size_t)phase_idx * eval77_fm_linear_phase_stride,
        eval77_fm_vector_payload + (size_t)fm_group * eval77_fm_vector_phase_stride,
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
    accum.sum.fill(0);
    accum.sum_sq.fill(0);
}

inline void eval77_fm_fast_add_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const int8_t *vector_quant = eval77_fm_fast_vector_ptr(phase_ptrs, id);
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

    const double score = (double)accum.linear_quant * eval77_fm_linear_scale_double[phase_idx] +
        (double)interaction_quant * eval77_fm_interaction_scale[phase_idx];
    const int rounded = score >= 0.0 ? (int)(score + 0.5) : (int)(score - 0.5);
    return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
}

inline int eval77_fm_fast_score_from_ids_subset_filter(
    const int phase_idx,
    const int active_ids[],
    const int n_active
) {
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
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const int8_t *vector_ptr = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    const __m128i q8 = _mm_load_si128((const __m128i*)vector_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline void eval77_fm_fast_prefetch_id_simd_dim16(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    _mm_prefetch((const char*)eval77_fm_fast_vector_ptr(phase_ptrs, id), _MM_HINT_T0);
    _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, id), _MM_HINT_T0);
}

inline int eval77_fm_fast_finish_simd_dim16(
    const int phase_idx,
    const Eval77FmFastSimdAccum &accum
);

inline void eval77_fm_grouped_add_vector_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    const int8_t *vector_ptr = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    const __m128i q8 = _mm_load_si128((const __m128i*)vector_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline int eval77_fm_grouped_score_ids_simd_dim16(
    const int phase_idx,
    const int active_ids[],
    const int n_active
) {
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);

    for (int i = 0; i < n_active; ++i) {
        int16_t linear_quant;
        std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]), sizeof(linear_quant));
        accum.linear_quant += linear_quant;
    }
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_add_vector_id_simd_dim16(accum, phase_ptrs, active_ids[i]);
    }
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}

inline int eval77_fm_fast_finish_simd_dim16(
    const int phase_idx,
    const Eval77FmFastSimdAccum &accum
) {
    const __m256i sum_sq_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = _mm256_sub_epi32(sum_sq_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = eval77_fm_reduce_epi32(pair_term);

    const double score = (double)accum.linear_quant * eval77_fm_linear_scale_double[phase_idx] +
        (double)interaction_quant * eval77_fm_interaction_scale[phase_idx];
    const int rounded = score >= 0.0 ? (int)(score + 0.5) : (int)(score - 0.5);
    return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
}
#endif
