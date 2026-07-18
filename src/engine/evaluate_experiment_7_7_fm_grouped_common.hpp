/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_grouped_common.hpp
        Fast fused scorer for the 7.7 beta + FM grouped-vector experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include <vector>
#include "evaluate_experiment_7_7_fm_simdopt_mmap_common.hpp"

constexpr int EVAL77_FM_GROUPED_VERSION = 10;
constexpr int EVAL77_FM_GROUPED_MAX_GROUPS = N_PHASES;
constexpr size_t EVAL77_FM_GROUPED_ALIGNMENT = 64;

int eval77_fm_group_count = 0;
std::array<uint8_t, N_PHASES> eval77_fm_phase_to_fm_group{};
std::array<float, EVAL77_FM_GROUPED_MAX_GROUPS> eval77_fm_group_vector_scale{};
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
std::vector<unsigned char> eval77_fm_grouped_materialized_payload;
std::array<const unsigned char *, N_PHASES>
    eval77_fm_grouped_materialized_phase_base{};
bool eval77_fm_grouped_materialized_layout = false;
#endif

inline bool eval77_fm_grouped_load_file(const char *file, const bool show_log) {
    std::string data_file;
    if (!eval77_fm_parse_subset_file_spec(file, data_file)) {
        return false;
    }
    if (show_log) {
        std::cerr << "7.7 beta + FM grouped experiment evaluation file " << data_file << std::endl;
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
        std::cerr << "7.7-FM subset pattern_mask=0x" << std::hex
                  << eval77_fm_subset_pattern_mask << std::dec
                  << " use_count="
                  << (eval77_fm_subset_use_count ? 1 : 0) << std::endl;
#endif
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
    eval77_fm_interaction_phase_mask = 0;
    eval77_fm_group_count = 0;
    eval77_fm_phase_to_fm_group.fill(0);
    eval77_fm_group_vector_scale.fill(0.0f);
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
    eval77_fm_grouped_materialized_payload.clear();
    eval77_fm_grouped_materialized_phase_base.fill(nullptr);
    eval77_fm_grouped_materialized_layout = false;
#endif

    if (!eval77_fm_mapped_file.open_readonly(data_file)) {
        std::cerr << "[ERROR] [FATAL] can't memory-map grouped eval "
                  << data_file << std::endl;
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
        eval77_fm_mapped_file.reset();
        if (!eval77_fm_load_egev4(file, show_log)) {
            return false;
        }
        eval77_fm_group_count = N_PHASES;
        for (int phase = 0; phase < N_PHASES; ++phase) {
            eval77_fm_phase_to_fm_group[phase] =
                static_cast<uint8_t>(phase);
            eval77_fm_group_vector_scale[phase] =
                eval77_fm_vector_scale[phase];
        }
        return true;
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
        if (phase >= 6 && vector_scale != 0.0) {
            eval77_fm_interaction_phase_mask |= uint64_t{1} << phase;
        }
    }

#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
    {
        const unsigned char *src_linear_payload = eval77_fm_linear_payload;
        const unsigned char *src_vector_payload = eval77_fm_vector_payload;
        const size_t src_linear_phase_stride = eval77_fm_linear_phase_stride;
        const size_t src_vector_phase_stride = eval77_fm_vector_phase_stride;
        const size_t src_vector_param_stride = eval77_fm_vector_param_stride;
        const size_t materialized_param_stride = sizeof(int16_t) + (size_t)eval77_fm_dim;
        const size_t materialized_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * materialized_param_stride;
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE_GROUP_RECORDS)
        std::array<int, EVAL77_FM_GROUPED_MAX_GROUPS>
            representative_phase_by_group{};
        representative_phase_by_group.fill(-1);
        for (int phase = 0; phase < N_PHASES; ++phase) {
            const int group = eval77_fm_phase_to_fm_group[phase];
            int &representative = representative_phase_by_group[group];
            if (representative < 0) {
                representative = phase;
                continue;
            }
            if (eval77_fm_linear_scale[phase] !=
                    eval77_fm_linear_scale[representative] ||
                std::memcmp(
                    src_linear_payload +
                        (size_t)phase * src_linear_phase_stride,
                    src_linear_payload +
                        (size_t)representative *
                            src_linear_phase_stride,
                    src_linear_phase_stride
                ) != 0) {
                std::cerr
                    << "[ERROR] [FATAL] whole-phase materialization "
                    << "requires identical linear parameters in group "
                    << group << "; phases " << representative << " and "
                    << phase << " differ" << std::endl;
                eval77_fm_mapped_file.reset();
                return false;
            }
        }
        eval77_fm_grouped_materialized_payload.assign(
            (size_t)eval77_fm_group_count * materialized_phase_stride,
            0
        );
        for (int group = 0; group < eval77_fm_group_count; ++group) {
            const int representative =
                representative_phase_by_group[group];
            if (representative < 0) {
                std::cerr
                    << "[ERROR] [FATAL] whole-phase materialization "
                    << "found an empty group " << group << std::endl;
                eval77_fm_mapped_file.reset();
                return false;
            }
            unsigned char *dst_phase =
                eval77_fm_grouped_materialized_payload.data() +
                (size_t)group * materialized_phase_stride;
            const unsigned char *src_linear_phase =
                src_linear_payload +
                (size_t)representative * src_linear_phase_stride;
            const unsigned char *src_vector_group =
                src_vector_payload +
                (size_t)group * src_vector_phase_stride;
            for (int id = 0;
                 id < EVAL77_FM_N_PARAMS_PER_PHASE;
                 ++id) {
                unsigned char *dst =
                    dst_phase +
                    (size_t)id * materialized_param_stride;
                std::memcpy(
                    dst,
                    src_linear_phase +
                        (size_t)id * sizeof(int16_t),
                    sizeof(int16_t)
                );
                std::memcpy(
                    dst + sizeof(int16_t),
                    src_vector_group +
                        (size_t)id * src_vector_param_stride,
                    (size_t)eval77_fm_dim
                );
            }
        }
        for (int phase = 0; phase < N_PHASES; ++phase) {
            const int group = eval77_fm_phase_to_fm_group[phase];
            eval77_fm_grouped_materialized_phase_base[phase] =
                eval77_fm_grouped_materialized_payload.data() +
                (size_t)group * materialized_phase_stride;
        }
#else
        eval77_fm_grouped_materialized_payload.assign((size_t)N_PHASES * materialized_phase_stride, 0);
        for (int phase = 0; phase < N_PHASES; ++phase) {
            const int group = eval77_fm_phase_to_fm_group[phase];
            unsigned char *dst_phase = eval77_fm_grouped_materialized_payload.data() +
                (size_t)phase * materialized_phase_stride;
            const unsigned char *src_linear_phase = src_linear_payload + (size_t)phase * src_linear_phase_stride;
            const unsigned char *src_vector_group = src_vector_payload + (size_t)group * src_vector_phase_stride;
            for (int id = 0; id < EVAL77_FM_N_PARAMS_PER_PHASE; ++id) {
                unsigned char *dst = dst_phase + (size_t)id * materialized_param_stride;
                std::memcpy(dst, src_linear_phase + (size_t)id * sizeof(int16_t), sizeof(int16_t));
                std::memcpy(dst + sizeof(int16_t),
                            src_vector_group + (size_t)id * src_vector_param_stride,
                            (size_t)eval77_fm_dim);
            }
            eval77_fm_grouped_materialized_phase_base[phase] =
                dst_phase;
        }
#endif
        eval77_fm_payload = eval77_fm_grouped_materialized_payload.data();
        eval77_fm_linear_payload = eval77_fm_payload;
        eval77_fm_vector_payload = eval77_fm_payload + sizeof(int16_t);
        eval77_fm_param_stride = materialized_param_stride;
        eval77_fm_phase_stride = materialized_phase_stride;
        eval77_fm_linear_param_stride = materialized_param_stride;
        eval77_fm_linear_phase_stride = materialized_phase_stride;
        eval77_fm_vector_param_stride = materialized_param_stride;
        eval77_fm_vector_phase_stride = materialized_phase_stride;
        eval77_fm_split_layout = false;
        eval77_fm_grouped_materialized_layout = true;
    }
#endif

    eval77_fm_loaded = true;
    if (show_log) {
        std::cerr << "7.7-FM grouped evaluation loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << eval77_fm_dim
                  << " fm_groups " << eval77_fm_group_count
                  << " layout "
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
                  << (eval77_fm_grouped_materialized_layout ?
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE_GROUP_RECORDS)
                      "grouped-materialized-whole-phase" :
#else
                      "grouped-materialized-interleaved" :
#endif
                      "grouped-split-aligned")
#else
                  << "grouped-split-aligned"
#endif
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

inline bool eval77_fm_fast_can_use_supported_simd_dim(const int phase_idx) {
    return eval77_fm_fast_can_use_dim16(phase_idx);
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
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE_GROUP_RECORDS)
    if (eval77_fm_grouped_materialized_layout) {
        const unsigned char *phase_base =
            eval77_fm_grouped_materialized_phase_base[phase_idx];
        return {
            phase_base,
            phase_base + sizeof(int16_t),
            eval77_fm_linear_param_stride,
            eval77_fm_vector_param_stride
        };
    }
#endif
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
    const int fm_group = eval77_fm_grouped_materialized_layout ? phase_idx : eval77_fm_phase_to_fm_group[phase_idx];
#else
    const int fm_group = eval77_fm_phase_to_fm_group[phase_idx];
#endif
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
#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    accum.sum.fill(0);
    accum.sum_sq.fill(0);
#endif
}

inline void eval77_fm_fast_add_id_scalar(
    Eval77FmFastScalarAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;

#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int8_t *vector_quant = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int32_t v = vector_quant[dim];
        accum.sum[dim] += v;
        accum.sum_sq[dim] += v * v;
    }
#endif
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

inline int eval77_fm_fast_score_from_ids_subset_filter(
    const int phase_idx,
    const int active_ids[],
    const int n_active
) {
    if (!eval77_fm_fast_can_use_phase(phase_idx)) {
        return 0;
    }
    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        return eval77_fm_fast_score_linear_ids_scalar(
            phase_idx, active_ids, n_active
        );
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
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    accum.linear_quant = 0;
#else
    const __m256i zero = _mm256_setzero_si256();
    accum.linear_quant = 0;
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
#endif
}

inline void eval77_fm_fast_add_id_simd_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    int16_t linear_quant;
    std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, id), sizeof(linear_quant));
    accum.linear_quant += linear_quant;

#if !defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    const int8_t *vector_ptr = eval77_fm_fast_vector_ptr(phase_ptrs, id);
    const __m128i q8 = _mm_load_si128((const __m128i*)vector_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
#endif
}

inline int eval77_fm_fast_finish_simd_dim16(
    const int phase_idx,
    const Eval77FmFastSimdAccum &accum
);

#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_DIRECT_RECORD)
inline const unsigned char *eval77_fm_grouped_materialized_record_ptr(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    constexpr size_t record_stride = sizeof(int16_t) + 16;
    return phase_ptrs.linear_base + static_cast<size_t>(id) * record_stride;
}

inline void eval77_fm_grouped_prefetch_materialized_record_dim16(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    _mm_prefetch(
        reinterpret_cast<const char *>(
            eval77_fm_grouped_materialized_record_ptr(phase_ptrs, id)
        ),
        _MM_HINT_T0
    );
}

inline void eval77_fm_grouped_add_materialized_record_dim16(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    const unsigned char *record_ptr =
        eval77_fm_grouped_materialized_record_ptr(phase_ptrs, id);
    int16_t linear_quant;
    std::memcpy(&linear_quant, record_ptr, sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const __m128i q8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(record_ptr + sizeof(int16_t))
    );
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(
        accum.sum_sq_pair32,
        _mm256_madd_epi16(q16, q16)
    );
}

#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_DIRECT_OFFSET)
inline void eval77_fm_grouped_add_materialized_offset_dim16(
    Eval77FmFastSimdAccum &accum,
    const unsigned char *phase_base,
    const int record_offset
) {
    const unsigned char *record_ptr = phase_base + record_offset;
    int16_t linear_quant;
    std::memcpy(&linear_quant, record_ptr, sizeof(linear_quant));
    accum.linear_quant += linear_quant;

    const __m128i q8 = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(record_ptr + sizeof(int16_t))
    );
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(
        accum.sum_sq_pair32,
        _mm256_madd_epi16(q16, q16)
    );
}

inline void eval77_fm_grouped_prefetch_materialized_offset_dim16(
    const unsigned char *phase_base,
    const int record_offset
) {
    _mm_prefetch(
        reinterpret_cast<const char *>(phase_base + record_offset),
        _MM_HINT_T0
    );
}

inline int eval77_fm_grouped_score_offsets_simd_dim16(
    const int phase_idx,
    const int record_offsets[],
    const int n_active
) {
    const Eval77FmFastPhasePtrs phase_ptrs =
        eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_prefetch_materialized_offset_dim16(
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }

    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        int32_t linear_quant = 0;
        for (int i = 0; i < n_active; ++i) {
            int16_t value;
            std::memcpy(
                &value,
                phase_ptrs.linear_base + record_offsets[i],
                sizeof(value)
            );
            linear_quant += value;
        }
        return eval77_fm_fast_finish_linear_quant(
            phase_idx,
            linear_quant
        );
    }

    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_add_materialized_offset_dim16(
            accum,
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}

#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
inline int eval77_fm_grouped_score_subset_offsets_simd_dim16(
    const int phase_idx,
    const int fm_offsets[],
    const int n_fm,
    const int linear_only_offsets[],
    const int n_linear_only
) {
    const Eval77FmFastPhasePtrs phase_ptrs =
        eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_fm; ++i) {
        eval77_fm_grouped_prefetch_materialized_offset_dim16(
            phase_ptrs.linear_base,
            fm_offsets[i]
        );
    }
    for (int i = 0; i < n_linear_only; ++i) {
        eval77_fm_grouped_prefetch_materialized_offset_dim16(
            phase_ptrs.linear_base,
            linear_only_offsets[i]
        );
    }

    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);
    for (int i = 0; i < n_fm; ++i) {
        eval77_fm_grouped_add_materialized_offset_dim16(
            accum,
            phase_ptrs.linear_base,
            fm_offsets[i]
        );
    }
    for (int i = 0; i < n_linear_only; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            phase_ptrs.linear_base + linear_only_offsets[i],
            sizeof(value)
        );
        accum.linear_quant += value;
    }
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}

#if defined(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK)
template <int pattern_type>
inline void eval77_fm_grouped_add_compiled_subset_block_dim16(
    Eval77FmFastSimdAccum &accum,
    const unsigned char *phase_base,
    const int record_offsets[]
) {
    static_assert(0 <= pattern_type && pattern_type < 16);
    constexpr bool use_fm =
        (static_cast<uint32_t>(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK) &
         (uint32_t{1} << pattern_type)) != 0;
    for (int i = 0; i < 4; ++i) {
        if constexpr (use_fm) {
            eval77_fm_grouped_add_materialized_offset_dim16(
                accum,
                phase_base,
                record_offsets[i]
            );
        } else {
            int16_t value;
            std::memcpy(
                &value,
                phase_base + record_offsets[i],
                sizeof(value)
            );
            accum.linear_quant += value;
        }
    }
}

inline int eval77_fm_grouped_score_compiled_subset_offsets_simd_dim16(
    const int phase_idx,
    const int record_offsets[]
) {
    constexpr int n_active = N_PATTERN_FEATURES + 1;
    const Eval77FmFastPhasePtrs phase_ptrs =
        eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_prefetch_materialized_offset_dim16(
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }

    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        int32_t linear_quant = 0;
        for (int i = 0; i < n_active; ++i) {
            int16_t value;
            std::memcpy(
                &value,
                phase_ptrs.linear_base + record_offsets[i],
                sizeof(value)
            );
            linear_quant += value;
        }
        return eval77_fm_fast_finish_linear_quant(
            phase_idx,
            linear_quant
        );
    }

    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);
#if (EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK == 0xFFFD || \
     EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK == 0xFFF9) && \
    (!defined(EVAL77_FM_COMPILED_SUBSET_USE_COUNT) || \
     EVAL77_FM_COMPILED_SUBSET_USE_COUNT == 1)
#if EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK == 0xFFFD
    constexpr int excluded_begin = 8;
#else
    constexpr int excluded_begin = 4;
#endif
    constexpr int excluded_end = 12;
    for (int i = 0; i < excluded_begin; ++i) {
        eval77_fm_grouped_add_materialized_offset_dim16(
            accum,
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }
    for (int i = excluded_begin; i < excluded_end; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            phase_ptrs.linear_base + record_offsets[i],
            sizeof(value)
        );
        accum.linear_quant += value;
    }
    for (int i = excluded_end; i < n_active; ++i) {
        eval77_fm_grouped_add_materialized_offset_dim16(
            accum,
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }
#else
    eval77_fm_grouped_add_compiled_subset_block_dim16<3>(
        accum, phase_ptrs.linear_base, record_offsets + 0
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<2>(
        accum, phase_ptrs.linear_base, record_offsets + 4
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<1>(
        accum, phase_ptrs.linear_base, record_offsets + 8
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<0>(
        accum, phase_ptrs.linear_base, record_offsets + 12
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<7>(
        accum, phase_ptrs.linear_base, record_offsets + 16
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<6>(
        accum, phase_ptrs.linear_base, record_offsets + 20
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<5>(
        accum, phase_ptrs.linear_base, record_offsets + 24
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<4>(
        accum, phase_ptrs.linear_base, record_offsets + 28
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<11>(
        accum, phase_ptrs.linear_base, record_offsets + 32
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<10>(
        accum, phase_ptrs.linear_base, record_offsets + 36
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<9>(
        accum, phase_ptrs.linear_base, record_offsets + 40
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<8>(
        accum, phase_ptrs.linear_base, record_offsets + 44
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<15>(
        accum, phase_ptrs.linear_base, record_offsets + 48
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<14>(
        accum, phase_ptrs.linear_base, record_offsets + 52
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<13>(
        accum, phase_ptrs.linear_base, record_offsets + 56
    );
    eval77_fm_grouped_add_compiled_subset_block_dim16<12>(
        accum, phase_ptrs.linear_base, record_offsets + 60
    );

#if defined(EVAL77_FM_COMPILED_SUBSET_USE_COUNT) && \
    EVAL77_FM_COMPILED_SUBSET_USE_COUNT == 0
    int16_t count_value;
    std::memcpy(
        &count_value,
        phase_ptrs.linear_base + record_offsets[N_PATTERN_FEATURES],
        sizeof(count_value)
    );
    accum.linear_quant += count_value;
#else
    eval77_fm_grouped_add_materialized_offset_dim16(
        accum,
        phase_ptrs.linear_base,
        record_offsets[N_PATTERN_FEATURES]
    );
#endif
#endif
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}
#endif
#endif

#if defined(EVAL77_FM_MOVE_ORDERING_PATTERN_MASK)
template <uint32_t pattern_mask, int pattern_type>
inline void eval77_fm_grouped_add_move_ordering_subset_block_dim16(
    Eval77FmFastSimdAccum &accum,
    const unsigned char *phase_base,
    const int record_offsets[]
) {
    static_assert(0 <= pattern_type && pattern_type < 16);
    constexpr bool use_fm =
        (pattern_mask & (uint32_t{1} << pattern_type)) != 0;
    for (int i = 0; i < 4; ++i) {
        if constexpr (use_fm) {
            eval77_fm_grouped_add_materialized_offset_dim16(
                accum,
                phase_base,
                record_offsets[i]
            );
        } else {
            int16_t value;
            std::memcpy(
                &value,
                phase_base + record_offsets[i],
                sizeof(value)
            );
            accum.linear_quant += value;
        }
    }
}

inline int eval77_fm_grouped_score_move_ordering_subset_offsets_simd_dim16(
    const int phase_idx,
    const int record_offsets[]
) {
    static_assert(
        (static_cast<uint32_t>(EVAL77_FM_MOVE_ORDERING_PATTERN_MASK) &
         ~uint32_t{0xFFFF}) == 0
    );
    constexpr uint32_t pattern_mask =
        static_cast<uint32_t>(EVAL77_FM_MOVE_ORDERING_PATTERN_MASK);
    constexpr int n_active = N_PATTERN_FEATURES + 1;
    const Eval77FmFastPhasePtrs phase_ptrs =
        eval77_fm_fast_phase_ptrs(phase_idx);
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_prefetch_materialized_offset_dim16(
            phase_ptrs.linear_base,
            record_offsets[i]
        );
    }

    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
        int32_t linear_quant = 0;
        for (int i = 0; i < n_active; ++i) {
            int16_t value;
            std::memcpy(
                &value,
                phase_ptrs.linear_base + record_offsets[i],
                sizeof(value)
            );
            linear_quant += value;
        }
        return eval77_fm_fast_finish_linear_quant(
            phase_idx,
            linear_quant
        );
    }

    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 3>(
        accum, phase_ptrs.linear_base, record_offsets + 0
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 2>(
        accum, phase_ptrs.linear_base, record_offsets + 4
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 1>(
        accum, phase_ptrs.linear_base, record_offsets + 8
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 0>(
        accum, phase_ptrs.linear_base, record_offsets + 12
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 7>(
        accum, phase_ptrs.linear_base, record_offsets + 16
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 6>(
        accum, phase_ptrs.linear_base, record_offsets + 20
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 5>(
        accum, phase_ptrs.linear_base, record_offsets + 24
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 4>(
        accum, phase_ptrs.linear_base, record_offsets + 28
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 11>(
        accum, phase_ptrs.linear_base, record_offsets + 32
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 10>(
        accum, phase_ptrs.linear_base, record_offsets + 36
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 9>(
        accum, phase_ptrs.linear_base, record_offsets + 40
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 8>(
        accum, phase_ptrs.linear_base, record_offsets + 44
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 15>(
        accum, phase_ptrs.linear_base, record_offsets + 48
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 14>(
        accum, phase_ptrs.linear_base, record_offsets + 52
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 13>(
        accum, phase_ptrs.linear_base, record_offsets + 56
    );
    eval77_fm_grouped_add_move_ordering_subset_block_dim16<pattern_mask, 12>(
        accum, phase_ptrs.linear_base, record_offsets + 60
    );
    eval77_fm_grouped_add_materialized_offset_dim16(
        accum,
        phase_ptrs.linear_base,
        record_offsets[N_PATTERN_FEATURES]
    );
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
}
#endif

#endif
#endif

inline void eval77_fm_fast_add_id_simd_supported_dim(
    Eval77FmFastSimdAccum &accum,
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
    eval77_fm_fast_add_id_simd_dim16(accum, phase_ptrs, id);
}

inline void eval77_fm_fast_prefetch_id_simd_dim16(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int id
) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, id), _MM_HINT_T0);
#else
    _mm_prefetch((const char*)eval77_fm_fast_vector_ptr(phase_ptrs, id), _MM_HINT_T0);
    _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, id), _MM_HINT_T0);
#endif
}

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

#if defined(EVALUATE_EXPERIMENT_7_7_FM_SIMD_LINEAR_SUM)
inline __m256i eval77_fm_grouped_gather_signed_linear(
    const int *linear_base,
    __m256i ids,
    const int stride_in_int16
) {
    if (stride_in_int16 != 1) {
        ids = _mm256_mullo_epi32(
            ids,
            _mm256_set1_epi32(stride_in_int16)
        );
    }
    const __m256i packed = _mm256_i32gather_epi32(
        linear_base,
        ids,
        sizeof(int16_t)
    );
    return _mm256_srai_epi32(_mm256_slli_epi32(packed, 16), 16);
}

inline int32_t eval77_fm_grouped_reduce_i32x8(const __m256i values) {
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(values),
        _mm256_extracti128_si256(values, 1)
    );
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline int32_t eval77_fm_grouped_sum_linear_simd(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int active_ids[],
    const int n_active
) {
    if (phase_ptrs.linear_stride % sizeof(int16_t) != 0) {
        int32_t sum = 0;
        for (int i = 0; i < n_active; ++i) {
            int16_t value;
            std::memcpy(
                &value,
                eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]),
                sizeof(value)
            );
            sum += value;
        }
        return sum;
    }

    const int stride_in_int16 = static_cast<int>(
        phase_ptrs.linear_stride / sizeof(int16_t)
    );
    const int *linear_base =
        reinterpret_cast<const int *>(phase_ptrs.linear_base);
    __m256i vector_sum = _mm256_setzero_si256();
    int i = 0;
    for (; i + 8 <= n_active; i += 8) {
        const __m256i ids = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(active_ids + i)
        );
        vector_sum = _mm256_add_epi32(
            vector_sum,
            eval77_fm_grouped_gather_signed_linear(
                linear_base,
                ids,
                stride_in_int16
            )
        );
    }
    int32_t sum = eval77_fm_grouped_reduce_i32x8(vector_sum);
    for (; i < n_active; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]),
            sizeof(value)
        );
        sum += value;
    }
    return sum;
}
#endif

#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
inline int32_t eval77_fm_grouped_sum_linear_prefetched(
    const Eval77FmFastPhasePtrs &phase_ptrs,
    const int active_ids[],
    const int n_active
) {
    for (int i = 0; i < n_active; ++i) {
        _mm_prefetch((const char*)eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]), _MM_HINT_T0);
    }

    int32_t sum = 0;
    for (int i = 0; i < n_active; ++i) {
        int16_t value;
        std::memcpy(&value, eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]), sizeof(value));
        sum += value;
    }
    return sum;
}
#endif

inline int eval77_fm_grouped_score_ids_simd_dim16(
    const int phase_idx,
    const int active_ids[],
    const int n_active
) {
    if (!eval77_fm_fast_phase_uses_interaction(phase_idx)) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_SIMD_LINEAR_SUM)
        const Eval77FmFastPhasePtrs phase_ptrs =
            eval77_fm_fast_phase_ptrs(phase_idx);
        return eval77_fm_fast_finish_linear_quant(
            phase_idx,
            eval77_fm_grouped_sum_linear_simd(
                phase_ptrs,
                active_ids,
                n_active
            )
        );
#else
        return eval77_fm_fast_score_linear_ids_scalar(
            phase_idx, active_ids, n_active
        );
#endif
    }
    const Eval77FmFastPhasePtrs phase_ptrs = eval77_fm_fast_phase_ptrs(phase_idx);
    Eval77FmFastSimdAccum accum;
    eval77_fm_fast_clear_simd_dim16(accum);

#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_MATERIALIZE)
    if (eval77_fm_grouped_materialized_layout) {
        for (int i = 0; i < n_active; ++i) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_DIRECT_RECORD)
            eval77_fm_grouped_prefetch_materialized_record_dim16(
                phase_ptrs,
                active_ids[i]
            );
#else
            eval77_fm_fast_prefetch_id_simd_dim16(phase_ptrs, active_ids[i]);
#endif
        }
#if defined(EVALUATE_EXPERIMENT_7_7_FM_SIMD_LINEAR_SUM)
        accum.linear_quant = eval77_fm_grouped_sum_linear_simd(
            phase_ptrs,
            active_ids,
            n_active
        );
        for (int i = 0; i < n_active; ++i) {
            eval77_fm_grouped_add_vector_id_simd_dim16(
                accum,
                phase_ptrs,
                active_ids[i]
            );
        }
#else
        for (int i = 0; i < n_active; ++i) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_DIRECT_RECORD)
            eval77_fm_grouped_add_materialized_record_dim16(
                accum,
                phase_ptrs,
                active_ids[i]
            );
#else
            eval77_fm_fast_add_id_simd_dim16(accum, phase_ptrs, active_ids[i]);
#endif
        }
#endif
        return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
    }
#endif

#if defined(EVALUATE_EXPERIMENT_7_7_FM_LINEAR_ONLY)
    accum.linear_quant = eval77_fm_grouped_sum_linear_prefetched(phase_ptrs, active_ids, n_active);
#else
#if defined(EVALUATE_EXPERIMENT_7_7_FM_SIMD_LINEAR_SUM)
    accum.linear_quant = eval77_fm_grouped_sum_linear_simd(
        phase_ptrs,
        active_ids,
        n_active
    );
#else
    for (int i = 0; i < n_active; ++i) {
        int16_t linear_quant;
        std::memcpy(&linear_quant, eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]), sizeof(linear_quant));
        accum.linear_quant += linear_quant;
    }
#endif
    for (int i = 0; i < n_active; ++i) {
        eval77_fm_grouped_add_vector_id_simd_dim16(accum, phase_ptrs, active_ids[i]);
    }
#endif
    return eval77_fm_fast_finish_simd_dim16(phase_idx, accum);
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
#endif
