/*
    Egaroucid Project

    @file evaluate_experiment_current_fm_tri_type_stream12_common.hpp
        Streaming dim12 scorer for the isolated current-model + tri-type FM experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include "evaluate_experiment_current_fm_tri_type_simdopt_common.hpp"

#if USE_SIMD
inline void current_fm_load_dim12_ptr(const int8_t *base, const int id, __m256i &lo, __m256i &hi) {
    const int8_t *src_ptr = base + (size_t)id * 12;
    const __m128i q8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    int32_t packed = 0;
    std::memcpy(&packed, src_ptr + 8, sizeof(packed));
    const __m128i q4 = _mm_cvtsi32_si128(packed);
    lo = _mm256_cvtepi8_epi32(q8);
    hi = _mm256_cvtepi8_epi32(q4);
}

template<bool UpdateTotal>
inline void current_fm_add_dim12_ptr(
    const int8_t *base,
    const int id,
    CurrentFmSimdAccum &accum,
    const int pattern_type
) {
    __m256i lo;
    __m256i hi;
    current_fm_load_dim12_ptr(base, id, lo, hi);
    const __m256i lo_sq = _mm256_mullo_epi32(lo, lo);
    const __m256i hi_sq = _mm256_mullo_epi32(hi, hi);
    if constexpr (UpdateTotal) {
        accum.sum_lo = _mm256_add_epi32(accum.sum_lo, lo);
        accum.sum_hi = _mm256_add_epi32(accum.sum_hi, hi);
        accum.sum_sq_lo = _mm256_add_epi32(accum.sum_sq_lo, lo_sq);
        accum.sum_sq_hi = _mm256_add_epi32(accum.sum_sq_hi, hi_sq);
    }
    accum.type_sum_lo[pattern_type] = _mm256_add_epi32(accum.type_sum_lo[pattern_type], lo);
    accum.type_sum_hi[pattern_type] = _mm256_add_epi32(accum.type_sum_hi[pattern_type], hi);
    accum.type_sum_sq_lo[pattern_type] = _mm256_add_epi32(accum.type_sum_sq_lo[pattern_type], lo_sq);
    accum.type_sum_sq_hi[pattern_type] = _mm256_add_epi32(accum.type_sum_sq_hi[pattern_type], hi_sq);
}

inline int current_fm_score_from_ids_stream12(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    if (n_active != CURRENT_FM_N_PATTERN_FEATURES + 1 ||
        current_fm_cross_file.dim != 12 ||
        current_fm_same_file.dim != 12 ||
        current_fm_count_file.dim != 12) {
        return current_fm_score_from_ids(phase_idx, active_ids, n_active);
    }

    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    const __m256i zero = _mm256_setzero_si256();
    const std::vector<int8_t> &cross_phase_vectors = current_fm_cross_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &same_phase_vectors = current_fm_same_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_phase_vectors = current_fm_count_file.vector_quant_by_phase[phase_idx];
    const int8_t *cross_ptr = cross_phase_vectors.empty() ? nullptr : cross_phase_vectors.data();
    const int8_t *same_ptr = same_phase_vectors.empty() ? nullptr : same_phase_vectors.data();
    const int8_t *count_ptr = count_phase_vectors.empty() ? nullptr : count_phase_vectors.data();

    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    __m256i count_pattern_sum_lo = zero;
    __m256i count_pattern_sum_hi = zero;
    __m256i count_vec_lo = zero;
    __m256i count_vec_hi = zero;
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < CURRENT_FM_N_PATTERN_FEATURES; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }

        linear_quant_sum += current_fm_cross_file.linear_quant[phase_base + (size_t)id];
        if (id >= CURRENT_FM_N_PATTERN_PARAMS_RAW) {
            continue;
        }

        const int pattern_type = i >> 2;
        if (cross_ptr != nullptr) {
            current_fm_add_dim12_ptr<true>(cross_ptr, id, cross_accum, pattern_type);
        }
        if (same_ptr != nullptr) {
            current_fm_add_dim12_ptr<false>(same_ptr, id, same_accum, pattern_type);
        }
        if (count_ptr != nullptr) {
            __m256i count_lo;
            __m256i count_hi;
            current_fm_load_dim12_ptr(count_ptr, id, count_lo, count_hi);
            count_pattern_sum_lo = _mm256_add_epi32(count_pattern_sum_lo, count_lo);
            count_pattern_sum_hi = _mm256_add_epi32(count_pattern_sum_hi, count_hi);
        }
    }

    const int count_id = active_ids[CURRENT_FM_N_PATTERN_FEATURES];
    if (0 <= count_id && count_id < CURRENT_FM_N_PARAMS_PER_PHASE) {
        linear_quant_sum += current_fm_cross_file.linear_quant[phase_base + (size_t)count_id];
        if (count_id >= CURRENT_FM_N_PATTERN_PARAMS_RAW && count_ptr != nullptr) {
            current_fm_load_dim12_ptr(count_ptr, count_id, count_vec_lo, count_vec_hi);
        }
    }

    __m256i cross_same_lo = zero;
    __m256i cross_same_hi = zero;
    __m256i same_interaction_lo = zero;
    __m256i same_interaction_hi = zero;
    for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
        cross_same_lo = _mm256_add_epi32(
            cross_same_lo,
            current_fm_pair_term_epi32(cross_accum.type_sum_lo[pattern_type], cross_accum.type_sum_sq_lo[pattern_type])
        );
        cross_same_hi = _mm256_add_epi32(
            cross_same_hi,
            current_fm_pair_term_epi32(cross_accum.type_sum_hi[pattern_type], cross_accum.type_sum_sq_hi[pattern_type])
        );
        same_interaction_lo = _mm256_add_epi32(
            same_interaction_lo,
            current_fm_pair_term_epi32(same_accum.type_sum_lo[pattern_type], same_accum.type_sum_sq_lo[pattern_type])
        );
        same_interaction_hi = _mm256_add_epi32(
            same_interaction_hi,
            current_fm_pair_term_epi32(same_accum.type_sum_hi[pattern_type], same_accum.type_sum_sq_hi[pattern_type])
        );
    }
    const __m256i cross_lo = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_lo, cross_accum.sum_sq_lo),
        cross_same_lo
    );
    const __m256i cross_hi = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_hi, cross_accum.sum_sq_hi),
        cross_same_hi
    );
    const int64_t cross_interaction_quant = current_fm_reduce_epi32(cross_lo) + current_fm_reduce_epi32(cross_hi);
    const int64_t same_interaction_quant = current_fm_reduce_epi32(same_interaction_lo) + current_fm_reduce_epi32(same_interaction_hi);
    const int64_t count_interaction_quant =
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_lo, count_vec_lo)) +
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_hi, count_vec_hi));

    const double cross_scale = current_fm_cross_file.vector_scale[phase_idx];
    const double same_scale = current_fm_same_file.vector_scale[phase_idx];
    const double count_scale = current_fm_count_file.vector_scale[phase_idx];
    const double score = (double)linear_quant_sum * linear_scale
        + current_fm_cross_weight * (0.5 * (double)cross_interaction_quant * cross_scale * cross_scale)
        + current_fm_same_weight * (0.5 * (double)same_interaction_quant * same_scale * same_scale)
        + current_fm_count_weight * ((double)count_interaction_quant * count_scale * count_scale);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline int current_fm_score_from_ids_stream12_unchecked(const int phase_idx, const int active_ids[]) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx ||
        current_fm_cross_file.dim != 12 ||
        current_fm_same_file.dim != 12 ||
        current_fm_count_file.dim != 12) {
        return current_fm_score_from_ids(phase_idx, active_ids, CURRENT_FM_N_PATTERN_FEATURES + 1);
    }

    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const int16_t *linear_ptr = current_fm_cross_file.linear_quant.data() + phase_base;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    const __m256i zero = _mm256_setzero_si256();
    const std::vector<int8_t> &cross_phase_vectors = current_fm_cross_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &same_phase_vectors = current_fm_same_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_phase_vectors = current_fm_count_file.vector_quant_by_phase[phase_idx];
    const int8_t *cross_ptr = cross_phase_vectors.empty() ? nullptr : cross_phase_vectors.data();
    const int8_t *same_ptr = same_phase_vectors.empty() ? nullptr : same_phase_vectors.data();
    const int8_t *count_ptr = count_phase_vectors.empty() ? nullptr : count_phase_vectors.data();

    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    __m256i count_pattern_sum_lo = zero;
    __m256i count_pattern_sum_hi = zero;
    __m256i count_vec_lo = zero;
    __m256i count_vec_hi = zero;
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < CURRENT_FM_N_PATTERN_FEATURES; ++i) {
        const int id = active_ids[i];
        linear_quant_sum += linear_ptr[id];

        const int pattern_type = i >> 2;
        if (cross_ptr != nullptr) {
            current_fm_add_dim12_ptr<true>(cross_ptr, id, cross_accum, pattern_type);
        }
        if (same_ptr != nullptr) {
            current_fm_add_dim12_ptr<false>(same_ptr, id, same_accum, pattern_type);
        }
        if (count_ptr != nullptr) {
            __m256i count_lo;
            __m256i count_hi;
            current_fm_load_dim12_ptr(count_ptr, id, count_lo, count_hi);
            count_pattern_sum_lo = _mm256_add_epi32(count_pattern_sum_lo, count_lo);
            count_pattern_sum_hi = _mm256_add_epi32(count_pattern_sum_hi, count_hi);
        }
    }

    const int count_id = active_ids[CURRENT_FM_N_PATTERN_FEATURES];
    linear_quant_sum += linear_ptr[count_id];
    if (count_ptr != nullptr) {
        current_fm_load_dim12_ptr(count_ptr, count_id, count_vec_lo, count_vec_hi);
    }

    __m256i cross_same_lo = zero;
    __m256i cross_same_hi = zero;
    __m256i same_interaction_lo = zero;
    __m256i same_interaction_hi = zero;
    for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
        cross_same_lo = _mm256_add_epi32(
            cross_same_lo,
            current_fm_pair_term_epi32(cross_accum.type_sum_lo[pattern_type], cross_accum.type_sum_sq_lo[pattern_type])
        );
        cross_same_hi = _mm256_add_epi32(
            cross_same_hi,
            current_fm_pair_term_epi32(cross_accum.type_sum_hi[pattern_type], cross_accum.type_sum_sq_hi[pattern_type])
        );
        same_interaction_lo = _mm256_add_epi32(
            same_interaction_lo,
            current_fm_pair_term_epi32(same_accum.type_sum_lo[pattern_type], same_accum.type_sum_sq_lo[pattern_type])
        );
        same_interaction_hi = _mm256_add_epi32(
            same_interaction_hi,
            current_fm_pair_term_epi32(same_accum.type_sum_hi[pattern_type], same_accum.type_sum_sq_hi[pattern_type])
        );
    }
    const __m256i cross_lo = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_lo, cross_accum.sum_sq_lo),
        cross_same_lo
    );
    const __m256i cross_hi = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_hi, cross_accum.sum_sq_hi),
        cross_same_hi
    );
    const int64_t cross_interaction_quant = current_fm_reduce_epi32(cross_lo) + current_fm_reduce_epi32(cross_hi);
    const int64_t same_interaction_quant = current_fm_reduce_epi32(same_interaction_lo) + current_fm_reduce_epi32(same_interaction_hi);
    const int64_t count_interaction_quant =
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_lo, count_vec_lo)) +
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_hi, count_vec_hi));

    const double cross_scale = current_fm_cross_file.vector_scale[phase_idx];
    const double same_scale = current_fm_same_file.vector_scale[phase_idx];
    const double count_scale = current_fm_count_file.vector_scale[phase_idx];
    const double score = (double)linear_quant_sum * linear_scale
        + current_fm_cross_weight * (0.5 * (double)cross_interaction_quant * cross_scale * cross_scale)
        + current_fm_same_weight * (0.5 * (double)same_interaction_quant * same_scale * same_scale)
        + current_fm_count_weight * ((double)count_interaction_quant * count_scale * count_scale);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline int current_fm_score_from_idx8_stream12_unchecked(
    const int phase_idx,
    const __m256i idx8_groups[8],
    const int count_id
) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx ||
        current_fm_cross_file.dim != 12 ||
        current_fm_same_file.dim != 12 ||
        current_fm_count_file.dim != 12) {
        alignas(32) int active_ids[CURRENT_FM_N_PATTERN_FEATURES + 1];
        int n_active = 0;
        for (int group = 0; group < 8; ++group) {
            alignas(32) int values[8];
            _mm256_store_si256((__m256i*)values, idx8_groups[group]);
            for (int i = 0; i < 8; ++i) {
                active_ids[n_active++] = values[i] - 1;
            }
        }
        active_ids[n_active++] = count_id;
        return current_fm_score_from_ids(phase_idx, active_ids, n_active);
    }

    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const int16_t *linear_ptr = current_fm_cross_file.linear_quant.data() + phase_base;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    const __m256i zero = _mm256_setzero_si256();
    const std::vector<int8_t> &cross_phase_vectors = current_fm_cross_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &same_phase_vectors = current_fm_same_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_phase_vectors = current_fm_count_file.vector_quant_by_phase[phase_idx];
    const int8_t *cross_ptr = cross_phase_vectors.empty() ? nullptr : cross_phase_vectors.data();
    const int8_t *same_ptr = same_phase_vectors.empty() ? nullptr : same_phase_vectors.data();
    const int8_t *count_ptr = count_phase_vectors.empty() ? nullptr : count_phase_vectors.data();

    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    __m256i count_pattern_sum_lo = zero;
    __m256i count_pattern_sum_hi = zero;
    __m256i count_vec_lo = zero;
    __m256i count_vec_hi = zero;
    int64_t linear_quant_sum = 0;

    for (int group = 0; group < 8; ++group) {
        alignas(32) int values[8];
        _mm256_store_si256((__m256i*)values, idx8_groups[group]);
        for (int lane = 0; lane < 8; ++lane) {
            const int id = values[lane] - 1;
            linear_quant_sum += linear_ptr[id];

            const int pattern_type = group * 2 + (lane >> 2);
            if (cross_ptr != nullptr) {
                current_fm_add_dim12_ptr<true>(cross_ptr, id, cross_accum, pattern_type);
            }
            if (same_ptr != nullptr) {
                current_fm_add_dim12_ptr<false>(same_ptr, id, same_accum, pattern_type);
            }
            if (count_ptr != nullptr) {
                __m256i count_lo;
                __m256i count_hi;
                current_fm_load_dim12_ptr(count_ptr, id, count_lo, count_hi);
                count_pattern_sum_lo = _mm256_add_epi32(count_pattern_sum_lo, count_lo);
                count_pattern_sum_hi = _mm256_add_epi32(count_pattern_sum_hi, count_hi);
            }
        }
    }

    linear_quant_sum += linear_ptr[count_id];
    if (count_ptr != nullptr) {
        current_fm_load_dim12_ptr(count_ptr, count_id, count_vec_lo, count_vec_hi);
    }

    __m256i cross_same_lo = zero;
    __m256i cross_same_hi = zero;
    __m256i same_interaction_lo = zero;
    __m256i same_interaction_hi = zero;
    for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
        cross_same_lo = _mm256_add_epi32(
            cross_same_lo,
            current_fm_pair_term_epi32(cross_accum.type_sum_lo[pattern_type], cross_accum.type_sum_sq_lo[pattern_type])
        );
        cross_same_hi = _mm256_add_epi32(
            cross_same_hi,
            current_fm_pair_term_epi32(cross_accum.type_sum_hi[pattern_type], cross_accum.type_sum_sq_hi[pattern_type])
        );
        same_interaction_lo = _mm256_add_epi32(
            same_interaction_lo,
            current_fm_pair_term_epi32(same_accum.type_sum_lo[pattern_type], same_accum.type_sum_sq_lo[pattern_type])
        );
        same_interaction_hi = _mm256_add_epi32(
            same_interaction_hi,
            current_fm_pair_term_epi32(same_accum.type_sum_hi[pattern_type], same_accum.type_sum_sq_hi[pattern_type])
        );
    }
    const __m256i cross_lo = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_lo, cross_accum.sum_sq_lo),
        cross_same_lo
    );
    const __m256i cross_hi = _mm256_sub_epi32(
        current_fm_pair_term_epi32(cross_accum.sum_hi, cross_accum.sum_sq_hi),
        cross_same_hi
    );
    const int64_t cross_interaction_quant = current_fm_reduce_epi32(cross_lo) + current_fm_reduce_epi32(cross_hi);
    const int64_t same_interaction_quant = current_fm_reduce_epi32(same_interaction_lo) + current_fm_reduce_epi32(same_interaction_hi);
    const int64_t count_interaction_quant =
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_lo, count_vec_lo)) +
        current_fm_reduce_epi32(_mm256_mullo_epi32(count_pattern_sum_hi, count_vec_hi));

    const double cross_scale = current_fm_cross_file.vector_scale[phase_idx];
    const double same_scale = current_fm_same_file.vector_scale[phase_idx];
    const double count_scale = current_fm_count_file.vector_scale[phase_idx];
    const double score = (double)linear_quant_sum * linear_scale
        + current_fm_cross_weight * (0.5 * (double)cross_interaction_quant * cross_scale * cross_scale)
        + current_fm_same_weight * (0.5 * (double)same_interaction_quant * same_scale * same_scale)
        + current_fm_count_weight * ((double)count_interaction_quant * count_scale * count_scale);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
#else
inline int current_fm_score_from_ids_stream12(const int phase_idx, const int active_ids[], const int n_active) {
    return current_fm_score_from_ids_quant(phase_idx, active_ids, n_active);
}
#endif
