/*
    Egaroucid Project

    @file evaluate_experiment_current_fm_tri_type_count_by_type_stream12_common.hpp
        Streaming dim12 scorer for the isolated current-model + tri-type FM experiment
        with pattern-type-specific stone-count interactions.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once

#define current_fm_count_file current_fm_unused_count_file
#define current_fm_load_egev4 current_fm_load_egev4_unused
#define current_fm_score_from_ids_quant current_fm_score_from_ids_quant_unused
#define current_fm_score_from_ids current_fm_score_from_ids_unused
#include "evaluate_experiment_current_fm_tri_type_simdopt_common.hpp"
#undef current_fm_count_file
#undef current_fm_load_egev4
#undef current_fm_score_from_ids_quant
#undef current_fm_score_from_ids

constexpr int CURRENT_FM_COUNT_TYPE_VERSION = 1;

struct CurrentFmCountTypeFile {
    std::array<float, N_PHASES> vector_scale{};
    std::array<std::vector<int8_t>, N_PHASES> pattern_vector_quant_by_phase;
    std::array<std::vector<int8_t>, N_PHASES> type_vector_quant_by_phase;
    int dim = 0;
};

CurrentFmCountTypeFile current_fm_count_type_file;

inline bool current_fm_load_count_type_egct(const std::string &file) {
    FILE *fp;
    if (!file_open(&fp, file.c_str(), "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open count-by-type FM eval " << file << std::endl;
        return false;
    }

    unsigned char fixed_header[42];
    if (!current_fm_read_exact(fp, fixed_header, 1, sizeof(fixed_header), "EGCT fixed header")) {
        fclose(fp);
        return false;
    }
    if (std::memcmp(fixed_header + 14, "EGCT", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] count-by-type current-FM expects EGCT files" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = current_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = current_fm_read_i32_le(fixed_header + 22);
    const int32_t n_pattern_param = current_fm_read_i32_le(fixed_header + 26);
    const int32_t max_stone_num = current_fm_read_i32_le(fixed_header + 30);
    const int32_t n_pattern_type = current_fm_read_i32_le(fixed_header + 34);
    const int32_t dim = current_fm_read_i32_le(fixed_header + 38);
    if (version != CURRENT_FM_COUNT_TYPE_VERSION ||
        n_phase != N_PHASES ||
        n_pattern_param != CURRENT_FM_N_PATTERN_PARAMS_RAW ||
        max_stone_num != MAX_STONE_NUM ||
        n_pattern_type != CURRENT_FM_N_PATTERN_TYPES ||
        dim <= 0 || dim > CURRENT_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] count-by-type current-FM EGCT header mismatch: version="
                  << version << " phases=" << n_phase << " pattern_params=" << n_pattern_param
                  << " max_stone_num=" << max_stone_num << " pattern_types=" << n_pattern_type
                  << " dim=" << dim << std::endl;
        fclose(fp);
        return false;
    }

    current_fm_count_type_file.dim = dim;
    if (!current_fm_read_exact(
            fp, current_fm_count_type_file.vector_scale.data(), sizeof(float), N_PHASES, "count-by-type scales")) {
        fclose(fp);
        return false;
    }

    const size_t pattern_values_per_phase = (size_t)CURRENT_FM_N_PATTERN_PARAMS_RAW * dim;
    const size_t type_values_per_phase = (size_t)MAX_STONE_NUM * CURRENT_FM_N_PATTERN_TYPES * dim;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        std::vector<int8_t> pattern_vectors(pattern_values_per_phase, 0);
        std::vector<int8_t> type_vectors(type_values_per_phase, 0);
        if (!current_fm_read_exact(fp, pattern_vectors.data(), sizeof(int8_t), pattern_vectors.size(), "count pattern vectors") ||
            !current_fm_read_exact(fp, type_vectors.data(), sizeof(int8_t), type_vectors.size(), "count type vectors")) {
            fclose(fp);
            return false;
        }
        const bool has_pattern_nonzero = std::any_of(
            pattern_vectors.begin(), pattern_vectors.end(), [](int8_t value) { return value != 0; }
        );
        const bool has_type_nonzero = std::any_of(
            type_vectors.begin(), type_vectors.end(), [](int8_t value) { return value != 0; }
        );
        if (has_pattern_nonzero) {
            current_fm_count_type_file.pattern_vector_quant_by_phase[phase] = std::move(pattern_vectors);
        } else {
            current_fm_count_type_file.pattern_vector_quant_by_phase[phase].clear();
        }
        if (has_type_nonzero) {
            current_fm_count_type_file.type_vector_quant_by_phase[phase] = std::move(type_vectors);
        } else {
            current_fm_count_type_file.type_vector_quant_by_phase[phase].clear();
        }
    }

    fclose(fp);
    return true;
}

inline bool current_fm_load_egev4(const char *file, bool show_log) {
    const std::vector<std::string> parts = current_fm_split_spec(file);
    if (parts.size() < 3 || parts[0].empty() || parts[1].empty() || parts[2].empty()) {
        std::cerr << "[ERROR] [FATAL] count-by-type current-FM expects -eval cross.egev4@same.egev4@count.egct[@cross_weight@same_weight@count_weight]" << std::endl;
        return false;
    }
    current_fm_cross_weight = parts.size() >= 4 && !parts[3].empty() ? std::stod(parts[3]) : 0.5;
    current_fm_same_weight = parts.size() >= 5 && !parts[4].empty() ? std::stod(parts[4]) : 0.5;
    current_fm_count_weight = parts.size() >= 6 && !parts[5].empty() ? std::stod(parts[5]) : 1.0;
    if (show_log) {
        std::cerr << "current-model + tri-type count-by-pattern-type FM cross file " << parts[0] << std::endl;
        std::cerr << "current-model + tri-type count-by-pattern-type FM same file " << parts[1] << std::endl;
        std::cerr << "current-model + tri-type count-by-pattern-type FM count file " << parts[2] << std::endl;
        std::cerr << "tri-type count-by-pattern-type FM weights cross=" << current_fm_cross_weight
                  << " same=" << current_fm_same_weight
                  << " count=" << current_fm_count_weight << std::endl;
    }

    if (!current_fm_load_single_egev4(parts[0], true, &current_fm_cross_file)) {
        return false;
    }
    if (!current_fm_load_single_egev4(parts[1], false, &current_fm_same_file)) {
        return false;
    }
    if (!current_fm_load_count_type_egct(parts[2])) {
        return false;
    }
    if (current_fm_cross_file.dim != current_fm_same_file.dim ||
        current_fm_cross_file.dim != current_fm_count_type_file.dim) {
        std::cerr << "[ERROR] [FATAL] count-by-type current-FM dimension mismatch: cross="
                  << current_fm_cross_file.dim << " same=" << current_fm_same_file.dim
                  << " count=" << current_fm_count_type_file.dim << std::endl;
        return false;
    }

    current_fm_loaded = true;
    return true;
}

inline int current_fm_score_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    CurrentFmQuantAccum cross_accum;
    CurrentFmQuantAccum same_accum;
    std::array<std::array<int32_t, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> count_pattern_type_sum{};
    int64_t linear_quant_sum = 0;

    const std::vector<int8_t> &count_pattern_vectors =
        current_fm_count_type_file.pattern_vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_type_vectors =
        current_fm_count_type_file.type_vector_quant_by_phase[phase_idx];
    int stone_count = -1;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        linear_quant_sum += current_fm_cross_file.linear_quant[phase_base + (size_t)id];
        const int pattern_type = current_fm_pattern_type_for_active(i, id);
        if (pattern_type >= 0) {
            current_fm_add_vector_quant<true>(current_fm_cross_file, phase_idx, id, cross_accum, pattern_type);
            current_fm_add_vector_quant<false>(current_fm_same_file, phase_idx, id, same_accum, pattern_type);
            if (!count_pattern_vectors.empty()) {
                const size_t count_idx = (size_t)id * current_fm_count_type_file.dim;
                for (int dim = 0; dim < current_fm_count_type_file.dim; ++dim) {
                    count_pattern_type_sum[pattern_type][dim] += count_pattern_vectors[count_idx + dim];
                }
            }
        } else if (id >= CURRENT_FM_N_PATTERN_PARAMS_RAW) {
            stone_count = id - CURRENT_FM_N_PATTERN_PARAMS_RAW;
        }
    }

    const double linear = (double)linear_quant_sum * linear_scale;
    const double cross_interaction = current_fm_calc_cross_interaction_quant(current_fm_cross_file, phase_idx, cross_accum);
    const double same_interaction = current_fm_calc_same_interaction_quant(current_fm_same_file, phase_idx, same_accum);
    int64_t count_interaction_quant = 0;
    if (!count_type_vectors.empty() && 0 <= stone_count && stone_count < MAX_STONE_NUM) {
        const size_t stone_base = (size_t)stone_count * CURRENT_FM_N_PATTERN_TYPES * current_fm_count_type_file.dim;
        for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
            const size_t type_base = stone_base + (size_t)pattern_type * current_fm_count_type_file.dim;
            for (int dim = 0; dim < current_fm_count_type_file.dim; ++dim) {
                count_interaction_quant +=
                    (int64_t)count_pattern_type_sum[pattern_type][dim] * count_type_vectors[type_base + dim];
            }
        }
    }
    const double count_scale = current_fm_count_type_file.vector_scale[phase_idx];
    const double count_interaction = (double)count_interaction_quant * count_scale * count_scale;
    const double score = linear
        + current_fm_cross_weight * cross_interaction
        + current_fm_same_weight * same_interaction
        + current_fm_count_weight * count_interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

#if USE_SIMD
std::array<std::vector<uint8_t>, N_PHASES> current_fm_stream12_cross_nonzero_mask;
std::array<std::vector<uint8_t>, N_PHASES> current_fm_stream12_same_nonzero_mask;
std::array<std::vector<uint8_t>, N_PHASES> current_fm_stream12_count_pattern_nonzero_mask;
bool current_fm_stream12_sparse_masks_ready = false;

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

inline void current_fm_stream12_make_nonzero_mask(
    const std::vector<int8_t> &vectors,
    const int n_params,
    std::vector<uint8_t> &dst
) {
    if (vectors.empty()) {
        dst.clear();
        return;
    }
    dst.assign((size_t)n_params, 0);
    for (int param = 0; param < n_params; ++param) {
        const size_t base = (size_t)param * 12;
        uint8_t nonzero = 0;
        for (int d = 0; d < 12; ++d) {
            nonzero = (uint8_t)(nonzero | (vectors[base + (size_t)d] != 0));
        }
        dst[(size_t)param] = nonzero;
    }
}

inline bool current_fm_stream12_prepare_sparse_masks() {
    current_fm_stream12_sparse_masks_ready = false;
    if (!current_fm_loaded ||
        current_fm_cross_file.dim != 12 ||
        current_fm_same_file.dim != 12 ||
        current_fm_count_type_file.dim != 12) {
        return true;
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        current_fm_stream12_make_nonzero_mask(
            current_fm_cross_file.vector_quant_by_phase[phase],
            CURRENT_FM_N_PARAMS_PER_PHASE,
            current_fm_stream12_cross_nonzero_mask[phase]
        );
        current_fm_stream12_make_nonzero_mask(
            current_fm_same_file.vector_quant_by_phase[phase],
            CURRENT_FM_N_PARAMS_PER_PHASE,
            current_fm_stream12_same_nonzero_mask[phase]
        );
        current_fm_stream12_make_nonzero_mask(
            current_fm_count_type_file.pattern_vector_quant_by_phase[phase],
            CURRENT_FM_N_PATTERN_PARAMS_RAW,
            current_fm_stream12_count_pattern_nonzero_mask[phase]
        );
    }
    current_fm_stream12_sparse_masks_ready = true;
    return true;
}

inline int current_fm_score_from_ids_stream12_sparse_unchecked(const int phase_idx, const int active_ids[]) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx ||
        current_fm_cross_file.dim != 12 ||
        current_fm_same_file.dim != 12 ||
        current_fm_count_type_file.dim != 12) {
        return current_fm_score_from_ids_quant(phase_idx, active_ids, CURRENT_FM_N_PATTERN_FEATURES + 1);
    }
    if (!current_fm_stream12_sparse_masks_ready) {
        current_fm_stream12_prepare_sparse_masks();
    }

    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const int16_t *linear_ptr = current_fm_cross_file.linear_quant.data() + phase_base;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    const __m256i zero = _mm256_setzero_si256();
    const std::vector<int8_t> &cross_phase_vectors = current_fm_cross_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &same_phase_vectors = current_fm_same_file.vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_pattern_vectors =
        current_fm_count_type_file.pattern_vector_quant_by_phase[phase_idx];
    const std::vector<int8_t> &count_type_vectors =
        current_fm_count_type_file.type_vector_quant_by_phase[phase_idx];
    const int8_t *cross_ptr = cross_phase_vectors.empty() ? nullptr : cross_phase_vectors.data();
    const int8_t *same_ptr = same_phase_vectors.empty() ? nullptr : same_phase_vectors.data();
    const int8_t *count_pattern_ptr = count_pattern_vectors.empty() ? nullptr : count_pattern_vectors.data();
    const int8_t *count_type_ptr = count_type_vectors.empty() ? nullptr : count_type_vectors.data();
    const uint8_t *cross_mask = current_fm_stream12_cross_nonzero_mask[phase_idx].empty()
        ? nullptr
        : current_fm_stream12_cross_nonzero_mask[phase_idx].data();
    const uint8_t *same_mask = current_fm_stream12_same_nonzero_mask[phase_idx].empty()
        ? nullptr
        : current_fm_stream12_same_nonzero_mask[phase_idx].data();
    const uint8_t *count_pattern_mask = current_fm_stream12_count_pattern_nonzero_mask[phase_idx].empty()
        ? nullptr
        : current_fm_stream12_count_pattern_nonzero_mask[phase_idx].data();

    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> count_pattern_sum_lo;
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> count_pattern_sum_hi;
    for (int i = 0; i < CURRENT_FM_N_PATTERN_TYPES; ++i) {
        count_pattern_sum_lo[i] = zero;
        count_pattern_sum_hi[i] = zero;
    }
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < CURRENT_FM_N_PATTERN_FEATURES; ++i) {
        const int id = active_ids[i];
        linear_quant_sum += linear_ptr[id];

        const int pattern_type = i >> 2;
        if (cross_mask != nullptr && cross_mask[id]) {
            current_fm_add_dim12_ptr<true>(cross_ptr, id, cross_accum, pattern_type);
        }
        if (same_mask != nullptr && same_mask[id]) {
            current_fm_add_dim12_ptr<false>(same_ptr, id, same_accum, pattern_type);
        }
        if (count_pattern_mask != nullptr && count_pattern_mask[id]) {
            __m256i count_lo;
            __m256i count_hi;
            current_fm_load_dim12_ptr(count_pattern_ptr, id, count_lo, count_hi);
            count_pattern_sum_lo[pattern_type] = _mm256_add_epi32(count_pattern_sum_lo[pattern_type], count_lo);
            count_pattern_sum_hi[pattern_type] = _mm256_add_epi32(count_pattern_sum_hi[pattern_type], count_hi);
        }
    }

    const int count_id = active_ids[CURRENT_FM_N_PATTERN_FEATURES];
    linear_quant_sum += linear_ptr[count_id];
    const int stone_count = count_id - CURRENT_FM_N_PATTERN_PARAMS_RAW;

    __m256i cross_same_lo = zero;
    __m256i cross_same_hi = zero;
    __m256i same_interaction_lo = zero;
    __m256i same_interaction_hi = zero;
    __m256i count_interaction_lo = zero;
    __m256i count_interaction_hi = zero;
    const bool has_count_type = count_type_ptr != nullptr && 0 <= stone_count && stone_count < MAX_STONE_NUM;
    const size_t count_type_stone_base =
        has_count_type ? (size_t)stone_count * CURRENT_FM_N_PATTERN_TYPES * 12 : 0;
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
        if (has_count_type) {
            __m256i count_vec_lo;
            __m256i count_vec_hi;
            const int8_t *count_type_base =
                count_type_ptr + count_type_stone_base + (size_t)pattern_type * 12;
            current_fm_load_dim12_ptr(count_type_base, 0, count_vec_lo, count_vec_hi);
            count_interaction_lo = _mm256_add_epi32(
                count_interaction_lo,
                _mm256_mullo_epi32(count_pattern_sum_lo[pattern_type], count_vec_lo)
            );
            count_interaction_hi = _mm256_add_epi32(
                count_interaction_hi,
                _mm256_mullo_epi32(count_pattern_sum_hi[pattern_type], count_vec_hi)
            );
        }
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
        current_fm_reduce_epi32(count_interaction_lo) + current_fm_reduce_epi32(count_interaction_hi);

    const double cross_scale = current_fm_cross_file.vector_scale[phase_idx];
    const double same_scale = current_fm_same_file.vector_scale[phase_idx];
    const double count_scale = current_fm_count_type_file.vector_scale[phase_idx];
    const double score = (double)linear_quant_sum * linear_scale
        + current_fm_cross_weight * (0.5 * (double)cross_interaction_quant * cross_scale * cross_scale)
        + current_fm_same_weight * (0.5 * (double)same_interaction_quant * same_scale * same_scale)
        + current_fm_count_weight * ((double)count_interaction_quant * count_scale * count_scale);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline int current_fm_score_from_ids_stream12(const int phase_idx, const int active_ids[], const int n_active) {
    if (n_active != CURRENT_FM_N_PATTERN_FEATURES + 1) {
        return current_fm_score_from_ids_quant(phase_idx, active_ids, n_active);
    }
    return current_fm_score_from_ids_stream12_sparse_unchecked(phase_idx, active_ids);
}
#else
inline bool current_fm_stream12_prepare_sparse_masks() {
    return true;
}

inline int current_fm_score_from_ids_stream12(const int phase_idx, const int active_ids[], const int n_active) {
    return current_fm_score_from_ids_quant(phase_idx, active_ids, n_active);
}
#endif
