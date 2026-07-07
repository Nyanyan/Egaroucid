/*
    Egaroucid Project

    @file evaluate_experiment_current_fm_tri_type_simdopt_common.hpp
        Shared loader and SIMD-optimized scorer for the isolated current-model + tri-type FM experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#if USE_SIMD
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif

constexpr int CURRENT_FM_N_PATTERN_PARAMS_RAW = 612360;
constexpr int CURRENT_FM_N_PARAMS_PER_PHASE = CURRENT_FM_N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int CURRENT_FM_SUPPORTED_HEADER_BYTES = 514;
constexpr int CURRENT_FM_MAX_DIM = 16;
constexpr int CURRENT_FM_N_PATTERN_FEATURES = 64;
constexpr int CURRENT_FM_N_PATTERN_TYPES = 16;

constexpr int current_fm_pattern_offsets[N_PATTERNS] = {
    0, 6561, 26244, 32805,
    52488, 59049, 78732, 80919,
    139968, 199017, 258066, 317115,
    376164, 435213, 494262, 553311
};

struct CurrentFmTriFile {
    std::vector<int16_t> linear_quant;
    std::array<float, N_PHASES> linear_scale{};
    std::array<float, N_PHASES> vector_scale{};
    std::array<std::vector<int8_t>, N_PHASES> vector_quant_by_phase;
    int dim = 0;
};

CurrentFmTriFile current_fm_cross_file;
CurrentFmTriFile current_fm_same_file;
CurrentFmTriFile current_fm_count_file;
double current_fm_cross_weight = 0.5;
double current_fm_same_weight = 0.5;
double current_fm_count_weight = 1.0;
bool current_fm_loaded = false;

inline bool current_fm_read_exact(FILE *fp, void *dst, size_t elem_size, size_t elem_count, const char *what) {
    if (fread(dst, elem_size, elem_count, fp) < elem_count) {
        std::cerr << "[ERROR] [FATAL] tri-type current-FM egev4 file is broken while reading " << what << std::endl;
        return false;
    }
    return true;
}

inline int32_t current_fm_read_i32_le(const unsigned char *p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline std::vector<std::string> current_fm_split_spec(const std::string &text) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t pos = text.find('@', start);
        if (pos == std::string::npos) {
            parts.emplace_back(text.substr(start));
            break;
        }
        parts.emplace_back(text.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

inline bool current_fm_load_single_egev4(const std::string &file, bool store_linear, CurrentFmTriFile *dst) {
    FILE *fp;
    if (!file_open(&fp, file.c_str(), "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }

    unsigned char fixed_header[34];
    if (!current_fm_read_exact(fp, fixed_header, 1, sizeof(fixed_header), "fixed header")) {
        fclose(fp);
        return false;
    }
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] tri-type current-FM expects EGEV4-style files" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = current_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = current_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = current_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = current_fm_read_i32_le(fixed_header + 30);
    if (version != 7 && version != 8) {
        std::cerr << "[ERROR] [FATAL] tri-type current-FM supports linear+FM egev4 versions 7 and 8, found version "
                  << version << std::endl;
        fclose(fp);
        return false;
    }
    if (n_phase != N_PHASES || n_param != CURRENT_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > CURRENT_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] current-FM egev4 header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param << " dim=" << dim << std::endl;
        fclose(fp);
        return false;
    }

    dst->dim = dim;
    if (!current_fm_read_exact(fp, dst->linear_scale.data(), sizeof(float), N_PHASES, "linear scales") ||
        !current_fm_read_exact(fp, dst->vector_scale.data(), sizeof(float), N_PHASES, "FM scales")) {
        fclose(fp);
        return false;
    }

    if (store_linear) {
        dst->linear_quant.assign((size_t)N_PHASES * CURRENT_FM_N_PARAMS_PER_PHASE, 0);
    } else {
        dst->linear_quant.clear();
    }

    for (int phase = 0; phase < N_PHASES; ++phase) {
        std::vector<int8_t> phase_vectors((size_t)CURRENT_FM_N_PARAMS_PER_PHASE * dim, 0);
        bool has_nonzero_vector = false;
        for (int param = 0; param < CURRENT_FM_N_PARAMS_PER_PHASE; ++param) {
            int16_t linear_value = 0;
            if (!current_fm_read_exact(fp, &linear_value, sizeof(int16_t), 1, "linear parameter")) {
                fclose(fp);
                return false;
            }
            if (store_linear) {
                dst->linear_quant[(size_t)phase * CURRENT_FM_N_PARAMS_PER_PHASE + param] = linear_value;
            }
            const size_t vector_idx = (size_t)param * dim;
            if (!current_fm_read_exact(fp, &phase_vectors[vector_idx], sizeof(int8_t), dim, "FM parameter")) {
                fclose(fp);
                return false;
            }
            for (int d = 0; d < dim; ++d) {
                has_nonzero_vector = has_nonzero_vector || phase_vectors[vector_idx + d] != 0;
            }
        }
        if (has_nonzero_vector) {
            dst->vector_quant_by_phase[phase] = std::move(phase_vectors);
        } else {
            dst->vector_quant_by_phase[phase].clear();
        }
    }

    fclose(fp);
    return true;
}

inline bool current_fm_load_egev4(const char *file, bool show_log) {
    const std::vector<std::string> parts = current_fm_split_spec(file);
    if (parts.size() < 3 || parts[0].empty() || parts[1].empty() || parts[2].empty()) {
        std::cerr << "[ERROR] [FATAL] tri-type current-FM expects -eval cross.egev4@same.egev4@count.egev4[@cross_weight@same_weight@count_weight]" << std::endl;
        return false;
    }
    current_fm_cross_weight = parts.size() >= 4 && !parts[3].empty() ? std::stod(parts[3]) : 0.5;
    current_fm_same_weight = parts.size() >= 5 && !parts[4].empty() ? std::stod(parts[4]) : 0.5;
    current_fm_count_weight = parts.size() >= 6 && !parts[5].empty() ? std::stod(parts[5]) : 1.0;
    if (show_log) {
        std::cerr << "current-model + tri-type FM experiment cross file " << parts[0] << std::endl;
        std::cerr << "current-model + tri-type FM experiment same file " << parts[1] << std::endl;
        std::cerr << "current-model + tri-type FM experiment count file " << parts[2] << std::endl;
        std::cerr << "tri-type FM weights cross=" << current_fm_cross_weight
                  << " same=" << current_fm_same_weight
                  << " count=" << current_fm_count_weight << std::endl;
    }

    if (!current_fm_load_single_egev4(parts[0], true, &current_fm_cross_file)) {
        return false;
    }
    if (!current_fm_load_single_egev4(parts[1], false, &current_fm_same_file)) {
        return false;
    }
    if (!current_fm_load_single_egev4(parts[2], false, &current_fm_count_file)) {
        return false;
    }
    current_fm_loaded = true;
    return true;
}

struct CurrentFmQuantAccum {
    std::array<int32_t, CURRENT_FM_MAX_DIM> sum{};
    std::array<int32_t, CURRENT_FM_MAX_DIM> sum_sq{};
    std::array<std::array<int32_t, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> type_sum{};
    std::array<std::array<int32_t, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> type_sum_sq{};
};

inline int current_fm_pattern_type_for_active(const int active_pos, const int id) {
    return (active_pos < CURRENT_FM_N_PATTERN_FEATURES && id < CURRENT_FM_N_PATTERN_PARAMS_RAW) ? active_pos / 4 : -1;
}

template<bool UpdateTotal>
inline void current_fm_add_vector_quant(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const int id,
    CurrentFmQuantAccum &accum,
    const int pattern_type
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return;
    }
    const size_t vector_idx = (size_t)id * src.dim;
    for (int dim = 0; dim < src.dim; ++dim) {
        const int32_t v = phase_vectors[vector_idx + dim];
        const int32_t vv = v * v;
        if constexpr (UpdateTotal) {
            accum.sum[dim] += v;
            accum.sum_sq[dim] += vv;
        }
        accum.type_sum[pattern_type][dim] += v;
        accum.type_sum_sq[pattern_type][dim] += vv;
    }
}

inline double current_fm_calc_cross_interaction_quant(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const CurrentFmQuantAccum &accum
) {
    int64_t interaction = 0;
    for (int dim = 0; dim < src.dim; ++dim) {
        int64_t same_type_interaction = 0;
        for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
            const int64_t type_sum = accum.type_sum[pattern_type][dim];
            same_type_interaction += type_sum * type_sum - accum.type_sum_sq[pattern_type][dim];
        }
        const int64_t sum = accum.sum[dim];
        interaction += sum * sum - accum.sum_sq[dim] - same_type_interaction;
    }
    const double vector_scale = src.vector_scale[phase_idx];
    return 0.5 * (double)interaction * vector_scale * vector_scale;
}

inline double current_fm_calc_same_interaction_quant(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const CurrentFmQuantAccum &accum
) {
    int64_t interaction = 0;
    for (int dim = 0; dim < src.dim; ++dim) {
        for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
            const int64_t type_sum = accum.type_sum[pattern_type][dim];
            interaction += type_sum * type_sum - accum.type_sum_sq[pattern_type][dim];
        }
    }
    const double vector_scale = src.vector_scale[phase_idx];
    return 0.5 * (double)interaction * vector_scale * vector_scale;
}

inline int current_fm_score_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    CurrentFmQuantAccum cross_accum;
    CurrentFmQuantAccum same_accum;
    std::array<int32_t, CURRENT_FM_MAX_DIM> count_pattern_sum{};
    std::array<int32_t, CURRENT_FM_MAX_DIM> count_vec{};
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t linear_idx = phase_base + (size_t)id;
        linear_quant_sum += current_fm_cross_file.linear_quant[linear_idx];
        const int pattern_type = current_fm_pattern_type_for_active(i, id);
        if (pattern_type >= 0) {
            current_fm_add_vector_quant<true>(current_fm_cross_file, phase_idx, id, cross_accum, pattern_type);
            current_fm_add_vector_quant<false>(current_fm_same_file, phase_idx, id, same_accum, pattern_type);
            const std::vector<int8_t> &count_vectors = current_fm_count_file.vector_quant_by_phase[phase_idx];
            if (!count_vectors.empty()) {
                const size_t count_idx = (size_t)id * current_fm_count_file.dim;
                for (int dim = 0; dim < current_fm_count_file.dim; ++dim) {
                    count_pattern_sum[dim] += count_vectors[count_idx + dim];
                }
            }
        } else if (id >= CURRENT_FM_N_PATTERN_PARAMS_RAW) {
            const std::vector<int8_t> &count_vectors = current_fm_count_file.vector_quant_by_phase[phase_idx];
            if (!count_vectors.empty()) {
                const size_t count_idx = (size_t)id * current_fm_count_file.dim;
                for (int dim = 0; dim < current_fm_count_file.dim; ++dim) {
                    count_vec[dim] = count_vectors[count_idx + dim];
                }
            }
        }
    }

    const double linear = (double)linear_quant_sum * linear_scale;
    const double cross_interaction = current_fm_calc_cross_interaction_quant(current_fm_cross_file, phase_idx, cross_accum);
    const double same_interaction = current_fm_calc_same_interaction_quant(current_fm_same_file, phase_idx, same_accum);
    int64_t count_interaction_quant = 0;
    for (int dim = 0; dim < current_fm_count_file.dim; ++dim) {
        count_interaction_quant += (int64_t)count_pattern_sum[dim] * count_vec[dim];
    }
    const double count_scale = current_fm_count_file.vector_scale[phase_idx];
    const double count_interaction = (double)count_interaction_quant * count_scale * count_scale;
    const double score = linear
        + current_fm_cross_weight * cross_interaction
        + current_fm_same_weight * same_interaction
        + current_fm_count_weight * count_interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

#if USE_SIMD
struct CurrentFmSimdAccum {
    __m256i sum_lo;
    __m256i sum_hi;
    __m256i sum_sq_lo;
    __m256i sum_sq_hi;
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> type_sum_lo;
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> type_sum_hi;
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> type_sum_sq_lo;
    std::array<__m256i, CURRENT_FM_N_PATTERN_TYPES> type_sum_sq_hi;
};

inline void current_fm_clear_simd_accum(CurrentFmSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.sum_lo = zero;
    accum.sum_hi = zero;
    accum.sum_sq_lo = zero;
    accum.sum_sq_hi = zero;
    for (int i = 0; i < CURRENT_FM_N_PATTERN_TYPES; ++i) {
        accum.type_sum_lo[i] = zero;
        accum.type_sum_hi[i] = zero;
        accum.type_sum_sq_lo[i] = zero;
        accum.type_sum_sq_hi[i] = zero;
    }
}

inline __m256i current_fm_pair_term_epi32(const __m256i sum, const __m256i sum_sq) {
    return _mm256_sub_epi32(_mm256_mullo_epi32(sum, sum), sum_sq);
}

inline int64_t current_fm_reduce_epi32(const __m256i values) {
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, values);
    int64_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += tmp[i];
    }
    return sum;
}

template<bool UpdateTotal>
inline void current_fm_add_vector_simd16(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const int id,
    CurrentFmSimdAccum &accum,
    const int pattern_type
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return;
    }
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 16;
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    const __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16));
    const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1));
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

template<bool UpdateTotal>
inline void current_fm_add_vector_simd8(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const int id,
    CurrentFmSimdAccum &accum,
    const int pattern_type
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return;
    }
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 8;
    const __m128i q8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    const __m256i lo = _mm256_cvtepi8_epi32(q8);
    const __m256i lo_sq = _mm256_mullo_epi32(lo, lo);
    if constexpr (UpdateTotal) {
        accum.sum_lo = _mm256_add_epi32(accum.sum_lo, lo);
        accum.sum_sq_lo = _mm256_add_epi32(accum.sum_sq_lo, lo_sq);
    }
    accum.type_sum_lo[pattern_type] = _mm256_add_epi32(accum.type_sum_lo[pattern_type], lo);
    accum.type_sum_sq_lo[pattern_type] = _mm256_add_epi32(accum.type_sum_sq_lo[pattern_type], lo_sq);
}

inline double current_fm_calc_cross_interaction_simd16(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const CurrentFmSimdAccum &accum
) {
    __m256i same_lo = _mm256_setzero_si256();
    __m256i same_hi = _mm256_setzero_si256();
    for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
        same_lo = _mm256_add_epi32(same_lo, current_fm_pair_term_epi32(accum.type_sum_lo[pattern_type], accum.type_sum_sq_lo[pattern_type]));
        same_hi = _mm256_add_epi32(same_hi, current_fm_pair_term_epi32(accum.type_sum_hi[pattern_type], accum.type_sum_sq_hi[pattern_type]));
    }
    const __m256i cross_lo = _mm256_sub_epi32(current_fm_pair_term_epi32(accum.sum_lo, accum.sum_sq_lo), same_lo);
    const __m256i cross_hi = _mm256_sub_epi32(current_fm_pair_term_epi32(accum.sum_hi, accum.sum_sq_hi), same_hi);
    const int64_t interaction = current_fm_reduce_epi32(cross_lo) + current_fm_reduce_epi32(cross_hi);
    const double vector_scale = src.vector_scale[phase_idx];
    return 0.5 * (double)interaction * vector_scale * vector_scale;
}

inline double current_fm_calc_same_interaction_simd16(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const CurrentFmSimdAccum &accum
) {
    __m256i same_lo = _mm256_setzero_si256();
    __m256i same_hi = _mm256_setzero_si256();
    for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
        same_lo = _mm256_add_epi32(same_lo, current_fm_pair_term_epi32(accum.type_sum_lo[pattern_type], accum.type_sum_sq_lo[pattern_type]));
        same_hi = _mm256_add_epi32(same_hi, current_fm_pair_term_epi32(accum.type_sum_hi[pattern_type], accum.type_sum_sq_hi[pattern_type]));
    }
    const int64_t interaction = current_fm_reduce_epi32(same_lo) + current_fm_reduce_epi32(same_hi);
    const double vector_scale = src.vector_scale[phase_idx];
    return 0.5 * (double)interaction * vector_scale * vector_scale;
}

inline void current_fm_load_vector_simd16(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const int id,
    __m256i &lo,
    __m256i &hi
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return;
    }
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 16;
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16));
    hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1));
}

inline void current_fm_load_vector_simd8(
    const CurrentFmTriFile &src,
    const int phase_idx,
    const int id,
    __m256i &lo,
    __m256i &hi
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    hi = _mm256_setzero_si256();
    if (phase_vectors.empty()) {
        return;
    }
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 8;
    const __m128i q8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    lo = _mm256_cvtepi8_epi32(q8);
}

inline double current_fm_calc_count_interaction_simd16(
    const int phase_idx,
    const __m256i pattern_sum_lo,
    const __m256i pattern_sum_hi,
    const __m256i count_vec_lo,
    const __m256i count_vec_hi
) {
    const __m256i pair_lo = _mm256_mullo_epi32(pattern_sum_lo, count_vec_lo);
    const __m256i pair_hi = _mm256_mullo_epi32(pattern_sum_hi, count_vec_hi);
    const int64_t interaction = current_fm_reduce_epi32(pair_lo) + current_fm_reduce_epi32(pair_hi);
    const double vector_scale = current_fm_count_file.vector_scale[phase_idx];
    return (double)interaction * vector_scale * vector_scale;
}

inline int current_fm_score_from_ids_simd16(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    __m256i count_pattern_sum_lo = _mm256_setzero_si256();
    __m256i count_pattern_sum_hi = _mm256_setzero_si256();
    __m256i count_vec_lo = _mm256_setzero_si256();
    __m256i count_vec_hi = _mm256_setzero_si256();
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t linear_idx = phase_base + (size_t)id;
        linear_quant_sum += current_fm_cross_file.linear_quant[linear_idx];
        const int pattern_type = current_fm_pattern_type_for_active(i, id);
        if (pattern_type >= 0) {
            current_fm_add_vector_simd16<true>(current_fm_cross_file, phase_idx, id, cross_accum, pattern_type);
            current_fm_add_vector_simd16<false>(current_fm_same_file, phase_idx, id, same_accum, pattern_type);
            __m256i pattern_lo = _mm256_setzero_si256();
            __m256i pattern_hi = _mm256_setzero_si256();
            current_fm_load_vector_simd16(current_fm_count_file, phase_idx, id, pattern_lo, pattern_hi);
            count_pattern_sum_lo = _mm256_add_epi32(count_pattern_sum_lo, pattern_lo);
            count_pattern_sum_hi = _mm256_add_epi32(count_pattern_sum_hi, pattern_hi);
        } else if (id >= CURRENT_FM_N_PATTERN_PARAMS_RAW) {
            current_fm_load_vector_simd16(current_fm_count_file, phase_idx, id, count_vec_lo, count_vec_hi);
        }
    }

    const double linear = (double)linear_quant_sum * linear_scale;
    const double cross_interaction = current_fm_calc_cross_interaction_simd16(current_fm_cross_file, phase_idx, cross_accum);
    const double same_interaction = current_fm_calc_same_interaction_simd16(current_fm_same_file, phase_idx, same_accum);
    const double count_interaction = current_fm_calc_count_interaction_simd16(
        phase_idx, count_pattern_sum_lo, count_pattern_sum_hi, count_vec_lo, count_vec_hi
    );
    const double score = linear
        + current_fm_cross_weight * cross_interaction
        + current_fm_same_weight * same_interaction
        + current_fm_count_weight * count_interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline int current_fm_score_from_ids_simd8(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    CurrentFmSimdAccum cross_accum;
    CurrentFmSimdAccum same_accum;
    current_fm_clear_simd_accum(cross_accum);
    current_fm_clear_simd_accum(same_accum);
    __m256i count_pattern_sum_lo = _mm256_setzero_si256();
    __m256i count_pattern_sum_hi = _mm256_setzero_si256();
    __m256i count_vec_lo = _mm256_setzero_si256();
    __m256i count_vec_hi = _mm256_setzero_si256();
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t linear_idx = phase_base + (size_t)id;
        linear_quant_sum += current_fm_cross_file.linear_quant[linear_idx];
        const int pattern_type = current_fm_pattern_type_for_active(i, id);
        if (pattern_type >= 0) {
            current_fm_add_vector_simd8<true>(current_fm_cross_file, phase_idx, id, cross_accum, pattern_type);
            current_fm_add_vector_simd8<false>(current_fm_same_file, phase_idx, id, same_accum, pattern_type);
            __m256i pattern_lo = _mm256_setzero_si256();
            __m256i pattern_hi = _mm256_setzero_si256();
            current_fm_load_vector_simd8(current_fm_count_file, phase_idx, id, pattern_lo, pattern_hi);
            count_pattern_sum_lo = _mm256_add_epi32(count_pattern_sum_lo, pattern_lo);
            count_pattern_sum_hi = _mm256_add_epi32(count_pattern_sum_hi, pattern_hi);
        } else if (id >= CURRENT_FM_N_PATTERN_PARAMS_RAW) {
            current_fm_load_vector_simd8(current_fm_count_file, phase_idx, id, count_vec_lo, count_vec_hi);
        }
    }

    const double linear = (double)linear_quant_sum * linear_scale;
    const double cross_interaction = current_fm_calc_cross_interaction_simd16(current_fm_cross_file, phase_idx, cross_accum);
    const double same_interaction = current_fm_calc_same_interaction_simd16(current_fm_same_file, phase_idx, same_accum);
    const double count_interaction = current_fm_calc_count_interaction_simd16(
        phase_idx, count_pattern_sum_lo, count_pattern_sum_hi, count_vec_lo, count_vec_hi
    );
    const double score = linear
        + current_fm_cross_weight * cross_interaction
        + current_fm_same_weight * same_interaction
        + current_fm_count_weight * count_interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
#endif

inline int current_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
#if USE_SIMD
    if (current_fm_cross_file.dim == 16 && current_fm_same_file.dim == 16 && current_fm_count_file.dim == 16) {
        return current_fm_score_from_ids_simd16(phase_idx, active_ids, n_active);
    }
    if (current_fm_cross_file.dim == 8 && current_fm_same_file.dim == 8 && current_fm_count_file.dim == 8) {
        return current_fm_score_from_ids_simd8(phase_idx, active_ids, n_active);
    }
#endif
    return current_fm_score_from_ids_quant(phase_idx, active_ids, n_active);
}
