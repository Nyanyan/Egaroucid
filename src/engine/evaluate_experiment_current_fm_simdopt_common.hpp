/*
    Egaroucid Project

    @file evaluate_experiment_current_fm_simdopt_common.hpp
        Shared loader and SIMD-optimized scorer for the isolated current-model + all-pair FM experiment
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

constexpr int current_fm_pattern_offsets[N_PATTERNS] = {
    0, 6561, 26244, 32805,
    52488, 59049, 78732, 80919,
    139968, 199017, 258066, 317115,
    376164, 435213, 494262, 553311
};

std::vector<int16_t> current_fm_linear_quant;
std::array<std::vector<int8_t>, N_PHASES> current_fm_vector_quant_by_phase;
std::array<float, N_PHASES> current_fm_linear_scale;
std::array<float, N_PHASES> current_fm_vector_scale;
int current_fm_dim = 0;
bool current_fm_loaded = false;

inline bool current_fm_read_exact(FILE *fp, void *dst, size_t elem_size, size_t elem_count, const char *what) {
    if (fread(dst, elem_size, elem_count, fp) < elem_count) {
        std::cerr << "[ERROR] [FATAL] current-FM SIMD egev4 file is broken while reading " << what << std::endl;
        return false;
    }
    return true;
}

inline int32_t current_fm_read_i32_le(const unsigned char *p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline bool current_fm_load_egev4(const char *file, bool show_log) {
    if (show_log) {
        std::cerr << "current-model + all-pair FM SIMD experiment evaluation file " << file << std::endl;
    }
    FILE *fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }

    unsigned char fixed_header[34];
    if (!current_fm_read_exact(fp, fixed_header, 1, sizeof(fixed_header), "fixed header")) {
        fclose(fp);
        return false;
    }
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] current-FM SIMD experiment expects an EGEV4-style file" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = current_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = current_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = current_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = current_fm_read_i32_le(fixed_header + 30);
    if (version != 7 && version != 8) {
        std::cerr << "[ERROR] [FATAL] current-FM SIMD experiment supports linear+FM egev4 versions 7 and 8, found version "
                  << version << std::endl;
        fclose(fp);
        return false;
    }
    if (n_phase != N_PHASES || n_param != CURRENT_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > CURRENT_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] current-FM SIMD egev4 header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param << " dim=" << dim << std::endl;
        fclose(fp);
        return false;
    }

    current_fm_dim = dim;
    if (!current_fm_read_exact(fp, current_fm_linear_scale.data(), sizeof(float), N_PHASES, "linear scales") ||
        !current_fm_read_exact(fp, current_fm_vector_scale.data(), sizeof(float), N_PHASES, "FM scales")) {
        fclose(fp);
        return false;
    }

    current_fm_linear_quant.assign((size_t)N_PHASES * CURRENT_FM_N_PARAMS_PER_PHASE, 0);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        std::vector<int8_t> phase_vectors((size_t)CURRENT_FM_N_PARAMS_PER_PHASE * dim, 0);
        bool has_nonzero_vector = false;
        for (int param = 0; param < CURRENT_FM_N_PARAMS_PER_PHASE; ++param) {
            const size_t linear_idx = (size_t)phase * CURRENT_FM_N_PARAMS_PER_PHASE + param;
            if (!current_fm_read_exact(fp, &current_fm_linear_quant[linear_idx], sizeof(int16_t), 1, "linear parameter")) {
                fclose(fp);
                return false;
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
            current_fm_vector_quant_by_phase[phase] = std::move(phase_vectors);
        } else {
            current_fm_vector_quant_by_phase[phase].clear();
        }
    }

    current_fm_loaded = true;
    if (show_log) {
        std::cerr << "current-FM SIMD egev4 loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << current_fm_dim << std::endl;
    }
    fclose(fp);
    return true;
}

inline int current_fm_score_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_linear_scale[phase_idx];
    const double vector_scale = current_fm_vector_scale[phase_idx];
    const std::vector<int8_t> &phase_vectors = current_fm_vector_quant_by_phase[phase_idx];
    std::array<int32_t, CURRENT_FM_MAX_DIM> sum{};
    std::array<int32_t, CURRENT_FM_MAX_DIM> sum_sq{};
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t param_idx = phase_base + (size_t)id;
        linear_quant_sum += current_fm_linear_quant[param_idx];
        if (!phase_vectors.empty()) {
            const size_t vector_idx = (size_t)id * current_fm_dim;
            for (int dim = 0; dim < current_fm_dim; ++dim) {
                const int32_t v = phase_vectors[vector_idx + dim];
                sum[dim] += v;
                sum_sq[dim] += v * v;
            }
        }
    }

    int64_t interaction_quant = 0;
    for (int dim = 0; dim < current_fm_dim; ++dim) {
        const int64_t s = sum[dim];
        interaction_quant += s * s - sum_sq[dim];
    }
    const double score = (double)linear_quant_sum * linear_scale
        + 0.5 * (double)interaction_quant * vector_scale * vector_scale;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

#if USE_SIMD
struct CurrentFmAllPairSimdAccum {
    __m256i sum_lo;
    __m256i sum_hi;
    __m256i sum_sq_lo;
    __m256i sum_sq_hi;
};

inline void current_fm_clear_simd_accum(CurrentFmAllPairSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.sum_lo = zero;
    accum.sum_hi = zero;
    accum.sum_sq_lo = zero;
    accum.sum_sq_hi = zero;
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

inline void current_fm_add_vector_simd16(
    const std::vector<int8_t> &phase_vectors,
    const int id,
    CurrentFmAllPairSimdAccum &accum
) {
    if (phase_vectors.empty()) {
        return;
    }
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 16;
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    const __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16));
    const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1));
    accum.sum_lo = _mm256_add_epi32(accum.sum_lo, lo);
    accum.sum_hi = _mm256_add_epi32(accum.sum_hi, hi);
    accum.sum_sq_lo = _mm256_add_epi32(accum.sum_sq_lo, _mm256_mullo_epi32(lo, lo));
    accum.sum_sq_hi = _mm256_add_epi32(accum.sum_sq_hi, _mm256_mullo_epi32(hi, hi));
}

inline int current_fm_score_from_ids_simd16(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_linear_scale[phase_idx];
    const double vector_scale = current_fm_vector_scale[phase_idx];
    const std::vector<int8_t> &phase_vectors = current_fm_vector_quant_by_phase[phase_idx];
    CurrentFmAllPairSimdAccum accum;
    current_fm_clear_simd_accum(accum);
    int64_t linear_quant_sum = 0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t linear_idx = phase_base + (size_t)id;
        linear_quant_sum += current_fm_linear_quant[linear_idx];
        current_fm_add_vector_simd16(phase_vectors, id, accum);
    }

    const __m256i pair_lo = current_fm_pair_term_epi32(accum.sum_lo, accum.sum_sq_lo);
    const __m256i pair_hi = current_fm_pair_term_epi32(accum.sum_hi, accum.sum_sq_hi);
    const int64_t interaction_quant = current_fm_reduce_epi32(pair_lo) + current_fm_reduce_epi32(pair_hi);
    const double score = (double)linear_quant_sum * linear_scale
        + 0.5 * (double)interaction_quant * vector_scale * vector_scale;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
#endif

inline int current_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
#if USE_SIMD
    if (current_fm_dim == 16) {
        return current_fm_score_from_ids_simd16(phase_idx, active_ids, n_active);
    }
#endif
    return current_fm_score_from_ids_quant(phase_idx, active_ids, n_active);
}
