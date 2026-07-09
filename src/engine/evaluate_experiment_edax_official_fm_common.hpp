/*
    Egaroucid Project

    @file evaluate_experiment_edax_official_fm_common.hpp
        FM vector loader and scorer for the isolated official-Edax + FM experiment
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

constexpr int EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE = 226315;
constexpr int EDAX_OFFICIAL_FM_MAX_DIM = 16;
constexpr int EDAX_OFFICIAL_FM_EGEV4_VERSION = 8;

std::array<std::vector<int8_t>, N_PHASES> edax_official_fm_vector_quant_by_phase;
std::array<float, N_PHASES> edax_official_fm_vector_scale;
int edax_official_fm_dim = 0;
bool edax_official_fm_loaded = false;

inline void edax_official_fm_clear_vectors() {
    for (std::vector<int8_t> &phase_vectors: edax_official_fm_vector_quant_by_phase) {
        phase_vectors.clear();
    }
    edax_official_fm_vector_scale.fill(1.0f);
    edax_official_fm_dim = 0;
    edax_official_fm_loaded = true;
}

inline bool edax_official_fm_read_exact(FILE *fp, void *dst, size_t elem_size, size_t elem_count, const char *what) {
    if (fread(dst, elem_size, elem_count, fp) < elem_count) {
        std::cerr << "[ERROR] [FATAL] Edax official + FM egev4 file is broken while reading " << what << std::endl;
        return false;
    }
    return true;
}

inline int32_t edax_official_fm_read_i32_le(const unsigned char *p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline bool edax_official_fm_load_egev4(const char *file, bool show_log) {
    if (file == nullptr || std::string(file).empty()) {
        edax_official_fm_clear_vectors();
        if (show_log) {
            std::cerr << "Edax official + FM uses zero FM vectors" << std::endl;
        }
        return true;
    }
    if (show_log) {
        std::cerr << "Edax official + FM vector file " << file << std::endl;
    }
    FILE *fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open Edax official + FM vectors " << file << std::endl;
        return false;
    }

    unsigned char fixed_header[34];
    if (!edax_official_fm_read_exact(fp, fixed_header, 1, sizeof(fixed_header), "fixed header")) {
        fclose(fp);
        return false;
    }
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] Edax official + FM expects an EGEV4-style vector file" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = edax_official_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = edax_official_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = edax_official_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = edax_official_fm_read_i32_le(fixed_header + 30);
    if (version != EDAX_OFFICIAL_FM_EGEV4_VERSION) {
        std::cerr << "[ERROR] [FATAL] Edax official + FM supports egev4 version "
                  << EDAX_OFFICIAL_FM_EGEV4_VERSION << ", found version " << version << std::endl;
        fclose(fp);
        return false;
    }
    if (n_phase != N_PHASES || n_param != EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > EDAX_OFFICIAL_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] Edax official + FM egev4 header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param << " dim=" << dim << std::endl;
        fclose(fp);
        return false;
    }

    std::array<float, N_PHASES> ignored_linear_scale;
    edax_official_fm_dim = dim;
    if (!edax_official_fm_read_exact(fp, ignored_linear_scale.data(), sizeof(float), N_PHASES, "ignored linear scales") ||
        !edax_official_fm_read_exact(fp, edax_official_fm_vector_scale.data(), sizeof(float), N_PHASES, "FM scales")) {
        fclose(fp);
        return false;
    }

    int16_t ignored_linear = 0;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        std::vector<int8_t> phase_vectors((size_t)EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE * (size_t)dim, 0);
        bool has_nonzero_vector = false;
        for (int param = 0; param < EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE; ++param) {
            if (!edax_official_fm_read_exact(fp, &ignored_linear, sizeof(int16_t), 1, "ignored linear parameter")) {
                fclose(fp);
                return false;
            }
            const size_t vector_idx = (size_t)param * (size_t)dim;
            if (!edax_official_fm_read_exact(fp, &phase_vectors[vector_idx], sizeof(int8_t), (size_t)dim, "FM parameter")) {
                fclose(fp);
                return false;
            }
            for (int d = 0; d < dim; ++d) {
                has_nonzero_vector = has_nonzero_vector || phase_vectors[vector_idx + (size_t)d] != 0;
            }
        }
        if (has_nonzero_vector) {
            edax_official_fm_vector_quant_by_phase[phase] = std::move(phase_vectors);
        } else {
            edax_official_fm_vector_quant_by_phase[phase].clear();
        }
    }

    edax_official_fm_loaded = true;
    if (show_log) {
        std::cerr << "Edax official + FM vectors loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << edax_official_fm_dim << std::endl;
    }
    fclose(fp);
    return true;
}

inline double edax_official_fm_interaction_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!edax_official_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx || edax_official_fm_dim <= 0) {
        return 0.0;
    }
    const std::vector<int8_t> &phase_vectors = edax_official_fm_vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return 0.0;
    }
    std::array<int32_t, EDAX_OFFICIAL_FM_MAX_DIM> sum{};
    std::array<int32_t, EDAX_OFFICIAL_FM_MAX_DIM> sum_sq{};
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t vector_idx = (size_t)id * (size_t)edax_official_fm_dim;
        for (int dim = 0; dim < edax_official_fm_dim; ++dim) {
            const int32_t v = phase_vectors[vector_idx + (size_t)dim];
            sum[dim] += v;
            sum_sq[dim] += v * v;
        }
    }

    int64_t interaction_quant = 0;
    for (int dim = 0; dim < edax_official_fm_dim; ++dim) {
        const int64_t s = sum[dim];
        interaction_quant += s * s - sum_sq[dim];
    }
    const double scale = (double)edax_official_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}

#if USE_SIMD
struct EdaxOfficialFmSimdAccum {
    __m256i sum16;
    __m256i sum_sq_pair32;
};

inline void edax_official_fm_clear_simd_accum(EdaxOfficialFmSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
}

inline __m256i edax_official_fm_pair_term_epi32(const __m256i sum, const __m256i sum_sq) {
    return _mm256_sub_epi32(sum, sum_sq);
}

inline int64_t edax_official_fm_reduce_epi32(const __m256i values) {
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, values);
    int64_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += tmp[i];
    }
    return sum;
}

inline void edax_official_fm_add_vector_simd16(
    const std::vector<int8_t> &phase_vectors,
    const int id,
    EdaxOfficialFmSimdAccum &accum
) {
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 16;
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline double edax_official_fm_interaction_from_ids_simd16(const int phase_idx, const int active_ids[], const int n_active) {
    if (!edax_official_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx || edax_official_fm_dim != 16) {
        return 0.0;
    }
    const std::vector<int8_t> &phase_vectors = edax_official_fm_vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return 0.0;
    }
    EdaxOfficialFmSimdAccum accum;
    edax_official_fm_clear_simd_accum(accum);
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EDAX_OFFICIAL_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        edax_official_fm_add_vector_simd16(phase_vectors, id, accum);
    }

    const __m256i sum_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = edax_official_fm_pair_term_epi32(sum_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = edax_official_fm_reduce_epi32(pair_term);
    const double scale = (double)edax_official_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}
#endif

inline double edax_official_fm_interaction_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
#if USE_SIMD
    if (edax_official_fm_dim == 16) {
        return edax_official_fm_interaction_from_ids_simd16(phase_idx, active_ids, n_active);
    }
#endif
    return edax_official_fm_interaction_from_ids_quant(phase_idx, active_ids, n_active);
}

inline int edax_official_fm_round_score(const int linear_raw, const double fm_score) {
    const double score = (double)linear_raw / 128.0 + fm_score;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
