/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_simdopt_common.hpp
        Shared loader and optimized scorer for the isolated 7.7 beta + FM experiment
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

constexpr int EVAL77_FM_N_PATTERN_PARAMS_RAW = 944784;
constexpr int EVAL77_FM_N_PARAMS_PER_PHASE = EVAL77_FM_N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int EVAL77_FM_MAX_DIM = 16;

constexpr int eval77_fm_pattern_offsets[N_PATTERNS] = {
    0, 59049, 118098, 177147,
    236196, 295245, 354294, 413343,
    472392, 531441, 590490, 649539,
    708588, 767637, 826686, 885735
};

std::vector<int16_t> eval77_fm_linear_quant;
std::array<std::vector<int8_t>, N_PHASES> eval77_fm_vector_quant_by_phase;
std::array<float, N_PHASES> eval77_fm_linear_scale;
std::array<float, N_PHASES> eval77_fm_vector_scale;
int eval77_fm_dim = 0;
bool eval77_fm_loaded = false;

inline bool eval77_fm_read_exact(FILE *fp, void *dst, size_t elem_size, size_t elem_count, const char *what) {
    if (fread(dst, elem_size, elem_count, fp) < elem_count) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM egev4 file is broken while reading " << what << std::endl;
        return false;
    }
    return true;
}

inline int32_t eval77_fm_read_i32_le(const unsigned char *p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline bool eval77_fm_load_egev4(const char *file, bool show_log) {
    if (show_log) {
        std::cerr << "7.7 beta + FM experiment evaluation file " << file << std::endl;
    }
    FILE *fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }

    unsigned char fixed_header[34];
    if (!eval77_fm_read_exact(fp, fixed_header, 1, sizeof(fixed_header), "fixed header")) {
        fclose(fp);
        return false;
    }
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM experiment expects an EGEV4-style file" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = eval77_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = eval77_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = eval77_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = eval77_fm_read_i32_le(fixed_header + 30);
    if (version != 7 && version != 8) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM experiment supports linear+FM egev4 versions 7 and 8, found version "
                  << version << std::endl;
        fclose(fp);
        return false;
    }
    if (n_phase != N_PHASES || n_param != EVAL77_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > EVAL77_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM egev4 header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param << " dim=" << dim << std::endl;
        fclose(fp);
        return false;
    }

    eval77_fm_dim = dim;
    if (!eval77_fm_read_exact(fp, eval77_fm_linear_scale.data(), sizeof(float), N_PHASES, "linear scales") ||
        !eval77_fm_read_exact(fp, eval77_fm_vector_scale.data(), sizeof(float), N_PHASES, "FM scales")) {
        fclose(fp);
        return false;
    }

    const size_t n_linear = (size_t)N_PHASES * EVAL77_FM_N_PARAMS_PER_PHASE;
    eval77_fm_linear_quant.assign(n_linear, 0);
    for (std::vector<int8_t> &phase_vectors: eval77_fm_vector_quant_by_phase) {
        phase_vectors.clear();
    }

    for (int phase = 0; phase < N_PHASES; ++phase) {
        std::vector<int8_t> phase_vectors((size_t)EVAL77_FM_N_PARAMS_PER_PHASE * (size_t)eval77_fm_dim, 0);
        bool has_nonzero_vector = false;
        for (int param = 0; param < EVAL77_FM_N_PARAMS_PER_PHASE; ++param) {
            const size_t linear_idx = (size_t)phase * EVAL77_FM_N_PARAMS_PER_PHASE + param;
            if (!eval77_fm_read_exact(fp, &eval77_fm_linear_quant[linear_idx], sizeof(int16_t), 1, "linear parameter")) {
                fclose(fp);
                return false;
            }
            const size_t vector_idx = (size_t)param * (size_t)eval77_fm_dim;
            if (!eval77_fm_read_exact(fp, &phase_vectors[vector_idx], sizeof(int8_t), eval77_fm_dim, "FM parameter")) {
                fclose(fp);
                return false;
            }
            for (int d = 0; d < eval77_fm_dim; ++d) {
                has_nonzero_vector = has_nonzero_vector || phase_vectors[vector_idx + (size_t)d] != 0;
            }
        }
        if (has_nonzero_vector) {
            eval77_fm_vector_quant_by_phase[phase] = std::move(phase_vectors);
        } else {
            eval77_fm_vector_quant_by_phase[phase].clear();
        }
    }

    eval77_fm_loaded = true;
    if (show_log) {
        std::cerr << "7.7-FM egev4 loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << eval77_fm_dim << std::endl;
    }
    fclose(fp);
    return true;
}

inline double eval77_fm_linear_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0.0;
    }
    const size_t phase_base = (size_t)phase_idx * EVAL77_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = eval77_fm_linear_scale[phase_idx];
    double linear = 0.0;
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t param_idx = phase_base + (size_t)id;
        linear += (double)eval77_fm_linear_quant[param_idx] * linear_scale;
    }
    return linear;
}

inline double eval77_fm_interaction_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx || eval77_fm_dim <= 0) {
        return 0.0;
    }
    const std::vector<int8_t> &phase_vectors = eval77_fm_vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return 0.0;
    }

    std::array<int32_t, EVAL77_FM_MAX_DIM> sum{};
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum_sq{};
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t vector_idx = (size_t)id * (size_t)eval77_fm_dim;
        for (int dim = 0; dim < eval77_fm_dim; ++dim) {
            const int32_t v = phase_vectors[vector_idx + (size_t)dim];
            sum[dim] += v;
            sum_sq[dim] += v * v;
        }
    }

    int64_t interaction_quant = 0;
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int64_t s = sum[dim];
        interaction_quant += s * s - sum_sq[dim];
    }
    const double scale = (double)eval77_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}

#if USE_SIMD
struct Eval77FmSimdAccum {
    __m256i sum16;
    __m256i sum_sq_pair32;
};

inline void eval77_fm_clear_simd_accum(Eval77FmSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
}

inline int64_t eval77_fm_reduce_epi32(const __m256i values) {
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, values);
    int64_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += tmp[i];
    }
    return sum;
}

inline void eval77_fm_add_vector_simd16(
    const std::vector<int8_t> &phase_vectors,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 16;
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline void eval77_fm_add_vector_simd8(
    const std::vector<int8_t> &phase_vectors,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 8;
    const __m128i q8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline void eval77_fm_add_vector_simd12(
    const std::vector<int8_t> &phase_vectors,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = phase_vectors.data() + (size_t)id * 12;
    const __m128i low8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    int32_t tail4 = 0;
    std::memcpy(&tail4, src_ptr + 8, sizeof(tail4));
    const __m128i high4 = _mm_slli_si128(_mm_cvtsi32_si128(tail4), 8);
    const __m128i q8 = _mm_or_si128(low8, high4);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline double eval77_fm_interaction_from_ids_simd_padded16(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx ||
        (eval77_fm_dim != 8 && eval77_fm_dim != 12 && eval77_fm_dim != 16)) {
        return 0.0;
    }
    const std::vector<int8_t> &phase_vectors = eval77_fm_vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return 0.0;
    }

    Eval77FmSimdAccum accum;
    eval77_fm_clear_simd_accum(accum);
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        if (eval77_fm_dim == 16) {
            eval77_fm_add_vector_simd16(phase_vectors, id, accum);
        } else if (eval77_fm_dim == 12) {
            eval77_fm_add_vector_simd12(phase_vectors, id, accum);
        } else {
            eval77_fm_add_vector_simd8(phase_vectors, id, accum);
        }
    }

    const __m256i sum_sq_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = _mm256_sub_epi32(sum_sq_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = eval77_fm_reduce_epi32(pair_term);
    const double scale = (double)eval77_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}
#endif

inline double eval77_fm_interaction_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
#if USE_SIMD
    if (eval77_fm_dim == 8 || eval77_fm_dim == 12 || eval77_fm_dim == 16) {
        return eval77_fm_interaction_from_ids_simd_padded16(phase_idx, active_ids, n_active);
    }
#endif
    return eval77_fm_interaction_from_ids_quant(phase_idx, active_ids, n_active);
}

inline int eval77_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    const double score = eval77_fm_linear_from_ids(phase_idx, active_ids, n_active) +
        eval77_fm_interaction_from_ids(phase_idx, active_ids, n_active);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
