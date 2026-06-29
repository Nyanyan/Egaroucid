/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_common.hpp
        Shared loader and scorer for the isolated 7.7 beta + FM experiment
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
std::vector<int8_t> eval77_fm_vector_quant;
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
    const size_t n_vector = n_linear * (size_t)eval77_fm_dim;
    eval77_fm_linear_quant.assign(n_linear, 0);
    eval77_fm_vector_quant.assign(n_vector, 0);

    for (int phase = 0; phase < N_PHASES; ++phase) {
        for (int param = 0; param < EVAL77_FM_N_PARAMS_PER_PHASE; ++param) {
            const size_t linear_idx = (size_t)phase * EVAL77_FM_N_PARAMS_PER_PHASE + param;
            if (!eval77_fm_read_exact(fp, &eval77_fm_linear_quant[linear_idx], sizeof(int16_t), 1, "linear parameter")) {
                fclose(fp);
                return false;
            }
            const size_t vector_idx = linear_idx * (size_t)eval77_fm_dim;
            if (!eval77_fm_read_exact(fp, &eval77_fm_vector_quant[vector_idx], sizeof(int8_t), eval77_fm_dim, "FM parameter")) {
                fclose(fp);
                return false;
            }
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

inline int eval77_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * EVAL77_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = eval77_fm_linear_scale[phase_idx];
    const double vector_scale = eval77_fm_vector_scale[phase_idx];
    std::array<double, EVAL77_FM_MAX_DIM> sum{};
    std::array<double, EVAL77_FM_MAX_DIM> sum_sq{};
    double linear = 0.0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t param_idx = phase_base + (size_t)id;
        linear += (double)eval77_fm_linear_quant[param_idx] * linear_scale;
        const size_t vector_idx = param_idx * (size_t)eval77_fm_dim;
        for (int dim = 0; dim < eval77_fm_dim; ++dim) {
            const double v = (double)eval77_fm_vector_quant[vector_idx + dim] * vector_scale;
            sum[dim] += v;
            sum_sq[dim] += v * v;
        }
    }

    double interaction = 0.0;
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        interaction += sum[dim] * sum[dim] - sum_sq[dim];
    }
    const double score = linear + 0.5 * interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}
