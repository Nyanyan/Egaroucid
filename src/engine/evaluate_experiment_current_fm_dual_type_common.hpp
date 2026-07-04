/*
    Egaroucid Project

    @file evaluate_experiment_current_fm_dual_type_common.hpp
        Shared loader and scorer for the isolated current-model + dual-type FM experiment
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

struct CurrentFmDualFile {
    std::vector<int16_t> linear_quant;
    std::array<float, N_PHASES> linear_scale{};
    std::array<float, N_PHASES> vector_scale{};
    std::array<std::vector<int8_t>, N_PHASES> vector_quant_by_phase;
    int dim = 0;
};

CurrentFmDualFile current_fm_cross_file;
CurrentFmDualFile current_fm_same_file;
double current_fm_cross_weight = 0.5;
double current_fm_same_weight = 0.5;
bool current_fm_loaded = false;

inline bool current_fm_read_exact(FILE *fp, void *dst, size_t elem_size, size_t elem_count, const char *what) {
    if (fread(dst, elem_size, elem_count, fp) < elem_count) {
        std::cerr << "[ERROR] [FATAL] dual-type current-FM egev4 file is broken while reading " << what << std::endl;
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

inline bool current_fm_load_single_egev4(const std::string &file, bool store_linear, CurrentFmDualFile *dst) {
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
        std::cerr << "[ERROR] [FATAL] dual-type current-FM expects EGEV4-style files" << std::endl;
        fclose(fp);
        return false;
    }

    const int32_t version = current_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = current_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = current_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = current_fm_read_i32_le(fixed_header + 30);
    if (version != 7 && version != 8) {
        std::cerr << "[ERROR] [FATAL] dual-type current-FM supports linear+FM egev4 versions 7 and 8, found version "
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
    if (parts.size() < 2 || parts[0].empty() || parts[1].empty()) {
        std::cerr << "[ERROR] [FATAL] dual-type current-FM expects -eval cross.egev4@same.egev4[@cross_weight@same_weight]" << std::endl;
        return false;
    }
    current_fm_cross_weight = parts.size() >= 3 && !parts[2].empty() ? std::stod(parts[2]) : 0.5;
    current_fm_same_weight = parts.size() >= 4 && !parts[3].empty() ? std::stod(parts[3]) : 0.5;
    if (show_log) {
        std::cerr << "current-model + dual-type FM experiment cross file " << parts[0] << std::endl;
        std::cerr << "current-model + dual-type FM experiment same file " << parts[1] << std::endl;
        std::cerr << "dual-type FM weights cross=" << current_fm_cross_weight
                  << " same=" << current_fm_same_weight << std::endl;
    }

    if (!current_fm_load_single_egev4(parts[0], true, &current_fm_cross_file)) {
        return false;
    }
    if (!current_fm_load_single_egev4(parts[1], false, &current_fm_same_file)) {
        return false;
    }
    current_fm_loaded = true;
    return true;
}

inline void current_fm_add_vector(
    const CurrentFmDualFile &src,
    const int phase_idx,
    const int id,
    std::array<double, CURRENT_FM_MAX_DIM> &sum,
    std::array<double, CURRENT_FM_MAX_DIM> &sum_sq,
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum,
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum_sq,
    const int pattern_type
) {
    const std::vector<int8_t> &phase_vectors = src.vector_quant_by_phase[phase_idx];
    if (phase_vectors.empty()) {
        return;
    }
    const double vector_scale = src.vector_scale[phase_idx];
    const size_t vector_idx = (size_t)id * src.dim;
    for (int dim = 0; dim < src.dim; ++dim) {
        const double v = (double)phase_vectors[vector_idx + dim] * vector_scale;
        sum[dim] += v;
        sum_sq[dim] += v * v;
        type_sum[pattern_type][dim] += v;
        type_sum_sq[pattern_type][dim] += v * v;
    }
}

inline double current_fm_calc_cross_interaction(
    const CurrentFmDualFile &src,
    const std::array<double, CURRENT_FM_MAX_DIM> &sum,
    const std::array<double, CURRENT_FM_MAX_DIM> &sum_sq,
    const std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum,
    const std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum_sq
) {
    double interaction = 0.0;
    for (int dim = 0; dim < src.dim; ++dim) {
        double same_type_interaction = 0.0;
        for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
            same_type_interaction += type_sum[pattern_type][dim] * type_sum[pattern_type][dim] - type_sum_sq[pattern_type][dim];
        }
        interaction += sum[dim] * sum[dim] - sum_sq[dim] - same_type_interaction;
    }
    return 0.5 * interaction;
}

inline double current_fm_calc_same_interaction(
    const CurrentFmDualFile &src,
    const std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum,
    const std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> &type_sum_sq
) {
    double interaction = 0.0;
    for (int dim = 0; dim < src.dim; ++dim) {
        for (int pattern_type = 0; pattern_type < CURRENT_FM_N_PATTERN_TYPES; ++pattern_type) {
            interaction += type_sum[pattern_type][dim] * type_sum[pattern_type][dim] - type_sum_sq[pattern_type][dim];
        }
    }
    return 0.5 * interaction;
}

inline int current_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    std::array<double, CURRENT_FM_MAX_DIM> cross_sum{};
    std::array<double, CURRENT_FM_MAX_DIM> cross_sum_sq{};
    std::array<double, CURRENT_FM_MAX_DIM> same_sum{};
    std::array<double, CURRENT_FM_MAX_DIM> same_sum_sq{};
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> cross_type_sum{};
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> cross_type_sum_sq{};
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> same_type_sum{};
    std::array<std::array<double, CURRENT_FM_MAX_DIM>, CURRENT_FM_N_PATTERN_TYPES> same_type_sum_sq{};
    double linear = 0.0;

    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const size_t linear_idx = phase_base + (size_t)id;
        linear += (double)current_fm_cross_file.linear_quant[linear_idx] * linear_scale;
        const int pattern_type = (i < CURRENT_FM_N_PATTERN_FEATURES && id < CURRENT_FM_N_PATTERN_PARAMS_RAW) ? i / 4 : -1;
        if (pattern_type >= 0) {
            current_fm_add_vector(current_fm_cross_file, phase_idx, id, cross_sum, cross_sum_sq, cross_type_sum, cross_type_sum_sq, pattern_type);
            current_fm_add_vector(current_fm_same_file, phase_idx, id, same_sum, same_sum_sq, same_type_sum, same_type_sum_sq, pattern_type);
        }
    }

    const double cross_interaction = current_fm_calc_cross_interaction(current_fm_cross_file, cross_sum, cross_sum_sq, cross_type_sum, cross_type_sum_sq);
    const double same_interaction = current_fm_calc_same_interaction(current_fm_same_file, same_type_sum, same_type_sum_sq);
    const double score = linear + current_fm_cross_weight * cross_interaction + current_fm_same_weight * same_interaction;
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

