/*
    Egaroucid Project

    @file evaluate_experiment_edax_official_impl.hpp
        Isolated Edax official linear evaluation experiment implementation
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
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
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_common.hpp"

constexpr int EDAX_OFFICIAL_N_FEATURES_CEIL = ((N_PATTERN_FEATURES + 15) / 16) * 16;
constexpr int EDAX_OFFICIAL_N_PATTERN_PARAMS_RAW = 226315;
constexpr int EDAX_OFFICIAL_N_PACKED_PARAMS = 114364;
constexpr int EDAX_OFFICIAL_N_PLY = 61;
constexpr int EDAX_OFFICIAL_WEIGHT_STRIDE = EDAX_OFFICIAL_N_PATTERN_PARAMS_RAW;

constexpr Feature_to_coord feature_to_coord[N_PATTERN_FEATURES] = {
    { 9, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C1, COORD_A3, COORD_C2, COORD_B3, COORD_C3, COORD_NO}},
    { 9, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F1, COORD_H3, COORD_F2, COORD_G3, COORD_F3, COORD_NO}},
    { 9, {COORD_A8, COORD_A7, COORD_B8, COORD_B7, COORD_A6, COORD_C8, COORD_B6, COORD_C7, COORD_C6, COORD_NO}},
    { 9, {COORD_H8, COORD_H7, COORD_G8, COORD_G7, COORD_H6, COORD_F8, COORD_G6, COORD_F7, COORD_F6, COORD_NO}},
    {10, {COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1, COORD_B2, COORD_B1, COORD_C1, COORD_D1, COORD_E1}},
    {10, {COORD_H5, COORD_H4, COORD_H3, COORD_H2, COORD_H1, COORD_G2, COORD_G1, COORD_F1, COORD_E1, COORD_D1}},
    {10, {COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7, COORD_B8, COORD_C8, COORD_D8, COORD_E8}},
    {10, {COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7, COORD_G8, COORD_F8, COORD_E8, COORD_D8}},
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}},
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}},
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_E1, COORD_F1, COORD_H1}},
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_E8, COORD_F8, COORD_H8}},
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_A5, COORD_A6, COORD_A8}},
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_H5, COORD_H6, COORD_H8}},
    { 8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}},
    { 8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}},
    { 8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}},
    { 8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}},
    { 8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}},
    { 8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}},
    { 8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}},
    { 8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}},
    { 8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}},
    { 8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}},
    { 8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}},
    { 8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}},
    { 8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_NO, COORD_NO}},
    { 8, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_NO, COORD_NO}},
    { 7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}},
    { 7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}},
    { 7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}},
    { 7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}},
    { 6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 6, {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 6, {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 5, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 5, {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 5, {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 5, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 4, {COORD_D1, COORD_C2, COORD_B3, COORD_A4, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 4, {COORD_A5, COORD_B6, COORD_C7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 4, {COORD_E1, COORD_F2, COORD_G3, COORD_H4, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 4, {COORD_H5, COORD_G6, COORD_F7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    { 0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}
};

constexpr int pattern_sizes[N_PATTERNS] = {
    9, 10, 10, 10, 8, 8, 8, 8, 7, 6, 5, 4, 0
};

constexpr int feature_to_pattern[N_PATTERN_FEATURES] = {
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12
};

Coord_to_feature coord_to_feature[HW2];
std::vector<int16_t> edax_official_weights;

inline void build_coord_to_feature() {
    for (int coord = 0; coord < HW2; ++coord) {
        coord_to_feature[coord].n_features = 0;
        for (int i = 0; i < MAX_CELL_PATTERNS; ++i) {
            coord_to_feature[coord].features[i] = {0, PNO};
        }
    }
    for (int feature = 0; feature < N_PATTERN_FEATURES; ++feature) {
        const Feature_to_coord &f = feature_to_coord[feature];
        for (int i = 0; i < f.n_cells; ++i) {
            int coord = f.cells[i];
            uint_fast8_t idx = coord_to_feature[coord].n_features++;
            coord_to_feature[coord].features[idx].feature = feature;
            coord_to_feature[coord].features[idx].x = pow3[f.n_cells - 1 - i];
        }
    }
}

constexpr uint32_t edax_official_eval_size[N_PATTERNS] = {
    19683, 59049, 59049, 59049, 6561, 6561, 6561, 6561, 2187, 729, 243, 81, 1
};

constexpr uint32_t edax_official_eval_packed_size[N_PATTERNS] = {
    10206, 29889, 29646, 29646, 3321, 3321, 3321, 3321, 1134, 378, 135, 45, 1
};

constexpr int edax_official_weight_base[N_PATTERN_FEATURES] = {
         0,      0,      0,      0,
     19683,  19683,  19683,  19683,
     78732,  78732,  78732,  78732,
    137781, 137781, 137781, 137781,
    196830, 196830, 196830, 196830,
    196830, 196830, 196830, 196830,
    196830, 196830, 196830, 196830,
    196830, 196830,
    196830, 196830, 196830, 196830,
    196830, 196830, 196830, 196830,
    196830, 196830, 196830, 196830,
    196830, 196830, 196830, 196830,
    196830
};

constexpr int edax_official_feature_offset[N_PATTERN_FEATURES] = {
        0,     0,     0,     0,
        0,     0,     0,     0,
        0,     0,     0,     0,
        0,     0,     0,     0,
        0,     0,     0,     0,
     6561,  6561,  6561,  6561,
    13122, 13122, 13122, 13122,
    19683, 19683,
    26244, 26244, 26244, 26244,
    28431, 28431, 28431, 28431,
    29160, 29160, 29160, 29160,
    29403, 29403, 29403, 29403,
    29484
};

inline uint16_t edax_official_bswap16(uint16_t x) {
    return (uint16_t)((x >> 8) | (x << 8));
}

inline uint32_t edax_official_bswap32(uint32_t x) {
    return ((x & 0x000000FFU) << 24) |
           ((x & 0x0000FF00U) << 8) |
           ((x & 0x00FF0000U) >> 8) |
           ((x & 0xFF000000U) >> 24);
}

inline uint32_t edax_official_opponent_feature(uint32_t idx, uint32_t n_digits) {
    static constexpr uint8_t opponent_digit[3] = {1, 0, 2};
    uint32_t res = opponent_digit[idx % 3];
    if (n_digits > 1) {
        res += edax_official_opponent_feature(idx / 3, n_digits - 1) * 3;
    }
    return res;
}

inline uint32_t edax_official_player_feature(const int sym[], uint32_t n_digits, uint32_t idx) {
    uint32_t res = 0;
    for (uint32_t i = 0; i < n_digits; ++i) {
        res += ((idx / pow3[sym[i]]) % 3) * pow3[i];
    }
    return res;
}

inline std::array<std::vector<int>, 2> edax_official_unpack(uint32_t n_digits, uint32_t size, const int sym[]) {
    std::array<std::vector<int>, 2> pack = {std::vector<int>(size), std::vector<int>(size)};
    uint32_t n_packed = 0;
    for (uint32_t idx = 0; idx < size; ++idx) {
        uint32_t symmetric_idx = edax_official_player_feature(sym, n_digits, idx);
        if (symmetric_idx < idx) {
            pack[0][idx] = pack[0][symmetric_idx];
        } else {
            pack[0][idx] = (int)n_packed++;
        }
        pack[1][edax_official_opponent_feature(idx, n_digits)] = pack[0][idx];
    }
    return pack;
}

inline int edax_official_swap_player_idx(int idx, int pattern_size) {
    int res = idx;
    for (int digit = 0; digit < pattern_size; ++digit) {
        const int trit = (idx / pow3[digit]) % 3;
        if (trit == 0) {
            res += pow3[digit];
        } else if (trit == 1) {
            res -= pow3[digit];
        }
    }
    return res;
}

inline int16_t* edax_official_weight_ptr(int ply, int player) {
    return edax_official_weights.data() + ((ply * 2 + player) * EDAX_OFFICIAL_WEIGHT_STRIDE);
}

inline bool edax_official_has_dat_extension(const std::string &path) {
    return path.size() >= 4 && path.substr(path.size() - 4) == ".dat";
}

inline bool edax_official_open_eval_file(FILE **fp, const char *file, std::string *selected_path) {
    std::vector<std::string> candidates;
    const std::string requested = file == nullptr ? "" : std::string(file);
    if (edax_official_has_dat_extension(requested)) {
        candidates.emplace_back(requested);
    }
    candidates.emplace_back(EXE_DIRECTORY_PATH + "resources/eval_edax.dat");
    candidates.emplace_back(EXE_DIRECTORY_PATH + "resources/eval.dat");
    if (!requested.empty() && !edax_official_has_dat_extension(requested)) {
        candidates.emplace_back(requested);
    }

    for (const std::string &candidate: candidates) {
        if (candidate.empty()) {
            continue;
        }
        if (file_open(fp, candidate.c_str(), "rb")) {
            *selected_path = candidate;
            return true;
        }
    }
    return false;
}

inline void edax_official_write_pattern(
    int16_t *dst0,
    int16_t *dst1,
    const std::vector<int16_t> &packed,
    const std::array<std::vector<int>, 2> &map,
    uint32_t pattern_idx,
    uint32_t &dst_offset,
    uint32_t &packed_offset
) {
    for (uint32_t idx = 0; idx < edax_official_eval_size[pattern_idx]; ++idx) {
        dst0[dst_offset] = packed[packed_offset + map[0][idx]];
        dst1[dst_offset] = packed[packed_offset + map[1][idx]];
        ++dst_offset;
    }
    packed_offset += edax_official_eval_packed_size[pattern_idx];
}

inline bool load_eval_file(const char* file, bool show_log) {
    if (show_log) {
        std::cerr << "Edax official evaluation file request " << file << std::endl;
    }
    static constexpr int sym_s10[] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    static constexpr int sym_c10[] = {9, 8, 7, 6, 4, 5, 3, 2, 1, 0};
    static constexpr int sym_c9[] = {0, 2, 1, 4, 3, 5, 7, 6, 8};

    FILE *fp = nullptr;
    std::string selected_path;
    if (!edax_official_open_eval_file(&fp, file, &selected_path)) {
        std::cerr << "[ERROR] [FATAL] can't open Edax eval.dat from request " << file << std::endl;
        return false;
    }
    if (show_log) {
        std::cerr << "Edax official evaluation file " << selected_path << std::endl;
    }

    uint32_t edax_header = 0;
    uint32_t eval_header = 0;
    uint32_t version = 0;
    uint32_t release = 0;
    uint32_t build = 0;
    double date = 0.0;
    size_t n_read = fread(&edax_header, sizeof(uint32_t), 1, fp);
    n_read += fread(&eval_header, sizeof(uint32_t), 1, fp);
    n_read += fread(&version, sizeof(uint32_t), 1, fp);
    n_read += fread(&release, sizeof(uint32_t), 1, fp);
    n_read += fread(&build, sizeof(uint32_t), 1, fp);
    n_read += fread(&date, sizeof(double), 1, fp);
    if (n_read != 6) {
        std::cerr << "[ERROR] [FATAL] Edax evaluation file header is broken" << std::endl;
        fclose(fp);
        return false;
    }
    bool byte_swap = false;
    if (version > 1000U) {
        const uint32_t swapped_version = edax_official_bswap32(version);
        if (swapped_version >= 1U && swapped_version <= 1000U) {
            byte_swap = true;
            version = swapped_version;
            release = edax_official_bswap32(release);
            build = edax_official_bswap32(build);
        } else {
            std::cerr << "[ERROR] [FATAL] " << selected_path << " does not look like an Edax eval.dat file" << std::endl;
            fclose(fp);
            return false;
        }
    }

    const auto map_s10 = edax_official_unpack(10, 59049, sym_s10);
    const auto map_s8 = edax_official_unpack(8, 6561, sym_s10 + 2);
    const auto map_s7 = edax_official_unpack(7, 2187, sym_s10 + 3);
    const auto map_s6 = edax_official_unpack(6, 729, sym_s10 + 4);
    const auto map_s5 = edax_official_unpack(5, 243, sym_s10 + 5);
    const auto map_s4 = edax_official_unpack(4, 81, sym_s10 + 6);
    const auto map_c9 = edax_official_unpack(9, 19683, sym_c9);
    const auto map_c10 = edax_official_unpack(10, 59049, sym_c10);

    edax_official_weights.assign(
        (size_t)EDAX_OFFICIAL_N_PLY * 2 * EDAX_OFFICIAL_WEIGHT_STRIDE,
        0
    );
    std::vector<int16_t> packed(EDAX_OFFICIAL_N_PACKED_PARAMS);
    for (int ply = 0; ply < EDAX_OFFICIAL_N_PLY; ++ply) {
        if (fread(packed.data(), sizeof(int16_t), EDAX_OFFICIAL_N_PACKED_PARAMS, fp) != EDAX_OFFICIAL_N_PACKED_PARAMS) {
            std::cerr << "[ERROR] [FATAL] Cannot read Edax evaluation weights for ply " << ply << std::endl;
            fclose(fp);
            return false;
        }
        if (byte_swap) {
            for (int16_t &value: packed) {
                value = (int16_t)edax_official_bswap16((uint16_t)value);
            }
        }

        int16_t *dst0 = edax_official_weight_ptr(ply, 0);
        int16_t *dst1 = edax_official_weight_ptr(ply, 1);
        uint32_t dst_offset = 0;
        uint32_t packed_offset = 0;
        edax_official_write_pattern(dst0, dst1, packed, map_c9, 0, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_c10, 1, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s10, 2, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s10, 3, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s8, 4, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s8, 5, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s8, 6, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s8, 7, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s7, 8, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s6, 9, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s5, 10, dst_offset, packed_offset);
        edax_official_write_pattern(dst0, dst1, packed, map_s4, 11, dst_offset, packed_offset);
        dst0[dst_offset] = packed[packed_offset];
        dst1[dst_offset] = packed[packed_offset];
    }
    fclose(fp);
    if (show_log) {
        std::cerr << "Edax official evaluation weights " << version << "." << release << "." << build << " loaded" << std::endl;
    }
    return true;
}

inline bool load_eval_move_ordering_end_file(const char*, bool show_log) {
    if (show_log) {
        std::cerr << "Edax official experiment uses neutral move-ordering-end evaluation" << std::endl;
    }
    return true;
}

inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log) {
    bool eval_loaded = load_eval_file(file, show_log);
    if (!eval_loaded) {
        std::cerr << "[ERROR] [FATAL] Edax official evaluation file not loaded" << std::endl;
        return false;
    }
    if (!load_eval_move_ordering_end_file(mo_end_nws_file, show_log)) {
        return false;
    }
    build_coord_to_feature();
    if (show_log) {
        std::cerr << "Edax official experiment evaluation function initialized" << std::endl;
    }
    return true;
}

bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval_edax.dat", EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev", show_log);
}

inline uint_fast16_t pick_pattern_idx(const uint_fast8_t b_arr[], const Feature_to_coord *f) {
    uint_fast16_t res = 0;
    for (int i = 0; i < f->n_cells; ++i) {
        res *= 3;
        res += b_arr[HW2_M1 - f->cells[i]];
    }
    return res;
}

#if !USE_SIMD
inline int edax_official_normalize_phase(int phase_idx) {
    return std::clamp(phase_idx, 0, EDAX_OFFICIAL_N_PLY - 1);
}

inline int edax_official_calc_raw(const int phase_idx, const uint16_t values[], const bool reverse_values) {
    const int16_t *weights = edax_official_weight_ptr(edax_official_normalize_phase(phase_idx), 0);
    int res = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        const int idx = reverse_values
            ? edax_official_swap_player_idx(values[i], pattern_sizes[feature_to_pattern[i]])
            : values[i];
        res += weights[edax_official_weight_base[i] + edax_official_feature_offset[i] + idx];
    }
    return res;
}

inline int calc_pattern_generic(const int phase_idx, const Eval_search *eval) {
    return edax_official_calc_raw(phase_idx, eval->features[eval->feature_idx], eval->reversed[eval->feature_idx]);
}
#endif

#if USE_SIMD
inline int edax_official_normalize_phase(int phase_idx) {
    return std::clamp(phase_idx, 0, EDAX_OFFICIAL_N_PLY - 1);
}

inline int edax_official_calc_raw(const int phase_idx, const int player_idx, const uint16_t values[]) {
    const int16_t *weights = edax_official_weight_ptr(edax_official_normalize_phase(phase_idx), player_idx);
    int res = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        res += weights[edax_official_weight_base[i] + edax_official_feature_offset[i] + values[i]];
    }
    return res;
}

inline void pack_simd_features(Eval_features *features, const uint16_t values[EDAX_OFFICIAL_N_FEATURES_CEIL]) {
    for (int i = 0; i < N_EVAL_VECTORS; ++i) {
        int b = i * 16;
        features->f256[i] = _mm256_setr_epi16(
            values[b], values[b + 1], values[b + 2], values[b + 3],
            values[b + 4], values[b + 5], values[b + 6], values[b + 7],
            values[b + 8], values[b + 9], values[b + 10], values[b + 11],
            values[b + 12], values[b + 13], values[b + 14], values[b + 15]
        );
    }
}

inline void unpack_simd_features(const Eval_features *features, uint16_t values[EDAX_OFFICIAL_N_FEATURES_CEIL]) {
    for (int i = 0; i < N_EVAL_VECTORS; ++i) {
        _mm256_storeu_si256((__m256i*)(values + i * 16), features->f256[i]);
    }
}

inline int calc_pattern(const int phase_idx, Eval_features *features) {
    uint16_t values[EDAX_OFFICIAL_N_FEATURES_CEIL];
    unpack_simd_features(features, values);
    return edax_official_calc_raw(phase_idx, 0, values);
}

inline void fill_simd_features_from_board(Board *board, Eval_features *features) {
    uint_fast8_t b_arr[HW2];
    uint16_t values[EDAX_OFFICIAL_N_FEATURES_CEIL] = {};
    board->translate_to_arr_player(b_arr);
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        values[i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    }
    pack_simd_features(features, values);
}
#else
inline int calc_pattern(const int phase_idx, Eval_search *eval) {
    return calc_pattern_generic(phase_idx, eval);
}
#endif

inline int calc_pattern_move_ordering_end(
#if USE_SIMD
    Eval_features*
#else
    Eval_search*
#endif
) {
    return 0;
}

inline int mid_evaluate(Board *board) {
    Search search(board);
    calc_eval_features(&(search.board), &(search.eval));
    int phase_idx = search.phase();
#if USE_SIMD
    int res = calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx]);
#else
    int res = calc_pattern(phase_idx, &search.eval);
#endif
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return std::clamp(res, -SCORE_MAX, SCORE_MAX);
}

inline int mid_evaluate_diff(Search *search) {
    int phase_idx = search->phase();
#if USE_SIMD
    int res = calc_pattern(phase_idx, &search->eval.features[search->eval.feature_idx]);
#else
    int res = calc_pattern(phase_idx, &search->eval);
#endif
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return std::clamp(res, -SCORE_MAX, SCORE_MAX);
}

inline int mid_evaluate_move_ordering_end(Search *search) {
#if USE_SIMD
    int res = calc_pattern_move_ordering_end(&search->eval.features[search->eval.feature_idx]);
#else
    int res = calc_pattern_move_ordering_end(&search->eval);
#endif
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return res;
}

inline void calc_eval_features(Board *board, Eval_search *eval) {
#if USE_SIMD
    fill_simd_features_from_board(board, &eval->features[0]);
    eval->feature_idx = 0;
#else
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        eval->features[0][i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    }
    eval->reversed[0] = 0;
    eval->feature_idx = 0;
#endif
}

#if USE_SIMD
inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board) {
    Board next = board->copy();
    next.move_board(flip);
    ++eval->feature_idx;
    fill_simd_features_from_board(&next, &eval->features[eval->feature_idx]);
}

inline void eval_pass(Eval_search *eval, const Board *board) {
    Board next = board->copy();
    next.pass();
    fill_simd_features_from_board(&next, &eval->features[eval->feature_idx]);
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board) {
    eval_move(eval, flip, board);
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
    eval_pass(eval, board);
}
#else
inline void eval_move(Eval_search *eval, const Flip *flip) {
    uint_fast8_t i, cell;
    uint64_t f;
    for (i = 0; i < N_PATTERN_FEATURES; ++i) {
        eval->features[eval->feature_idx + 1][i] = eval->features[eval->feature_idx][i];
    }
    if (eval->reversed[eval->feature_idx]) {
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
            }
        }
    } else {
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
            }
        }
    }
    eval->reversed[eval->feature_idx + 1] = eval->reversed[eval->feature_idx] ^ 1;
    ++eval->feature_idx;
}

inline void eval_pass(Eval_search *eval) {
    eval->reversed[eval->feature_idx] ^= 1;
}

inline void eval_move_endsearch(Eval_search *eval, const Flip *flip) {
    eval_move(eval, flip);
}

inline void eval_pass_endsearch(Eval_search *eval) {
    eval_pass(eval);
}
#endif

inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_undo_endsearch(Eval_search *eval) {
    --eval->feature_idx;
}
