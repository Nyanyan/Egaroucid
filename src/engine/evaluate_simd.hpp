/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function with AVX2
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include <cstring>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_common.hpp"

/*
    @brief evaluation pattern definition for SIMD
*/
#define CEIL_N_SYMMETRY_PATTERNS 64         // N_SYMMETRY_PATTRENS
#define N_PATTERN_PARAMS_RAW 573966
#define N_PATTERN_PARAMS (N_PATTERN_PARAMS_RAW + 1) // +1 for byte bound
#define FEATURE1_START_IDX 39366            // feature1 special case
#define SIMD_EVAL_MAX_VALUE 4092            // evaluate range [-4092, 4092]
#define N_SIMD_EVAL_FEATURES_SIMPLE 2
#define N_SIMD_EVAL_FEATURES_COMP 2
#define N_SIMD_EVAL_FEATURE_CELLS 16
#define N_SIMD_EVAL_FEATURE_GROUP 4

// number of cells included in the group
#define MAX_N_CELLS_GROUP0 9
#define MAX_N_CELLS_GROUP1 10
#define MAX_N_CELLS_GROUP2 10
#define MAX_N_CELLS_GROUP3 10


/*
    @brief evaluation pattern definition for SIMD move ordering end
*/
#define N_PATTERN_PARAMS_MO_END (236196 + 1) // +1 for byte bound
#define SIMD_EVAL_MAX_VALUE_MO_END 16380
#define SHIFT_EVAL_MO_END 101574 // pattern_starts[8] - 1

constexpr Feature_to_coord feature_to_coord[CEIL_N_SYMMETRY_PATTERNS] = {
    // 0 hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}},
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}},
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}},
    
    // 1 d6 + 2C + X
    {9, {COORD_B1, COORD_C1, COORD_D2, COORD_E3, COORD_G2, COORD_F4, COORD_G5, COORD_H6, COORD_H7, COORD_NO}},
    {9, {COORD_H2, COORD_H3, COORD_G4, COORD_F5, COORD_G7, COORD_E6, COORD_D7, COORD_C8, COORD_B8, COORD_NO}},
    {9, {COORD_G8, COORD_F8, COORD_E7, COORD_D6, COORD_B7, COORD_C5, COORD_B4, COORD_A3, COORD_A2, COORD_NO}},
    {9, {COORD_A7, COORD_A6, COORD_B5, COORD_C4, COORD_B2, COORD_D3, COORD_E2, COORD_F1, COORD_G1, COORD_NO}},

    // 2 hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}},
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}},
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}},

    // 3 d7 + 2 corner
    {9, {COORD_A1, COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_H8, COORD_NO}},
    {9, {COORD_H1, COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_A8, COORD_NO}},
    {9, {COORD_H8, COORD_G8, COORD_F7, COORD_E6, COORD_D5, COORD_C4, COORD_B3, COORD_A2, COORD_A1, COORD_NO}},
    {9, {COORD_A8, COORD_A7, COORD_B6, COORD_C5, COORD_D4, COORD_E3, COORD_F2, COORD_G1, COORD_H1, COORD_NO}},

    // 4 hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}},
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}},

    // 5 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}},

    // 6 d5 + 2X
    {7, {COORD_B2, COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_G7, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_G2, COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_B7, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_G7, COORD_E8, COORD_D7, COORD_C6, COORD_B5, COORD_A4, COORD_B2, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_B7, COORD_A5, COORD_B4, COORD_C3, COORD_D2, COORD_E1, COORD_G2, COORD_NO, COORD_NO, COORD_NO}},

    // 7 d8 + 2C
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_A2, COORD_B1}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_D4, COORD_C3, COORD_B2, COORD_A1, COORD_H7, COORD_G8}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_H2, COORD_G1}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_A7, COORD_B8}},
 
    // 8 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}},
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}},
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},

    // 9 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}},

    // 10 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}},
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}},
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}},
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}},

    // 11 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}},

    // 12 edge + y
    {10, {COORD_C2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_F2}},
    {10, {COORD_B3, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B6}},
    {10, {COORD_C7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_F7}},
    {10, {COORD_G3, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G6}},

    // 13 narrow triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_A3, COORD_A4, COORD_A5}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_H3, COORD_H4, COORD_H5}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_A6, COORD_A5, COORD_A4}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_H6, COORD_H5, COORD_H4}},

    // 14 fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}},
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}},
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}},
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}},

    // 15 anvil
    {10, {COORD_C6, COORD_D6, COORD_D7, COORD_D8, COORD_C8, COORD_F8, COORD_E8, COORD_E7, COORD_E6, COORD_F6}},
    {10, {COORD_C3, COORD_C4, COORD_B4, COORD_A4, COORD_A3, COORD_A6, COORD_A5, COORD_B5, COORD_C5, COORD_C6}},
    {10, {COORD_F3, COORD_E3, COORD_E2, COORD_E1, COORD_F1, COORD_C1, COORD_D1, COORD_D2, COORD_D3, COORD_C3}},
    {10, {COORD_F6, COORD_F5, COORD_G5, COORD_H5, COORD_H6, COORD_H3, COORD_H4, COORD_G4, COORD_F4, COORD_F3}}
};

constexpr Coord_to_feature coord_to_feature[HW2] = {

};

/*
    @brief constants used for evaluation function with SIMD
*/
__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_SIMD_EVAL_FEATURES];
__m256i eval_move_unflipped_16bit[N_16BIT][N_SIMD_EVAL_FEATURE_GROUP][N_SIMD_EVAL_FEATURES];
__m256i eval_simd_offsets_simple[N_SIMD_EVAL_FEATURES_SIMPLE]; // 16bit * 16 * N
__m256i eval_simd_offsets_comp[N_SIMD_EVAL_FEATURES_COMP * 2]; // 32bit * 8 * N

__m256i eval_surround_mask;
__m256i eval_surround_shift1879;

/*
    @brief evaluation parameters
*/
// normal
int16_t pattern_arr[N_PHASES][N_PATTERN_PARAMS];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
// move ordering evaluation
int16_t pattern_move_ordering_end_arr[N_PATTERN_PARAMS_MO_END];

inline bool load_eval_file(const char* file, bool show_log) {
    if (show_log)
        std::cerr << "evaluation file " << file << std::endl;
    bool failed = false;
    std::vector<int16_t> unzipped_params = load_unzip_egev2(file, show_log, &failed);
    if (failed) {
        return false;
    }
    size_t param_idx = 0;
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
        pattern_arr[phase_idx][0] = 0; // memory bound
        std::memcpy(pattern_arr[phase_idx] + 1, &unzipped_params[param_idx], sizeof(short) * N_PATTERN_PARAMS_RAW);
        param_idx += N_PATTERN_PARAMS_RAW;
        std::memcpy(eval_num_arr[phase_idx], &unzipped_params[param_idx], sizeof(short) * MAX_STONE_NUM);
        param_idx += MAX_STONE_NUM;
        std::memcpy(eval_sur0_sur1_arr[phase_idx], &unzipped_params[param_idx], sizeof(short) * MAX_SURROUND * MAX_SURROUND);
        param_idx += MAX_SURROUND * MAX_SURROUND;
    }
    // check max value
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
        for (int i = 1; i < N_PATTERN_PARAMS; ++i) {
            if (pattern_arr[phase_idx][i] < -SIMD_EVAL_MAX_VALUE) {
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[phase_idx][i] << std::endl;
                pattern_arr[phase_idx][i] = -SIMD_EVAL_MAX_VALUE;
            }
            if (pattern_arr[phase_idx][i] > SIMD_EVAL_MAX_VALUE) {
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[phase_idx][i] << std::endl;
                pattern_arr[phase_idx][i] = SIMD_EVAL_MAX_VALUE;
            }
            pattern_arr[phase_idx][i] += SIMD_EVAL_MAX_VALUE;
        }
    }
    return true;
}

inline bool load_eval_move_ordering_end_file(const char* file, bool show_log) {
    if (show_log)
        std::cerr << "evaluation for move ordering end file " << file << std::endl;
    FILE* fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    pattern_move_ordering_end_arr[0] = 0; // memory bound
    if (fread(pattern_move_ordering_end_arr + 1, 2, N_PATTERN_PARAMS_MO_END - 1, fp) < N_PATTERN_PARAMS_MO_END - 1) {
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end broken" << std::endl;
        fclose(fp);
        return false;
    }
    // check max value
    for (int i = 1; i < N_PATTERN_PARAMS_MO_END; ++i) {
        if (pattern_move_ordering_end_arr[i] < -SIMD_EVAL_MAX_VALUE_MO_END) {
            std::cerr << "[ERROR] evaluation value too low. you can ignore this error. index " << i << " found " << pattern_move_ordering_end_arr[i] << std::endl;
            pattern_move_ordering_end_arr[i] = -SIMD_EVAL_MAX_VALUE_MO_END;
        }
        if (pattern_move_ordering_end_arr[i] > SIMD_EVAL_MAX_VALUE_MO_END) {
            std::cerr << "[ERROR] evaluation value too high. you can ignore this error. index " << i << " found " << pattern_move_ordering_end_arr[i] << std::endl;
            pattern_move_ordering_end_arr[i] = SIMD_EVAL_MAX_VALUE_MO_END;
        }
        pattern_move_ordering_end_arr[i] += SIMD_EVAL_MAX_VALUE_MO_END;
    }
    return true;
}

inline void pre_calculate_eval_constant() {
    constexpr int pattern_starts[N_PATTERNS] = {
        1, 6562, 13123, 19684, // features[0]
        1, 244, 973, 3160, // features[1] from FEATURE1_START_IDX
        //39367, 39610, 40339, 42526, // features[1]
        101575, 160624, 219673, 278722, // features[2]
        337771, 396820, 455869, 514918 // features[3]
    };
    { // calc_eval_features initialization
        int16_t f2c[16];
        for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i) {
            for (int j = 0; j < MAX_PATTERN_CELLS - 1; ++j) {
                for (int k = 0; k < 16; ++k) {
                    f2c[k] = (j < feature_to_coord[i * 16 + k].n_cells - 1) ? 3 : 1;
                }
                feature_to_coord_simd_mul[i][j] = _mm256_set_epi16(
                    f2c[0], f2c[1], f2c[2], f2c[3], 
                    f2c[4], f2c[5], f2c[6], f2c[7], 
                    f2c[8], f2c[9], f2c[10], f2c[11], 
                    f2c[12], f2c[13], f2c[14], f2c[15]
                );
            }
        }
        int32_t f2c32[8];
        for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i) {
            for (int j = 0; j < MAX_PATTERN_CELLS; ++j) {
                for (int k = 0; k < 8; ++k)
                    f2c32[k] = feature_to_coord[i * 16 + k * 2 + 1].cells[j];
                feature_to_coord_simd_cell[i][j][0] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
                for (int k = 0; k < 8; ++k)
                    f2c32[k] = feature_to_coord[i * 16 + k * 2].cells[j];
                feature_to_coord_simd_cell[i][j][1] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
            }
        }
        eval_simd_offsets_simple[0] = _mm256_set_epi16(
            (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], 
            (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], 
            (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], 
            (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3]
        );
        eval_simd_offsets_simple[1] = _mm256_set_epi16(
            (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], 
            (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], 
            (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], 
            (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7]
        );
    }
    { // eval_move initialization
        uint16_t c2f[CEIL_N_SYMMETRY_PATTERNS];
        for (int cell = 0; cell < HW2; ++cell) { // 0 for h8, 63 for a1
            for (int i = 0; i < CEIL_N_SYMMETRY_PATTERNS; ++i)
                c2f[i] = 0;
            for (int i = 0; i < coord_to_feature[cell].n_features; ++i)
                c2f[coord_to_feature[cell].features[i].feature] = coord_to_feature[cell].features[i].x;
            for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i) {
                int idx = i * 16;
                coord_to_feature_simd[cell][i] = _mm256_set_epi16(
                    c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], 
                    c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                    c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], 
                    c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
                );
            }
        }
        for (int bits = 0; bits < N_16BIT; ++bits) { // 1: unflipped discs, 0: others
            for (int group = 0; group < N_SIMD_EVAL_FEATURE_GROUP; ++group) { // a1-h2, a3-h4, ..., a7-h8
                for (int i = 0; i < CEIL_N_SYMMETRY_PATTERNS; ++i)
                    c2f[i] = 0;
                for (int cell = 0; cell < N_SIMD_EVAL_FEATURE_CELLS; ++cell) {
                    if (1 & (bits >> cell)) {
                        int global_cell = group * N_SIMD_EVAL_FEATURE_CELLS + cell;
                        for (int i = 0; i < coord_to_feature[global_cell].n_features; ++i) {
                            c2f[coord_to_feature[global_cell].features[i].feature] += coord_to_feature[global_cell].features[i].x;
                        }
                    }
                }
                for (int simd_feature_idx = 0; simd_feature_idx < N_SIMD_EVAL_FEATURES; ++simd_feature_idx) {
                    int idx = simd_feature_idx * 16;
                    eval_move_unflipped_16bit[bits][group][simd_feature_idx] = _mm256_set_epi16(
                        c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], 
                        c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                        c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], 
                        c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
                    );
                }
            }
        }
        for (int i = 0; i < N_SIMD_EVAL_FEATURES_COMP; ++i) {
            int i4 = i * 4;
            eval_simd_offsets_comp[i * 2] = _mm256_set_epi32(
                pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], 
                pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4]
            );
            eval_simd_offsets_comp[i * 2 + 1] = _mm256_set_epi32(
                pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], 
                pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4]
            );
        }
        eval_lower_mask = _mm256_set1_epi32(0x0000FFFF);
    }
    { // calc_surround initialization
        eval_surround_mask = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
        eval_surround_shift1879 = _mm256_set_epi64x(1, HW, HW_M1, HW_P1);
    }
}

/*
    @brief initialize the evaluation function

    @param file                 evaluation file name
    @param show_log             debug information?
    @return evaluation function conpletely initialized?
*/
inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log) {
    bool eval_loaded = load_eval_file(file, show_log);
    if (!eval_loaded) {
        std::cerr << "[ERROR] [FATAL] evaluation file not loaded" << std::endl;
        return false;
    }
    bool eval_move_ordering_end_nws_loaded = load_eval_move_ordering_end_file(mo_end_nws_file, show_log);
    if (!eval_move_ordering_end_nws_loaded) {
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end not loaded" << std::endl;
        return false;
    }
    pre_calculate_eval_constant();
    if (show_log)
        std::cerr << "evaluation function initialized" << std::endl;
    return true;
}

/*
    @brief Wrapper of evaluation initializing

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

/*
    @brief Wrapper of evaluation initializing

    @return evaluation function conpletely initialized?
*/
bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval.egev2", EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev", show_log);
}
/*
    @brief calculate surround value used in evaluation function

    @param discs                a bitboard representing discs
    @param empties              a bitboard representing empties
    @return surround value
*/
inline int calc_surround(const uint64_t discs, const uint64_t empties) {
    __m256i pl = _mm256_set1_epi64x(discs);
    pl = _mm256_and_si256(pl, eval_surround_mask);
    pl = _mm256_or_si256(_mm256_sllv_epi64(pl, eval_surround_shift1879), _mm256_srlv_epi64(pl, eval_surround_shift1879));
    __m128i res = _mm_or_si128(_mm256_castsi256_si128(pl), _mm256_extracti128_si256(pl, 1));
    res = _mm_or_si128(res, _mm_shuffle_epi32(res, 0x4e));
    return pop_count_ull(_mm_cvtsi128_si64(res) & empties);
}
#define CALC_SURROUND_FUNCTION

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline __m256i calc_idx8_comp(const __m128i feature, const int i) {
    return _mm256_add_epi32(_mm256_cvtepu16_epi32(feature), eval_simd_offsets_comp[i]);
}

inline __m256i gather_eval(const int *start_addr, const __m256i idx8) {
    return _mm256_i32gather_epi32(start_addr, idx8, 2); // stride is 2 byte, because 16 bit array used, HACK: if (SIMD_EVAL_MAX_VALUE * 2) * (N_ADD=8) < 2 ^ 16, AND is unnecessary
    // return _mm256_and_si256(_mm256_i32gather_epi32(start_addr, idx8, 2), eval_lower_mask);
}

inline int calc_pattern(const int phase_idx, Eval_features *features) {
    const int *start_addr = (int*)pattern_arr[phase_idx];
    const int *start_addr2 = (int*)&pattern_arr[phase_idx][FEATURE1_START_IDX];
    __m256i res256 =                  gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[0]));    // hv4 corner9
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[1])));   // hv2 hv3
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[2])));  // d7 d8+2C
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[3])));  // d5 d6
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[4], 0)));       // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[5], 1)));       // edge+2X triangle
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[6], 2)));       // fish kite
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[7], 3)));       // edge+2Y narrow_triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE * N_SYMMETRY_PATTERNS;
}

inline int calc_pattern_move_ordering_end(Eval_features *features) {
    const int *start_addr = (int*)(pattern_move_ordering_end_arr - SHIFT_EVAL_MO_END);
    __m256i res256 =                  gather_eval(start_addr, calc_idx8_comp(features->f128[4], 0));        // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[5], 1)));       // edge+2X triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE_MO_END * N_SYMMETRY_PATTERNS_MO_END;
}

inline void calc_eval_features(Board *board, Eval_search *eval);

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline int mid_evaluate(Board *board) {
    Search search(board);
    calc_eval_features(&(search.board), &(search.eval));
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search.phase();
    empties = ~(search.board.player | search.board.opponent);
    sur0 = calc_surround(search.board.player, empties);
    sur1 = calc_surround(search.board.opponent, empties);
    num0 = pop_count_ull(search.board.player);
    int res = calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_diff(Search *search) {
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search->phase();
    empties = ~(search->board.player | search->board.opponent);
    sur0 = calc_surround(search->board.player, empties);
    sur1 = calc_surround(search->board.opponent, empties);
    num0 = pop_count_ull(search->board.player);
    int res = calc_pattern(phase_idx, &search->eval.features[search->eval.feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_move_ordering_end(Search *search) {
    int res = calc_pattern_move_ordering_end(&search->eval.features[search->eval.feature_idx]);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return res;
}

inline void calc_feature_vector(__m256i &f, const int *b_arr_int, const int i, const int n) {
    f = _mm256_set1_epi16(0);
    for (int j = 0; j < n; ++j) { // n: max n_cells in pattern - 1
        f = _mm256_add_epi16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][0], 4));
        f = _mm256_add_epi16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][1], 4), 16));
        f = _mm256_mullo_epi16(f, feature_to_coord_simd_mul[i][j]);
    }
    f = _mm256_add_epi16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][0], 4));
    f = _mm256_add_epi16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][1], 4), 16));
}

/*
    @brief calculate features for pattern evaluation

    @param search               search information
*/
inline void calc_eval_features(Board *board, Eval_search *eval) {
    int b_arr_int[HW2 + 1];
    board->translate_to_arr_player_rev(b_arr_int);
    b_arr_int[COORD_NO] = 0;
    calc_feature_vector(eval->features[0].f256[0], b_arr_int, 0, MAX_N_CELLS_GROUP0 - 1);
    calc_feature_vector(eval->features[0].f256[1], b_arr_int, 1, MAX_N_CELLS_GROUP1 - 1);
    calc_feature_vector(eval->features[0].f256[2], b_arr_int, 2, MAX_N_CELLS_GROUP2 - 1);
    calc_feature_vector(eval->features[0].f256[3], b_arr_int, 3, MAX_N_CELLS_GROUP3 - 1);
    eval->feature_idx = 0;
    eval->features[eval->feature_idx].f256[0] = _mm256_add_epi16(eval->features[eval->feature_idx].f256[0], eval_simd_offsets_simple[0]); // global index
    eval->features[eval->feature_idx].f256[1] = _mm256_add_epi16(eval->features[eval->feature_idx].f256[1], eval_simd_offsets_simple[1]); // global index
}

/*
    @brief move evaluation features

        put cell        2 -> 1 (empty -> opponent)  sub
        player discs    0 -> 1 (player -> opponent) add
        opponent discs  1 -> 0 (player -> opponent) sub
        flipped discs   1 -> 1 (opponent -> opponent)
        empty cells     2 -> 2 (empty -> empty)
    
    @param eval                 evaluation features
    @param flip                 flip information
*/
inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board) {
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f0, f1, f2, f3;
    uint16_t unflipped_p;
    uint16_t unflipped_o;
    // put cell 2 -> 1
    f0 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[0], coord_to_feature_simd[flip->pos][0]);
    f1 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[1], coord_to_feature_simd[flip->pos][1]);
    f2 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[2], coord_to_feature_simd[flip->pos][2]);
    f3 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[3], coord_to_feature_simd[flip->pos][3]);
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        // player discs 0 -> 1
        unflipped_p = ~flipped_group[i] & player_group[i];
        f0 = _mm256_add_epi16(f0, eval_move_unflipped_16bit[unflipped_p][i][0]);
        f1 = _mm256_add_epi16(f1, eval_move_unflipped_16bit[unflipped_p][i][1]);
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[unflipped_p][i][2]);
        f3 = _mm256_add_epi16(f3, eval_move_unflipped_16bit[unflipped_p][i][3]);
        // opponent discs 1 -> 0
        unflipped_o = ~flipped_group[i] & opponent_group[i];
        f0 = _mm256_sub_epi16(f0, eval_move_unflipped_16bit[unflipped_o][i][0]);
        f1 = _mm256_sub_epi16(f1, eval_move_unflipped_16bit[unflipped_o][i][1]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[unflipped_o][i][2]);
        f3 = _mm256_sub_epi16(f3, eval_move_unflipped_16bit[unflipped_o][i][3]);
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[0] = f0;
    eval->features[eval->feature_idx].f256[1] = f1;
    eval->features[eval->feature_idx].f256[2] = f2;
    eval->features[eval->feature_idx].f256[3] = f3;
}

/*
    @brief undo evaluation features

    @param eval                 evaluation features
*/
inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

/*
    @brief pass evaluation features

        player discs    0 -> 1 (player -> opponent) add
        opponent discs  1 -> 0 (player -> opponent) sub

    @param eval                 evaluation features
*/
inline void eval_pass(Eval_search *eval, const Board *board) {
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f0, f1, f2, f3;
    f0 = eval->features[eval->feature_idx].f256[0];
    f1 = eval->features[eval->feature_idx].f256[1];
    f2 = eval->features[eval->feature_idx].f256[2];
    f3 = eval->features[eval->feature_idx].f256[3];
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        f0 = _mm256_add_epi16(f0, eval_move_unflipped_16bit[player_group[i]][i][0]);
        f1 = _mm256_add_epi16(f1, eval_move_unflipped_16bit[player_group[i]][i][1]);
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[player_group[i]][i][2]);
        f3 = _mm256_add_epi16(f3, eval_move_unflipped_16bit[player_group[i]][i][3]);
        f0 = _mm256_sub_epi16(f0, eval_move_unflipped_16bit[opponent_group[i]][i][0]);
        f1 = _mm256_sub_epi16(f1, eval_move_unflipped_16bit[opponent_group[i]][i][1]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[opponent_group[i]][i][2]);
        f3 = _mm256_sub_epi16(f3, eval_move_unflipped_16bit[opponent_group[i]][i][3]);
    }
    eval->features[eval->feature_idx].f256[0] = f0;
    eval->features[eval->feature_idx].f256[1] = f1;
    eval->features[eval->feature_idx].f256[2] = f2;
    eval->features[eval->feature_idx].f256[3] = f3;
}




// only corner+block cross edge+2X triangle
inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board) {
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    // put cell 2 -> 1
    __m256i f2 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[2], coord_to_feature_simd[flip->pos][2]);
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        // player discs 0 -> 1
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & player_group[i]][i][2]);
        // opponent discs 1 -> 0
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & opponent_group[i]][i][2]);
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[2] = f2;
}

inline void eval_undo_endsearch(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f2 = eval->features[eval->feature_idx].f256[2];
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[player_group[i]][i][2]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[opponent_group[i]][i][2]);
    }
    eval->features[eval->feature_idx].f256[2] = f2;
}
