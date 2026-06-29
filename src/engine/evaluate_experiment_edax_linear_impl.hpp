/*
    Egaroucid Project

    @file evaluate_experiment_edax_linear_impl.hpp
        Isolated Edax-linear evaluation experiment implementation
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <cstring>
#include <future>
#include <iostream>
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

constexpr int EDAX_LINEAR_N_FEATURES_CEIL = ((N_PATTERN_FEATURES + 15) / 16) * 16;
constexpr int EDAX_LINEAR_N_PATTERN_PARAMS_RAW = 226315;
constexpr int EDAX_LINEAR_EVAL_MAX_VALUE = 4092;

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
int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];

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

inline int swap_player_idx(int i, int pattern_size) {
    int ri = i;
    for (int j = 0; j < pattern_size; ++j) {
        if ((i / pow3[j]) % 3 == 0) {
            ri += pow3[j];
        } else if ((i / pow3[j]) % 3 == 1) {
            ri -= pow3[j];
        }
    }
    return ri;
}

void init_pattern_arr_rev(int phase_idx, int pattern_idx, int siz) {
    for (int i = 0; i < (int)pow3[siz]; ++i) {
        int ri = swap_player_idx(i, siz);
        pattern_arr[1][phase_idx][pattern_idx][ri] = pattern_arr[0][phase_idx][pattern_idx][i];
    }
}

inline bool load_eval_file(const char* file, bool show_log) {
    if (show_log) {
        std::cerr << "Edax-linear experiment evaluation file " << file << std::endl;
    }
    bool failed = false;
    std::vector<int16_t> unzipped_params = load_unzip_egev2(file, show_log, &failed);
    if (failed) {
        return false;
    }
    const size_t expected = (size_t)N_PHASES * EDAX_LINEAR_N_PATTERN_PARAMS_RAW;
    if (unzipped_params.size() < expected) {
        std::cerr << "[ERROR] [FATAL] Edax-linear evaluation file has " << unzipped_params.size()
                  << " params; expected at least " << expected << std::endl;
        return false;
    }
    size_t param_idx = 0;
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
        for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
            std::memcpy(pattern_arr[0][phase_idx][pattern_idx], &unzipped_params[param_idx], sizeof(short) * pow3[pattern_sizes[pattern_idx]]);
            param_idx += pow3[pattern_sizes[pattern_idx]];
        }
    }
    if (thread_pool.size() >= 2) {
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
            for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
                bool pushed = false;
                while (!pushed) {
                    tasks[i] = thread_pool.push(&pushed, std::bind(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx]));
                }
                ++i;
            }
        }
        for (std::future<void> &task: tasks) {
            task.get();
        }
    } else {
        for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
            for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
                init_pattern_arr_rev(phase_idx, pattern_idx, pattern_sizes[pattern_idx]);
            }
        }
    }
    return true;
}

inline bool load_eval_move_ordering_end_file(const char*, bool show_log) {
    if (show_log) {
        std::cerr << "Edax-linear experiment uses neutral move-ordering-end evaluation" << std::endl;
    }
    return true;
}

inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log) {
    bool eval_loaded = load_eval_file(file, show_log);
    if (!eval_loaded) {
        std::cerr << "[ERROR] [FATAL] Edax-linear evaluation file not loaded" << std::endl;
        return false;
    }
    if (!load_eval_move_ordering_end_file(mo_end_nws_file, show_log)) {
        return false;
    }
    build_coord_to_feature();
    if (show_log) {
        std::cerr << "Edax-linear experiment evaluation function initialized" << std::endl;
    }
    return true;
}

bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval.egev2", EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev", show_log);
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
inline int calc_pattern_generic(const int phase_idx, const Eval_search *eval) {
    int res = 0;
    const int player_idx = eval->reversed[eval->feature_idx] ? 1 : 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        res += pattern_arr[player_idx][phase_idx][feature_to_pattern[i]][eval->features[eval->feature_idx][i]];
    }
    return res;
}
#endif

#if USE_SIMD
inline void pack_simd_features(Eval_features *features, const uint16_t values[EDAX_LINEAR_N_FEATURES_CEIL]) {
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

inline void unpack_simd_features(const Eval_features *features, uint16_t values[EDAX_LINEAR_N_FEATURES_CEIL]) {
    for (int i = 0; i < N_EVAL_VECTORS; ++i) {
        _mm256_storeu_si256((__m256i*)(values + i * 16), features->f256[i]);
    }
}

inline int calc_pattern(const int phase_idx, Eval_features *features) {
    uint16_t values[EDAX_LINEAR_N_FEATURES_CEIL];
    unpack_simd_features(features, values);
    int res = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        res += pattern_arr[0][phase_idx][feature_to_pattern[i]][values[i]];
    }
    return res;
}

inline void fill_simd_features_from_board(Board *board, Eval_features *features) {
    uint_fast8_t b_arr[HW2];
    uint16_t values[EDAX_LINEAR_N_FEATURES_CEIL] = {};
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
