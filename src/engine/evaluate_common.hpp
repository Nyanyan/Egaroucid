/*
    Egaroucid Project

    @file evaluate_common.hpp
        Common things of evaluation function
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <cstring>
#include "board.hpp"
#ifdef EGAROUCID_NNUE
#include "nnue/nnue_eval.hpp"
#endif

/*
    @brief evaluation pattern definition
*/
// disc patterns
constexpr int N_PATTERNS = 16;          // number of patterns used
constexpr int MAX_CELL_PATTERNS = 15;   // 1 cell belongs up to 15 patterns
constexpr int MAX_PATTERN_CELLS = 10;   // up to 10 cells for a pattern
constexpr int MAX_EVALUATE_IDX = 59049; // 3^10: up to 10 cells for pattern
constexpr int N_PATTERN_FEATURES = 64;  // 64 features are used
#if USE_SIMD_EVALUATION
constexpr int N_EVAL_VECTORS = 4; // 16 (elems per 256 bit vector) * N_EVAL_VECTORS >= N_PATTERN_FEATURES
#endif

// additional features
constexpr int MAX_STONE_NUM = 65; // [0,64]

// evaluation phase definition
constexpr int N_PHASES = 60;
constexpr int PHASE_N_DISCS = 1; // 60 (moves) / N_PHASES

// move ordering evaluation function
constexpr int MAX_EVALUATE_IDX_MO = 59049; // 3^10: up to 10 cells for pattern
constexpr int N_PATTERNS_MO_END = 4; // only 4 patterns are used for move ordering end
constexpr int N_PATTERN_FEATURES_MO_END = 16; // 16 features are used for move ordering end

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
constexpr int STEP = 32; // 1 disc = 32
constexpr int STEP_2 = 16; // STEP / 2

// constexpr int STEP_MO_END = 32; // 1 disc = 64
// constexpr int STEP_2_MO_END = 16; // STEP / 2

/*
    @brief 3 ^ N definition
*/
constexpr int PNO = 0;
constexpr int P30 = 1;
constexpr int P31 = 3;
constexpr int P32 = 9;
constexpr int P33 = 27;
constexpr int P34 = 81;
constexpr int P35 = 243;
constexpr int P36 = 729;
constexpr int P37 = 2187;
constexpr int P38 = 6561;
constexpr int P39 = 19683;
constexpr int P310 = 59049;

/*
    @brief coordinate definition
*/
constexpr int COORD_A1 = 63;
constexpr int COORD_B1 = 62;
constexpr int COORD_C1 = 61;
constexpr int COORD_D1 = 60;
constexpr int COORD_E1 = 59;
constexpr int COORD_F1 = 58;
constexpr int COORD_G1 = 57;
constexpr int COORD_H1 = 56;

constexpr int COORD_A2 = 55;
constexpr int COORD_B2 = 54;
constexpr int COORD_C2 = 53;
constexpr int COORD_D2 = 52;
constexpr int COORD_E2 = 51;
constexpr int COORD_F2 = 50;
constexpr int COORD_G2 = 49;
constexpr int COORD_H2 = 48;

constexpr int COORD_A3 = 47;
constexpr int COORD_B3 = 46;
constexpr int COORD_C3 = 45;
constexpr int COORD_D3 = 44;
constexpr int COORD_E3 = 43;
constexpr int COORD_F3 = 42;
constexpr int COORD_G3 = 41;
constexpr int COORD_H3 = 40;

constexpr int COORD_A4 = 39;
constexpr int COORD_B4 = 38;
constexpr int COORD_C4 = 37;
constexpr int COORD_D4 = 36;
constexpr int COORD_E4 = 35;
constexpr int COORD_F4 = 34;
constexpr int COORD_G4 = 33;
constexpr int COORD_H4 = 32;

constexpr int COORD_A5 = 31;
constexpr int COORD_B5 = 30;
constexpr int COORD_C5 = 29;
constexpr int COORD_D5 = 28;
constexpr int COORD_E5 = 27;
constexpr int COORD_F5 = 26;
constexpr int COORD_G5 = 25;
constexpr int COORD_H5 = 24;

constexpr int COORD_A6 = 23;
constexpr int COORD_B6 = 22;
constexpr int COORD_C6 = 21;
constexpr int COORD_D6 = 20;
constexpr int COORD_E6 = 19;
constexpr int COORD_F6 = 18;
constexpr int COORD_G6 = 17;
constexpr int COORD_H6 = 16;

constexpr int COORD_A7 = 15;
constexpr int COORD_B7 = 14;
constexpr int COORD_C7 = 13;
constexpr int COORD_D7 = 12;
constexpr int COORD_E7 = 11;
constexpr int COORD_F7 = 10;
constexpr int COORD_G7 = 9;
constexpr int COORD_H7 = 8;

constexpr int COORD_A8 = 7;
constexpr int COORD_B8 = 6;
constexpr int COORD_C8 = 5;
constexpr int COORD_D8 = 4;
constexpr int COORD_E8 = 3;
constexpr int COORD_F8 = 2;
constexpr int COORD_G8 = 1;
constexpr int COORD_H8 = 0;

constexpr int COORD_NO = 64;

constexpr int N_ZEROS_PLUS = 1 << 12; // for egev2 compression, 4096

/*
    @brief definition of patterns in evaluation function

    pattern -> coordinate

    @param n_cells              number of cells included in the pattern
    @param cells                coordinates of each cell
*/
struct Feature_to_coord {
    uint_fast8_t n_cells;
    uint_fast8_t cells[MAX_PATTERN_CELLS];
};

/*
    @brief definition of patterns in evaluation function

    coordinate -> pattern

    @param feature              the index of feature
    @param x                    the offset value of the cell in this feature
*/
struct Coord_feature {
    uint_fast8_t feature;
    uint_fast16_t x;
};

/*
    @brief definition of patterns in evaluation function

    coordinate -> all patterns

    @param n_features           number of features the cell is used by
    @param features             information for each feature
*/
struct Coord_to_feature {
    uint_fast8_t n_features;
    Coord_feature features[MAX_CELL_PATTERNS];
};

#if defined(USE_NNUE_EVALUATION)
#ifndef EVAL_NNUE_MAX_FT_DIM
#define EVAL_NNUE_MAX_FT_DIM 128
#endif
#ifndef EVAL_NNUE_MAX_PATTERN_FEATURES
#define EVAL_NNUE_MAX_PATTERN_FEATURES 65
#endif

struct alignas(32) Eval_search {
    int16_t accumulator[HW2 - 4][2][EVAL_NNUE_MAX_FT_DIM];
    uint16_t pattern_features[HW2 - 4][2][EVAL_NNUE_MAX_PATTERN_FEATURES];
    uint8_t nnue_phase[HW2 - 4];
    uint_fast8_t feature_idx;
};
#elif USE_SIMD
union Eval_features {
    __m256i f256[N_EVAL_VECTORS];
    __m128i f128[N_EVAL_VECTORS * 2];
};

struct Eval_search {
    Eval_features features[HW2 - 4];
#ifdef EGAROUCID_NNUE_OWN_LANES  // with the NNUE patterns the features are the lanes
    // NNUE instance indices in fixed colors and occupancy, kept with the same feature_idx. They are
    // computed only when a network evaluation needs them; each move just records itself.
    // stored as nnue::Lanes, the type they are read and written as (an Eval_features accessed through a
    // Lanes pointer broke strict aliasing: PGO builds searched different trees)
    nnue::Lanes nnue_lanes[HW2 - 4];
    nnue::Lanes nnue_occ[HW2 - 4];
    uint64_t nnue_flip[HW2 - 4];        // move into slot t: flipped discs, square, mover is not the root side
    uint8_t nnue_pos[HW2 - 4];
    uint8_t nnue_mover_other[HW2 - 4];
    uint8_t nnue_valid[HW2 - 4];        // lanes of slot t are computed
    uint8_t nnue_pass_parity;           // passes since the features were built, mod 2
#endif
    uint_fast8_t feature_idx;
};
#else
struct Eval_search {
    uint_fast16_t features[HW2 - 4][N_PATTERN_FEATURES];
    bool reversed[HW2 - 4];
    uint_fast8_t feature_idx;
};
#endif

/*
    @brief copy the evaluation state of a search that starts a parallel split

    Only the slots 0 .. feature_idx are live: a move writes slot feature_idx + 1 before anything reads it,
    and the NNUE lanes of a slot are rebuilt from the last computed slot at or below it. The slots above
    are not copied (most of a 24 KB Search with the NNUE lanes at a typical split ply).

    @param dst                  destination
    @param src                  source
*/
inline void copy_eval_search_live(Eval_search *dst, const Eval_search &src) {
    const size_t n = (size_t)src.feature_idx + 1;
#if defined(USE_NNUE_EVALUATION)
    std::memcpy(dst->accumulator, src.accumulator, n * sizeof(src.accumulator[0]));
    std::memcpy(dst->pattern_features, src.pattern_features, n * sizeof(src.pattern_features[0]));
    std::memcpy(dst->nnue_phase, src.nnue_phase, n * sizeof(src.nnue_phase[0]));
#elif USE_SIMD
    std::memcpy(dst->features, src.features, n * sizeof(src.features[0]));
#ifdef EGAROUCID_NNUE_OWN_LANES
    std::memcpy(dst->nnue_lanes, src.nnue_lanes, n * sizeof(src.nnue_lanes[0]));
    std::memcpy(dst->nnue_occ, src.nnue_occ, n * sizeof(src.nnue_occ[0]));
    std::memcpy(dst->nnue_flip, src.nnue_flip, n * sizeof(src.nnue_flip[0]));
    std::memcpy(dst->nnue_pos, src.nnue_pos, n * sizeof(src.nnue_pos[0]));
    std::memcpy(dst->nnue_mover_other, src.nnue_mover_other, n * sizeof(src.nnue_mover_other[0]));
    std::memcpy(dst->nnue_valid, src.nnue_valid, n * sizeof(src.nnue_valid[0]));
    dst->nnue_pass_parity = src.nnue_pass_parity;
#endif
#else
    std::memcpy(dst->features, src.features, n * sizeof(src.features[0]));
    std::memcpy(dst->reversed, src.reversed, n * sizeof(src.reversed[0]));
#endif
    dst->feature_idx = src.feature_idx;
#if defined(EGAROUCID_VALIDATE_SPLIT_COPY)
    // poison the slots that were not copied: a search that read them would go wrong or crash
#if defined(USE_NNUE_EVALUATION)
    std::memset(&dst->accumulator[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->accumulator[0]));
    std::memset(&dst->pattern_features[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->pattern_features[0]));
#elif USE_SIMD
    std::memset(&dst->features[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->features[0]));
#ifdef EGAROUCID_NNUE_OWN_LANES
    std::memset(&dst->nnue_lanes[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->nnue_lanes[0]));
    std::memset(&dst->nnue_occ[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->nnue_occ[0]));
    std::memset(&dst->nnue_flip[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->nnue_flip[0]));
    std::memset(&dst->nnue_valid[n], 1, (HW2 - 4 - n) * sizeof(dst->nnue_valid[0]));  // claims stale lanes are computed
#endif
#else
    std::memset(&dst->features[n], 0xCD, (HW2 - 4 - n) * sizeof(dst->features[0]));
#endif
#endif
}

/*
    @brief constants of 3 ^ N
*/
constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief evaluation function for game over

    @param b                    board
    @return final score
*/
inline int end_evaluate(Board *b) {
    return b->score_player();
}

/*
    @brief evaluation function for game over

    @param b                    board
    @param e                    number of empty squares
    @return final score
*/
inline int end_evaluate(Board *b, int e) {
    int score = b->count_player() * 2 + e;
    score += (((score >> 6) & 1) + (((score + HW2_M1) >> 7) & 1) - 1) * e;
    return score - HW2;
}

/*
    @brief evaluation function for game over (odd empties)

    @param b                    board
    @param e                    number of empty squares
    @return final score
*/
inline int end_evaluate_odd(Board *b, int e) {
    int score = b->count_player() * 2 + e;
    score += (((score >> 5) & 2) - 1) * e;
    return score - HW2;
}

inline std::vector<int16_t> load_unzip_egev2(const char* file, bool show_log, bool *failed) {
    *failed = false;
    std::vector<int16_t> res;
    FILE* fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        *failed = true;
        return res;
    }
    int n_unzipped_params = -1;
    if (fread(&n_unzipped_params, 4, 1, fp) < 1) {
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        *failed = true;
        return res;
    }
    if (show_log) {
        std::cerr << n_unzipped_params << " elems found in " << file << std::endl;
    }
    short *unzipped_params = (short*)malloc(sizeof(short) * n_unzipped_params);
    if (fread(unzipped_params, 2, n_unzipped_params, fp) < n_unzipped_params) {
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        free(unzipped_params);
        *failed = true;
        return res;
    }
    for (int i = 0; i < n_unzipped_params; ++i) {
        if (unzipped_params[i] >= N_ZEROS_PLUS) {
            for (int j = 0; j < unzipped_params[i] - N_ZEROS_PLUS; ++j) {
                res.emplace_back(0);
            }
        } else{
            res.emplace_back(unzipped_params[i]);
        }
    }
    if (show_log) {
        std::cerr << res.size() << " elems found in unzipped " << file << std::endl;
    }
    free(unzipped_params);
    return res;
}
