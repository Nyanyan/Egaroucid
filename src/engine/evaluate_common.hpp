/*
    Egaroucid Project

    @file evaluate_common.hpp
        Common things of evaluation function
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "board.hpp"

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

#if USE_SIMD
union Eval_features {
    __m256i f256[N_EVAL_VECTORS];
    __m128i f128[N_EVAL_VECTORS * 2];
};

struct Eval_search {
    Eval_features features[HW2 - 4];
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