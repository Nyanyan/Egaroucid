/*
    Egaroucid Project

    @file setting.hpp
        Main settings of Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

/*
    @brief performance settings
*/
// use SIMD
#define USE_SIMD true

// vertical mirror
#define USE_FAST_VERTICAL_MIRROR true

// pop_count
#define USE_BUILTIN_POPCOUNT true

// NTZ
#define USE_BUILTIN_NTZ true
#define USE_MINUS_NTZ false

// last parity ordering optimization
#define LAST_PO_OPTIMIZE true

// use parallel clog search
#define USE_PARALLEL_CLOG_SEARCH true

// use SIMD in evaluation (pattern) function
#define USE_SIMD_EVALUATION true

// use bit gather optimization
#define USE_BIT_GATHER_OPTIMIZE true


/*
    @brief search settings
*/
// parity ordering
#define USE_END_PO true

// stability cut
#define USE_END_SC true

// Multi-ProbCut
#define USE_MID_MPC true

// Null Move Pruning
#define USE_MID_NMP false

// use probcut to predict it seems to be an all node
#define USE_ALL_NODE_PREDICTION false

// use search algs
#define USE_NEGA_ALPHA_ORDERING false
#define USE_NEGA_ALPHA_END false
#define USE_NEGA_ALPHA_END_FAST false

// transposition table
#define USE_TRANSPOSITION_TABLE true

/*
    @brief debug settings
*/
// use SIMD in evaluation (pattern) function
#define USE_SIMD_DEBUG false

// evaluation harness
#define USE_EVALUATION_HARNESS false

// search statistics
#define USE_SEARCH_STATISTICS false

/*
    @brief tuning
*/
// move ordering
#define TUNE_MOVE_ORDERING_END false
