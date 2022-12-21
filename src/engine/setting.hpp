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

// use ARM
#define USE_ARM false

#if USE_SIMD
    #if USE_ARM
        // vertical mirror
        #define USE_FAST_VERTICAL_MIRROR true

        // pop_count
        #define USE_BUILTIN_POPCOUNT true

        // NTZ
        #define USE_MINUS_NTZ false

        // use SIMD in evaluation (pattern) function
        #define USE_SIMD_EVALUATION true
    #else
        // vertical mirror
        #define USE_FAST_VERTICAL_MIRROR true

        // pop_count
        #define USE_BUILTIN_POPCOUNT true

        // NTZ
        #define USE_BUILTIN_NTZ true

        // next bit
        #define USE_FAST_NEXT_BIT true

        // use SIMD in evaluation (pattern) function
        #define USE_SIMD_EVALUATION true

        // use bit gather optimization
        #define USE_BIT_GATHER_OPTIMIZE true

        // use fast join_h_line
        #define USE_FAST_JOIN_H_LINE true
    #endif
#else
    // NTZ
    #define USE_MINUS_NTZ false
#endif

// last parity ordering optimization
#define LAST_PO_OPTIMIZE true

// use parallel clog search
#define USE_PARALLEL_CLOG_SEARCH true

// MPC pre calculation
#define USE_MPC_PRE_CALCULATION true


/*
    @brief search settings
*/
// parity ordering
#define USE_END_PO true

// stability cut
#define USE_END_SC true
#define USE_LAST4_SC false

// enhanced enhanced transposition cutoff
#define USE_MID_ETC false

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
