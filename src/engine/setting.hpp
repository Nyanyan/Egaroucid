/*
    Egaroucid Project

    @file setting.hpp
        Main settings of Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

/* Compile Options
    -DHAS_AVX512        : Use AVX-512
    -DHAS_NO_AVX2       : no AVX2
    -DHAS_ARM_PROCESSOR : ARM Processor
    -DHAS_AMD_PROCESSOR : Optimization for AMD CPU
    -DHAS_32_BIT_OS     : 32bit environment
*/

#pragma once
#include <string>

/*
    @brief Option
*/
// use SIMD
#ifndef HAS_NO_AVX2 // SIMD build
    #define USE_SIMD true
    // #define HAS_AVX512
    #ifdef HAS_AVX512 // AVX512 build
        #define USE_AVX512 true
    #endif
#endif

// use ARM
#ifdef HAS_ARM_PROCESSOR
    #define USE_ARM true
#endif

// optimize for AMD processors
#ifdef HAS_AMD_PROCESSOR
    #define USE_AMD true
#endif

#ifndef HAS_32_BIT_OS
    #ifdef _WIN32
        #ifdef _WIN64
            #define USE_64_BIT true
        #else
            #define USE_64_BIT false
        #endif
    #else
        #define USE_64_BIT true
    #endif
#endif




/*
@brief performance settings
*/

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

        #if USE_AVX512
            #define USE_AVX512_STABILITY true
        #endif

        // CRC32C Hash
        #if IS_GGS_TOURNAMENT
            #define USE_CRC32C_HASH true
            #define USE_CRC32C_HASH_LTT true
        #else
            #define USE_CRC32C_HASH false
            #define USE_CRC32C_HASH_LTT true
        #endif

        // TT init with SIMD
        #define USE_SIMD_TT_INIT false

    #endif
#else
    // NTZ
    #define USE_MINUS_NTZ false
#endif





/*
    @brief search settings
*/
// parity ordering
#define USE_END_PO true

// stability cut
#define USE_END_SC true
#define USE_LAST4_SC false

// enhanced transposition cutoff
#define USE_MID_ETC true

// Multi-ProbCut
#define USE_MID_MPC true

// last parity ordering optimization
#define LAST_PO_OPTIMIZE true

// use parallel clog search
#define USE_PARALLEL_CLOG_SEARCH true

// use only Dim0 evaluation
#define USE_DIM0_ONLY_EVALUATION true

// MPC pre calculation
#define USE_MPC_PRE_CALCULATION true

// YBWC
#define USE_YBWC_NWS true
#define USE_YBWC_NEGASCOUT true
#define USE_YBWC_NEGASCOUT_ANALYZE false

// aspiration search in negascout
#define USE_ASPIRATION_NEGASCOUT false

// Hash level setting
#define USE_CHANGEABLE_HASH_LEVEL true

// transposition table stack / heap
// if false, USE_CHANGEABLE_HASH_LEVEL must be true
#define TT_USE_STACK true

// flip SIMD / AVX512 optimization for each compiler
#define AUTO_FLIP_OPT_BY_COMPILER true

// Root Lazy SMP search
#define USE_LAZY_SMP true

// YBWC splitted task termination (less idoling, more nodes)
#ifndef USE_YBWC_SPLITTED_TASK_TERMINATION
#define USE_YBWC_SPLITTED_TASK_TERMINATION false
#endif

// last flip pass optimization
#define LAST_FLIP_PASS_OPT true

// killer move in move ordering
#define USE_KILLER_MOVE_MO false
#define USE_KILLER_MOVE_NWS_MO false



/*
    @brief debug settings
*/

// search statistics
#define USE_SEARCH_STATISTICS false

// Local TT statistics
#define USE_LOCAL_TT_STATISTICS false

// thread monitor
#define USE_THREAD_MONITOR false

// YBWC split statistics
#ifndef USE_YBWC_SPLIT_STATISTICS
#define USE_YBWC_SPLIT_STATISTICS false
#endif

// Endgame YBWC tuning.  These remain externally overridable so cold-TT A/B
// binaries can differ by one scheduling factor without editing the source.
#ifndef YBWC_END_SPLIT_MIN_DEPTH
#define YBWC_END_SPLIT_MIN_DEPTH 15
#endif
#ifndef YBWC_SELECTIVE_END_SPLIT_MIN_DEPTH
// The tournament's first endgame pass is 74%.  Higher-selectivity and exact
// searches retain the established split depth.
#define YBWC_SELECTIVE_END_SPLIT_MIN_DEPTH 18
#endif
#ifndef YBWC_END_MAX_SPLIT_TASKS
#define YBWC_END_MAX_SPLIT_TASKS THREAD_SIZE_INF
#endif
// Zero preserves the depth-dependent defaults in ybwc.hpp.  Values 1-3 use a
// fixed minimum number of remaining younger moves at every endgame split.
#ifndef YBWC_END_MIN_REMAINING_MOVES
#define YBWC_END_MIN_REMAINING_MOVES 0
#endif
#ifndef YBWC_END_WAIT_HELP
#define YBWC_END_WAIT_HELP false
#endif

// Search the moves used by the GGS ambiguity probe concurrently.  Keep this
// externally overridable so the scheduling policy can be A/B benchmarked.
#ifndef USE_AI_GET_VALUES_CANDIDATE_PARALLEL
#define USE_AI_GET_VALUES_CANDIDATE_PARALLEL true
#endif

// Poll split cancellation inside the specialized endgame NWS.  A sparse poll
// avoids paying an atomic load at every endgame node.
#ifndef USE_END_NWS_CANCELLATION_POLLING
#define USE_END_NWS_CANCELLATION_POLLING true
#endif
#ifndef END_NWS_CANCELLATION_POLL_MASK
#define END_NWS_CANCELLATION_POLL_MASK 1023
#endif

// Tree-parallel midgame search scales poorly beyond this many in-flight split
// tasks on the tournament workload. Exact searches keep the full pool.
#if IS_GGS_TOURNAMENT
    #ifndef YBWC_MID_MAX_SPLIT_TASKS
        #define YBWC_MID_MAX_SPLIT_TASKS 9
    #endif
#else
    #define YBWC_MID_MAX_SPLIT_TASKS THREAD_SIZE_INF
#endif

/*
    @brief tuning
*/

// move ordering
#define TUNE_MOVE_ORDERING false

// probcut
#define TUNE_PROBCUT_MID false
#define TUNE_PROBCUT_END false

// local strategy
#define TUNE_LOCAL_STRATEGY false


/*
    @brief test
*/

// endgame accuracy test
#define TEST_ENDGAME_ACCURACY false




/*
    @brief path definition
*/
#ifdef __APPLE__
    #ifdef GUI_BUILD
        #include <Siv3D.hpp>
        const std::string EXE_DIRECTORY_PATH = FileSystem::RelativePath(Resource(U"")).narrow();
    #else
        const std::string EXE_DIRECTORY_PATH = "./";
    #endif
#elif __linux__
    #include <filesystem>
    #include <linux/limits.h>
    #include <unistd.h>
    inline std::string get_exe_directory_path_linux() {
        char raw_path[PATH_MAX + 1] = {};
        const ssize_t len = readlink("/proc/self/exe", raw_path, PATH_MAX);
        if (len <= 0) {
            return "./";
        }
        raw_path[len] = '\0';
        std::filesystem::path binary_path(raw_path);
        return binary_path.parent_path().string() + "/";
    }
    const std::string EXE_DIRECTORY_PATH = get_exe_directory_path_linux();
#else // Windows
    const std::string EXE_DIRECTORY_PATH = "./";
#endif
