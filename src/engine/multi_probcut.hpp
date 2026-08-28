/*
    Egaroucid Project

    @file multi_probcut.hpp
        MPC (Multi-ProbCut)
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "util.hpp"

constexpr int USE_MPC_MIN_DEPTH = 3;

//constexpr int MPC_ADD_DEPTH_VALUE_THRESHOLD = 5;
//constexpr int MPC_SUB_DEPTH_VALUE_THRESHOLD = 20;
constexpr double MPC_ERROR_SCALE = 1.0;
constexpr int MPC_ERROR0_OFFSET = 3;
constexpr int MPC_DEPTH_NUMERATOR = 2;
constexpr int MPC_DEPTH_DENOMINATOR = 5;
#ifndef MPC_SIGMA_SCALE
    #define MPC_SIGMA_SCALE 1.0
#endif
#ifndef MPC_PROBCUT_G_OFFSET
    #define MPC_PROBCUT_G_OFFSET 0.3
#endif

// constants from standard normal distribution table
// two-sided test                                         74.0  88.0  93.0  98.0  99.0  99.9 100 (%)
constexpr double SELECTIVITY_MPCT[N_SELECTIVITY_LEVEL] = {1.13, 1.55, 1.81, 2.32, 2.57, 3.29, 9.99};

/*
    @brief constants for ProbCut error calculation
*/
constexpr double probcut_a = 0.8335834703936896;
constexpr double probcut_b = -4.71778909968251;
constexpr double probcut_c = 1.1467905781538477;
constexpr double probcut_d = -0.5274699259330169;
constexpr double probcut_e = 6.5091001393587335;
constexpr double probcut_f = 3.9546352081550378;
constexpr double probcut_g = 1.5719077939546169 + MPC_PROBCUT_G_OFFSET;

constexpr double probcut_end_a = -1.3182333120273682;
constexpr double probcut_end_b = -6.99290557735024;
constexpr double probcut_end_c = -0.05280654146244756;
constexpr double probcut_end_d = 0.48284187178125065;
constexpr double probcut_end_e = 5.289589936037036;
constexpr double probcut_end_f = 11.940601436361513;

/*
    Recalibrated endgame MPC model.  The tables were fitted from exact-labelled
    search contexts and are intentionally restricted to their covered range.
    Outside depth 10--18 and selectivity 74--93, the legacy model is used.

    The shallow depth candidates were also measured at +2 plies.  Although the
    deeper probes reduced node counts, they reduced parallel NPS enough to lose
    wall time, so this table keeps the faster original parity-preserving depths.
*/
constexpr int END_MPC_MODEL_MIN_DEPTH = 10;
constexpr int END_MPC_MODEL_MAX_DEPTH = 18;
constexpr int END_MPC_MODEL_SIZE = END_MPC_MODEL_MAX_DEPTH - END_MPC_MODEL_MIN_DEPTH + 1;

#ifndef END_MPC_USE_RECALIBRATED_SHALLOW
    #define END_MPC_USE_RECALIBRATED_SHALLOW 1
#endif
#ifndef END_MPC_USE_STATIC_EVAL_CUT
    #define END_MPC_USE_STATIC_EVAL_CUT 1
#endif
#ifndef END_MPC_SHALLOW_GATE_SLACK_VALUE
    #define END_MPC_SHALLOW_GATE_SLACK_VALUE 4
#endif

constexpr double END_MPC_SHALLOW_CUSHION = 1.10;
constexpr int END_MPC_SHALLOW_GATE_SLACK = END_MPC_SHALLOW_GATE_SLACK_VALUE;
constexpr int END_MPC_SHALLOW_DEPTH[END_MPC_MODEL_SIZE] = {
    4, 5, 4, 5, 4, 7, 6, 7, 6
};
constexpr double END_MPC_SHALLOW_SIGMA[END_MPC_MODEL_SIZE] = {
    4.784937620133198, 4.404753276200285, 5.2572064673113585,
    4.228319028081452, 5.128290894292946, 3.8714224779536655,
    4.498944083748551, 3.5529833917572193, 3.9725633191799896
};
constexpr double END_MPC_SHALLOW_LOWER_TAIL[3] = {
    1.0550310553705888, 1.4922446086063115, 1.7959223408965452
};
constexpr double END_MPC_SHALLOW_UPPER_TAIL[3] = {
    1.0859804512675133, 1.5634343141830378, 1.8446710563924853
};
// ceil(cushion * directional_tail * sigma), precomputed for the hot path.
constexpr int END_MPC_SHALLOW_ERROR_HIGH[3][END_MPC_MODEL_SIZE] = {
    {6, 6, 7, 5, 6, 5, 6, 5, 5},
    {8, 8, 9, 7, 9, 7, 8, 6, 7},
    {10, 9, 11, 9, 11, 8, 9, 8, 8}
};
constexpr int END_MPC_SHALLOW_ERROR_LOW[3][END_MPC_MODEL_SIZE] = {
    {6, 6, 7, 6, 7, 5, 6, 5, 5},
    {9, 8, 10, 8, 9, 7, 8, 7, 7},
    {10, 9, 11, 9, 11, 8, 10, 8, 9}
};

// The static model predicts exact_value = static_eval + bias.  Its independent
// 99% tails are deliberately more conservative than the requested selectivity.
constexpr double END_MPC_STATIC_BIAS = 3.2015961138098543;
constexpr double END_MPC_STATIC_CUSHION = 1.05;
constexpr double END_MPC_STATIC_SIGMA[END_MPC_MODEL_SIZE] = {
    7.451537457705234, 7.716189916857538, 7.9373539190235975,
    8.298192366205916, 8.207674719706846, 7.795496183403784,
    7.969549491857762, 7.768698310940474, 7.755365107145744
};
constexpr double END_MPC_STATIC_LOWER_TAIL = 2.6966500549415615;
constexpr double END_MPC_STATIC_UPPER_TAIL = 3.0400938484780884;
constexpr int END_MPC_STATIC_HIGH_OFFSET[END_MPC_MODEL_SIZE] = {
    18, 19, 20, 21, 21, 19, 20, 19, 19
};
constexpr int END_MPC_STATIC_LOW_OFFSET[END_MPC_MODEL_SIZE] = {
    -27, -28, -29, -30, -30, -29, -29, -29, -28
};

inline bool use_recalibrated_end_mpc(uint_fast8_t mpc_level, int depth) {
    return
        mpc_level <= MPC_93_LEVEL &&
        END_MPC_MODEL_MIN_DEPTH <= depth && depth <= END_MPC_MODEL_MAX_DEPTH;
}

inline int end_mpc_shallow_depth(int depth) {
    return END_MPC_SHALLOW_DEPTH[depth - END_MPC_MODEL_MIN_DEPTH];
}

inline int end_mpc_shallow_error(uint_fast8_t mpc_level, int depth, bool high) {
    const int index = depth - END_MPC_MODEL_MIN_DEPTH;
    return high
        ? END_MPC_SHALLOW_ERROR_HIGH[mpc_level][index]
        : END_MPC_SHALLOW_ERROR_LOW[mpc_level][index];
}

inline int end_mpc_static_threshold(int depth, int boundary, bool high) {
    const int index = depth - END_MPC_MODEL_MIN_DEPTH;
    return boundary + (high
        ? END_MPC_STATIC_HIGH_OFFSET[index]
        : END_MPC_STATIC_LOW_OFFSET[index]);
}

#if defined(END_PROBCUT_CONTEXT_TRACE)
inline thread_local int end_probcut_context_trace_remaining = 0;
inline thread_local int end_probcut_context_trace_min_depth = 0;
inline thread_local int end_probcut_context_trace_max_depth = HW2;
inline thread_local int end_probcut_context_trace_per_depth_remaining[HW2 + 1] = {};

inline void end_probcut_trace_context(
    const Search *search,
    int deep_depth,
    int shallow_depth,
    int alpha,
    int beta,
    const char *direction,
    int d0_value,
    int legal_count,
    int threshold,
    bool gate_passed,
    bool static_probe,
    uint_fast8_t mpc_level
) {
    if (
        end_probcut_context_trace_remaining <= 0 ||
        deep_depth < end_probcut_context_trace_min_depth ||
        deep_depth > end_probcut_context_trace_max_depth ||
        end_probcut_context_trace_per_depth_remaining[deep_depth] <= 0
    ) {
        return;
    }
    --end_probcut_context_trace_remaining;
    --end_probcut_context_trace_per_depth_remaining[deep_depth];
    std::cout
        << "END_PROBCUT_CONTEXT_V2\t"
        << search->board.to_str() << '\t'
        << deep_depth << '\t'
        << shallow_depth << '\t'
        << static_cast<int>(search->n_discs) << '\t'
        << static_cast<int>(search->n_discs - search->root_n_discs) << '\t'
        << alpha << '\t'
        << beta << '\t'
        << direction << '\t'
        << d0_value << '\t'
        << legal_count << '\t'
        << static_cast<int>(mpc_level) << '\t'
        << threshold << '\t'
        << static_cast<int>(gate_passed) << '\t'
        << static_cast<int>(static_probe)
        << '\n';
}
#endif


#if USE_MPC_PRE_CALCULATION
int mpc_error[N_SELECTIVITY_LEVEL][HW2 + 1][HW2 - 3][HW2 - 3];
int mpc_error_end[N_SELECTIVITY_LEVEL][HW2 + 1][HW2 - 3];
#endif

/*
    @brief ProbCut error calculation for midgame

    @param n_discs              number of discs on the board
    @param depth1               depth of shallow search
    @param depth2               depth of deep search
    @return expected error
*/
inline double probcut_sigma(int n_discs, int depth1, int depth2) {
    double res = probcut_a * ((double)n_discs / 64.0) + probcut_b * ((double)depth1 / 60.0) + probcut_c * ((double)depth2 / 60.0);
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return MPC_SIGMA_SCALE * res;
}

/*
    @brief ProbCut error calculation for endgame

    @param n_discs              number of discs on the board
    @param depth                depth of shallow search
    @return expected error
*/
inline double probcut_sigma_end(int n_discs, int depth) {
    double res = probcut_end_a * ((double)n_discs / 64.0) + probcut_end_b * ((double)depth / 60.0);
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f;
    return MPC_SIGMA_SCALE * res;
}

inline int probcut_error(uint_fast8_t mpc_level, double sigma) {
    return ceil(MPC_ERROR_SCALE * SELECTIVITY_MPCT[mpc_level] * sigma);
}

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, const bool is_end_search, std::vector<bool*> &searchings);
int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, const bool is_end_search, bool *searching);

inline bool mpc_end_static_eval_cut(
    Search *search,
    int alpha,
    int beta,
    int depth,
    int d0_value,
    uint64_t legal,
    int *v
) {
#if END_MPC_USE_STATIC_EVAL_CUT
    const int legal_count = pop_count_ull(legal);
    const int static_high_threshold = end_mpc_static_threshold(depth, beta, true);
    const int static_low_threshold = end_mpc_static_threshold(depth, alpha, false);
#if defined(END_PROBCUT_CONTEXT_TRACE)
    end_probcut_trace_context(
        search, depth, 0, alpha, beta, "high", d0_value, legal_count,
        static_high_threshold, d0_value >= static_high_threshold, true,
        search->mpc_level
    );
    end_probcut_trace_context(
        search, depth, 0, alpha, beta, "low", d0_value, legal_count,
        static_low_threshold, d0_value <= static_low_threshold, true,
        search->mpc_level
    );
#endif
    if (d0_value >= static_high_threshold) {
        *v = beta + (beta & 1);
        return true;
    }
    if (d0_value <= static_low_threshold) {
        *v = alpha - (alpha & 1);
        return true;
    }
#else
    (void)search;
    (void)alpha;
    (void)beta;
    (void)depth;
    (void)d0_value;
    (void)legal;
    (void)v;
#endif
    return false;
}

template<typename Searchings>
inline bool mpc_end_recalibrated_shallow(
    Search *search,
    int alpha,
    int beta,
    int depth,
    int d0_value,
    uint64_t legal,
    int *v,
    Searchings &searchings
) {
    const uint_fast8_t mpc_level = search->mpc_level;
    const int legal_count = pop_count_ull(legal);
    const int shallow_depth = end_mpc_shallow_depth(depth);
    const int high_threshold = beta + end_mpc_shallow_error(mpc_level, depth, true);
    const int low_threshold = alpha - end_mpc_shallow_error(mpc_level, depth, false);
    const bool high_gate = d0_value >= high_threshold - END_MPC_SHALLOW_GATE_SLACK;
    const bool low_gate = d0_value <= low_threshold + END_MPC_SHALLOW_GATE_SLACK;
#if defined(END_PROBCUT_CONTEXT_TRACE)
    end_probcut_trace_context(
        search, depth, shallow_depth, alpha, beta, "high", d0_value,
        legal_count, high_threshold, high_gate, false, mpc_level
    );
    end_probcut_trace_context(
        search, depth, shallow_depth, alpha, beta, "low", d0_value,
        legal_count, low_threshold, low_gate, false, mpc_level
    );
#endif
    if (!high_gate && !low_gate) {
        return false;
    }

    search->mpc_level = MPC_100_LEVEL;
#if !USE_DIM0_ONLY_EVALUATION
    const bool saved_use_dim0_mpc_eval = search->use_dim0_mpc_eval;
    search->use_dim0_mpc_eval = false;
#endif
    if (
        high_gate && high_threshold <= SCORE_MAX &&
        nega_alpha_ordering_nws(
            search, high_threshold - 1, shallow_depth, false, legal, false,
            searchings
        ) >= high_threshold
    ) {
        *v = beta + (beta & 1);
#if !USE_DIM0_ONLY_EVALUATION
        search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
        search->mpc_level = mpc_level;
        return true;
    }
    if (
        low_gate && low_threshold >= -SCORE_MAX &&
        nega_alpha_ordering_nws(
            search, low_threshold, shallow_depth, false, legal, false,
            searchings
        ) <= low_threshold
    ) {
        *v = alpha - (alpha & 1);
#if !USE_DIM0_ONLY_EVALUATION
        search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
        search->mpc_level = mpc_level;
        return true;
    }
#if !USE_DIM0_ONLY_EVALUATION
    search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
    search->mpc_level = mpc_level;
    return false;
}

template<bool IsEndSearch>
inline int mpc_static_error(uint_fast8_t mpc_level, int n_discs, int depth) {
#if USE_MPC_PRE_CALCULATION
    if constexpr (IsEndSearch) {
        return mpc_error_end[mpc_level][n_discs][0];
    } else {
        return mpc_error[mpc_level][n_discs][0][depth];
    }
#else
    const double mpct = SELECTIVITY_MPCT[mpc_level];
    if constexpr (IsEndSearch) {
        return ceil(MPC_ERROR_SCALE * mpct * probcut_sigma_end(n_discs, 0));
    } else {
        return ceil(MPC_ERROR_SCALE * mpct * probcut_sigma(n_discs, 0, depth));
    }
#endif
}

template<bool IsEndSearch>
inline void mpc_search_errors(uint_fast8_t mpc_level, int n_discs, int search_depth, int depth, int *error_search, int *eval_error) {
#if USE_MPC_PRE_CALCULATION
    if constexpr (IsEndSearch) {
        *error_search = mpc_error_end[mpc_level][n_discs][search_depth];
        if (eval_error) {
            int error_0 = mpc_error_end[mpc_level][n_discs][0];
            *eval_error = (error_0 + *error_search + 1) / 2;
        }
    } else {
        *error_search = mpc_error[mpc_level][n_discs][search_depth][depth];
        if (eval_error) {
            int error_0 = mpc_error[mpc_level][n_discs][0][depth];
            *eval_error = (error_0 + *error_search + 1) / 2;
        }
    }
#else
    const double mpct = SELECTIVITY_MPCT[mpc_level];
    double sigma_search;
    if constexpr (IsEndSearch) {
        sigma_search = probcut_sigma_end(n_discs, search_depth);
        if (eval_error) {
            double sigma_0 = probcut_sigma_end(n_discs, 0);
            *eval_error = ceil(MPC_ERROR_SCALE * mpct * 0.5 * (sigma_0 + sigma_search));
        }
    } else {
        sigma_search = probcut_sigma(n_discs, search_depth, depth);
        if (eval_error) {
            double sigma_0 = probcut_sigma(n_discs, 0, depth);
            *eval_error = ceil(MPC_ERROR_SCALE * mpct * 0.5 * (sigma_0 + sigma_search));
        }
    }
    *error_search = ceil(MPC_ERROR_SCALE * mpct * sigma_search);
#endif
}

/*
    @brief Multi-ProbCut for normal search

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                depth of deep search
    @param legal                for use of previously calculated legal bitboard
    @param v                    an integer to store result
    @param searching            flag for terminating this search
    @return cutoff occurred?
*/
template<bool IsEndSearch, typename Searchings>
inline bool mpc_impl(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, Searchings &searchings) {
    int search_depth = ((depth * MPC_DEPTH_NUMERATOR / MPC_DEPTH_DENOMINATOR) & 0b11111110) + (depth & 1);
    const uint_fast8_t mpc_level = search->mpc_level;
    // int search_depth = ((depth / 2) & 0b11111110) + (depth & 1); // depth / 2 + parity
#if USE_DIM0_ONLY_EVALUATION
    int d0value = mid_evaluate_diff(search);
#else
    const bool use_dim0_mpc_eval = eval_fm_enabled && eval_fm_use_dim0_mpc_search && !IsEndSearch;
    int d0value = use_dim0_mpc_eval ? mid_evaluate_dim0(search) : mid_evaluate_diff(search);
#endif
    /*
    if (alpha - MPC_ADD_DEPTH_VALUE_THRESHOLD < d0value && d0value < beta + MPC_ADD_DEPTH_VALUE_THRESHOLD && depth >= 20 && search_depth < depth - 2) {
        search_depth += 2; // if value is near [alpha, beta], increase search_depth
        //if (search_depth >= depth) {
        //    return false;
        //}
    }
    */
    /*
    if ((d0value < alpha - MPC_SUB_DEPTH_VALUE_THRESHOLD || beta + MPC_SUB_DEPTH_VALUE_THRESHOLD < d0value) && search_depth >= 2) {
        search_depth -= 2; // if value is far from [alpha, beta], decrease search_depth
    }
    */

    if constexpr (IsEndSearch) {
        if ((alpha & 1) == 0) {
            alpha += 1;
        }
        if ((beta & 1) == 0) {
            beta -= 1;
        }
        if (use_recalibrated_end_mpc(mpc_level, depth)) {
            if (mpc_end_static_eval_cut(
                search, alpha, beta, depth, d0value, legal, v
            )) {
                return true;
            }
#if END_MPC_USE_RECALIBRATED_SHALLOW
            return mpc_end_recalibrated_shallow(
                search, alpha, beta, depth, d0value, legal, v, searchings
            );
#endif
        }
    }

    if (search_depth == 0) {
        int static_error = mpc_static_error<IsEndSearch>(mpc_level, search->n_discs, depth);
#if defined(END_PROBCUT_CONTEXT_TRACE)
        if constexpr (IsEndSearch) {
            end_probcut_trace_context(
                search, depth, 0, alpha, beta, "high", d0value,
                pop_count_ull(legal), beta + static_error,
                d0value >= beta + static_error, true, mpc_level
            );
            end_probcut_trace_context(
                search, depth, 0, alpha, beta, "low", d0value,
                pop_count_ull(legal), alpha - static_error,
                d0value <= alpha - static_error, true, mpc_level
            );
        }
#endif
        if (d0value >= beta + static_error) {
            *v = beta;
            if constexpr (IsEndSearch) {
                *v += beta & 1;
            }
            return true;
        }
        if (d0value <= alpha - static_error) {
            *v = alpha;
            if constexpr (IsEndSearch) {
                *v -= alpha & 1;
            }
            return true;
        }
    } else {
        int error_search;
        mpc_search_errors<IsEndSearch>(mpc_level, search->n_discs, search_depth, depth, &error_search, nullptr);
        // if (IsEndSearch) {
        //     error_search += 1.5;
        // }
        int error_0 = std::max(1, error_search - MPC_ERROR0_OFFSET);
#if defined(END_PROBCUT_CONTEXT_TRACE)
        if constexpr (IsEndSearch) {
            end_probcut_trace_context(
                search, depth, search_depth, alpha, beta, "high", d0value,
                pop_count_ull(legal), beta + error_search,
                d0value >= beta + error_0, false, mpc_level
            );
            end_probcut_trace_context(
                search, depth, search_depth, alpha, beta, "low", d0value,
                pop_count_ull(legal), alpha - error_search,
                d0value <= alpha - error_0, false, mpc_level
            );
        }
#endif
        search->mpc_level = MPC_100_LEVEL;
#if !USE_DIM0_ONLY_EVALUATION
        const bool saved_use_dim0_mpc_eval = search->use_dim0_mpc_eval;
        search->use_dim0_mpc_eval = use_dim0_mpc_eval;
#endif
        if (d0value >= beta + error_0) {
            int pc_beta = beta + error_search;
            if (pc_beta <= SCORE_MAX) {
                if (nega_alpha_ordering_nws(search, pc_beta - 1, search_depth, false, legal, false, searchings) >= pc_beta) {
                    *v = beta;
                    if constexpr (IsEndSearch) {
                        *v += beta & 1;
                    }
#if !USE_DIM0_ONLY_EVALUATION
                    search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
                    search->mpc_level = mpc_level;
                    return true;
                }
            }
        }
        if (d0value <= alpha - error_0) {
            int pc_alpha = alpha - error_search;
            if (pc_alpha >= -SCORE_MAX) {
                if (nega_alpha_ordering_nws(search, pc_alpha, search_depth, false, legal, false, searchings) <= pc_alpha) {
                    *v = alpha;
                    if constexpr (IsEndSearch) {
                        *v -= alpha & 1;
                    }
#if !USE_DIM0_ONLY_EVALUATION
                    search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
                    search->mpc_level = mpc_level;
                    return true;
                }
            }
        }
#if !USE_DIM0_ONLY_EVALUATION
        search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
        search->mpc_level = mpc_level;
    }
    return false;
}

inline bool mpc_mid(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, std::vector<bool*> &searchings) {
    return mpc_impl<false>(search, alpha, beta, depth, legal, v, searchings);
}

inline bool mpc_end(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, std::vector<bool*> &searchings) {
    return mpc_impl<true>(search, alpha, beta, depth, legal, v, searchings);
}

inline bool mpc_mid(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, bool *searching) {
    return mpc_impl<false>(search, alpha, beta, depth, legal, v, searching);
}

inline bool mpc_end(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, bool *searching) {
    return mpc_impl<true>(search, alpha, beta, depth, legal, v, searching);
}


#if USE_ALL_NODE_PREDICTION_NWS
inline bool predict_all_node(Search* search, int alpha, int depth, uint64_t legal, const bool is_end_search, bool *searching) {
    uint_fast8_t mpc_level = MPC_93_LEVEL;
    int search_depth = mpc_search_depth_arr[is_end_search][depth];
    int error_search, error_0;
#if USE_MPC_PRE_CALCULATION
    if (is_end_search) {
        error_search = mpc_error_end[mpc_level][search->n_discs][search_depth];
        error_0 = mpc_error_end[mpc_level][search->n_discs][0];
    } else{
        error_search = mpc_error[mpc_level][search->n_discs][search_depth][depth];
        error_0 = mpc_error[mpc_level][search->n_discs][0][depth];
    }
#else
    double mpct = SELECTIVITY_MPCT[mpc_level];
    if (is_end_search) {
        error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
        error_0 = ceil(mpct * probcut_sigma_end(search->n_discs, 0));
    }else{
        error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
        error_0 = ceil(mpct * probcut_sigma(search->n_discs, 0, depth));
    }
#endif
#if USE_DIM0_ONLY_EVALUATION
    int d0value = mid_evaluate_diff(search);
#else
    const bool use_dim0_mpc_eval = eval_fm_enabled && eval_fm_use_dim0_mpc_search && !is_end_search;
    int d0value = use_dim0_mpc_eval ? mid_evaluate_dim0(search) : mid_evaluate_diff(search);
#endif
    if (d0value <= alpha - (error_search + error_0) / 2) {
        int pc_alpha = alpha - error_search;
        if (pc_alpha > -SCORE_MAX) {
#if !USE_DIM0_ONLY_EVALUATION
            const bool saved_use_dim0_mpc_eval = search->use_dim0_mpc_eval;
            search->use_dim0_mpc_eval = use_dim0_mpc_eval;
#endif
            if (nega_alpha_ordering_nws(search, pc_alpha, search_depth, false, legal, false, searching) <= pc_alpha) {
#if !USE_DIM0_ONLY_EVALUATION
                search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
                return true;
            }
#if !USE_DIM0_ONLY_EVALUATION
            search->use_dim0_mpc_eval = saved_use_dim0_mpc_eval;
#endif
        }
    }
    return false;
}
#endif



#if USE_MPC_PRE_CALCULATION
void mpc_init() {
    int mpc_level, n_discs, depth1, depth2;
    for (mpc_level = 0; mpc_level < N_SELECTIVITY_LEVEL; ++mpc_level) {
        for (n_discs = 0; n_discs < HW2 + 1; ++n_discs) {
            for (depth1 = 0; depth1 < HW2 - 3; ++depth1) {
                mpc_error_end[mpc_level][n_discs][depth1] = probcut_error(mpc_level, probcut_sigma_end(n_discs, depth1));
                for (depth2 = 0; depth2 < HW2 - 3; ++depth2) {
                    mpc_error[mpc_level][n_discs][depth1][depth2] = probcut_error(mpc_level, probcut_sigma(n_discs, depth1, depth2));
                }
            }
        }
    }
}
#endif

#if TUNE_PROBCUT_MID
inline Search_result tree_search_legal(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, thread_id_t thread_id, bool *searching);
void get_data_probcut_mid() {
    std::ofstream ofs("probcut_mid.txt");
    Board board;
    Flip flip;
    Search_result short_ans, long_ans;
    bool searching = true;
    for (int i = 0; i < 10000; ++i) {
        // for (int depth = 18; depth <= 18; ++depth) {
        for (int depth = 4; depth <= 14; ++depth) {
            for (int n_discs = 4; n_discs < HW2 - depth - 2; ++n_discs) {
                board.reset();
                for (int j = 4; j < n_discs && board.check_pass(); ++j) { // random move
                    uint64_t legal = board.get_legal();
                    int random_idx = myrandrange(0, pop_count_ull(legal));
                    int t = 0;
                    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                        if (t == random_idx) {
                            calc_flip(&flip, &board, cell);
                            break;
                        }
                        ++t;
                    }
                    board.move_board(&flip);
                }
                if (board.check_pass()) {
                    int short_depth = myrandrange(1, depth - 1);
                    short_depth &= 0xfffffffe;
                    short_depth |= depth & 1;
                    //int short_depth = mpc_search_depth_arr[0][depth];
                    if (short_depth == 0) {
                        short_ans.value = mid_evaluate(&board);
                    } else {
                        short_ans = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, short_depth, MPC_100_LEVEL, false, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                    }
                    long_ans = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, depth, MPC_100_LEVEL, false, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                    // n_discs short_depth long_depth error
                    std::cerr << i << " " << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                    ofs << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                }
            }
        }
    }
}
#endif

#if TUNE_PROBCUT_END
inline Search_result tree_search_legal(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, thread_id_t thread_id, bool *searching);
void get_data_probcut_end() {
    std::ofstream ofs("probcut_end.txt");
    Board board;
    Flip flip;
    Search_result short_ans, long_ans;
    bool searching = true;
    for (int i = 0; i < 10000; ++i) {
        for (int depth = 2; depth <= 24; ++depth) {
            board.reset();
            for (int j = 0; j < HW2 - 4 - depth && board.check_pass(); ++j) { // random move
                uint64_t legal = board.get_legal();
                int random_idx = myrandrange(0, pop_count_ull(legal));
                int t = 0;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                    if (t == random_idx) {
                        calc_flip(&flip, &board, cell);
                        break;
                    }
                    ++t;
                }
                board.move_board(&flip);
            }
            if (board.check_pass()) {
                int short_depth = myrandrange(2, std::min(18, depth - 1));
                short_depth &= 0xfffffffe;
                short_depth |= depth & 1;
                //int short_depth = mpc_search_depth_arr[1][depth];
                if (short_depth == 0) {
                    short_ans.value = mid_evaluate(&board);
                } else {
                    short_ans = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, short_depth, MPC_100_LEVEL, false, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                }
                long_ans = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, depth, MPC_100_LEVEL, false, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                // n_discs short_depth error
                std::cerr << i << " " << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
                ofs << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
            }
        }
    }
}
#endif
