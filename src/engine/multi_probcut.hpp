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
constexpr int MPC_DEPTH_NUMERATOR = 2;
constexpr int MPC_DEPTH_DENOMINATOR = 5;
#ifndef MPC_SIGMA_SCALE
    #define MPC_SIGMA_SCALE 1.0
#endif
#ifndef MPC_PROBCUT_G_OFFSET
    #define MPC_PROBCUT_G_OFFSET 0.0
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
    int error_0;
    if constexpr (IsEndSearch) {
        *error_search = mpc_error_end[mpc_level][n_discs][search_depth];
        error_0 = mpc_error_end[mpc_level][n_discs][0];
    } else {
        *error_search = mpc_error[mpc_level][n_discs][search_depth][depth];
        error_0 = mpc_error[mpc_level][n_discs][0][depth];
    }
    *eval_error = (error_0 + *error_search + 1) / 2;
#else
    const double mpct = SELECTIVITY_MPCT[mpc_level];
    double sigma_search;
    double sigma_0;
    if constexpr (IsEndSearch) {
        sigma_search = probcut_sigma_end(n_discs, search_depth);
        sigma_0 = probcut_sigma_end(n_discs, 0);
    } else {
        sigma_search = probcut_sigma(n_discs, search_depth, depth);
        sigma_0 = probcut_sigma(n_discs, 0, depth);
    }
    *error_search = ceil(MPC_ERROR_SCALE * mpct * sigma_search);
    *eval_error = ceil(MPC_ERROR_SCALE * mpct * 0.5 * (sigma_0 + sigma_search));
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
template<bool IsEndSearch>
inline bool mpc_impl(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, std::vector<bool*> &searchings) {
    int search_depth = ((depth * MPC_DEPTH_NUMERATOR / MPC_DEPTH_DENOMINATOR) & 0b11111110) + (depth & 1);
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
    }

    if (search_depth == 0) {
        int static_error = mpc_static_error<IsEndSearch>(search->mpc_level, search->n_discs, depth);
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
        uint_fast8_t mpc_level = search->mpc_level;
        int error_search, eval_error;
        mpc_search_errors<IsEndSearch>(mpc_level, search->n_discs, search_depth, depth, &error_search, &eval_error);
        // if (IsEndSearch) {
        //     error_search += 1.5;
        // }
        search->mpc_level = MPC_100_LEVEL;
#if !USE_DIM0_ONLY_EVALUATION
        const bool saved_use_dim0_mpc_eval = search->use_dim0_mpc_eval;
        search->use_dim0_mpc_eval = use_dim0_mpc_eval;
#endif
        if (d0value >= beta - eval_error) {
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
        if (d0value <= alpha + eval_error) {
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
    std::vector<bool*> searchings = {searching};
    return mpc_mid(search, alpha, beta, depth, legal, v, searchings);
}

inline bool mpc_end(Search* search, int alpha, int beta, int depth, uint64_t legal, int* v, bool *searching) {
    std::vector<bool*> searchings = {searching};
    return mpc_end(search, alpha, beta, depth, legal, v, searchings);
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
