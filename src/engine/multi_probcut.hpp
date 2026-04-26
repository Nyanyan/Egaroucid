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
constexpr int MPC_ERROR0_OFFSET = 4;

// constants from standard normal distribution table
// two-sided test                                         74.0  88.0  93.0  98.0  99.0  99.9 100 (%)
constexpr double SELECTIVITY_MPCT[N_SELECTIVITY_LEVEL] = {1.13, 1.55, 1.81, 2.32, 2.57, 3.29, 9.99};

/*
    @brief constants for ProbCut error calculation
*/
constexpr double probcut_a = 0.7308488452189136;
constexpr double probcut_b = -4.5708322989025865;
constexpr double probcut_c = 1.096319765006055;
constexpr double probcut_d = -0.8362251801219095;
constexpr double probcut_e = 4.610017383697701;
constexpr double probcut_f = 3.818582623595395;
constexpr double probcut_g = 1.7775013664098447;


#if USE_MPC_PRE_CALCULATION
int mpc_error[N_SELECTIVITY_LEVEL][HW2 + 1][HW2 - 3][HW2 - 3];
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
    return res;
}

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, const bool is_end_search, std::vector<bool*> &searchings);

/*
    @brief Multi-ProbCut for normal search

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                depth of deep search
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param v                    an integer to store result
    @param searching            flag for terminating this search
    @return cutoff occurred?
*/
inline bool mpc(Search* search, int alpha, int beta, int depth, uint64_t legal, const bool is_end_search, int* v, std::vector<bool*> &searchings) {
    int search_depth;
    if (is_end_search) {
        search_depth = ((depth / 3) & 0b11111110) + (depth & 1); // depth / 3 + parity
    } else {
        search_depth = ((depth / 2) & 0b11111110) + (depth & 1); // depth / 2 + parity
    }
    // int search_depth = ((depth / 2) & 0b11111110) + (depth & 1); // depth / 3 + parity
    int d0value = mid_evaluate_diff(search);
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

    if (is_end_search) {
        if ((alpha & 1) == 0) {
            alpha += 1;
        }
        if ((beta & 1) == 0) {
            beta -= 1;
        }
    }
    
    if (search_depth == 0) {
#if USE_MPC_PRE_CALCULATION
        int error = mpc_error[search->mpc_level][search->n_discs][0][depth];
#else
        double mpct = SELECTIVITY_MPCT[search->mpc_level];
        int error = ceil(mpct * probcut_sigma(search->n_discs, 0, depth));
#endif
        if (d0value >= beta + error) {
            *v = beta;
            if (is_end_search) {
                *v += beta & 1;
            }
            return true;
        }
        if (d0value <= alpha - error) {
            *v = alpha;
            if (is_end_search) {
                *v -= alpha & 1;
            }
            return true;
        }
    } else {
        uint_fast8_t mpc_level = search->mpc_level;
#if USE_MPC_PRE_CALCULATION
        int error_search = mpc_error[mpc_level][search->n_discs][search_depth][depth];
#else
        double mpct = SELECTIVITY_MPCT[mpc_level];
        int error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
#endif
        // if (is_end_search) {
        //     error_search += 1.5;
        // }
        int error_0 = std::max(1, error_search - MPC_ERROR0_OFFSET);
        search->mpc_level = MPC_100_LEVEL;
        if (d0value >= beta + error_0) {
            int pc_beta = beta + error_search;
            if (pc_beta <= SCORE_MAX) {
                if (nega_alpha_ordering_nws(search, pc_beta - 1, search_depth, false, legal, false, searchings) >= pc_beta) {
                    *v = beta;
                    if (is_end_search) {
                        *v += beta & 1;
                    }
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
                    if (is_end_search) {
                        *v -= alpha & 1;
                    }
                    search->mpc_level = mpc_level;
                    return true;
                }
            }
        }
        search->mpc_level = mpc_level;
    }
    return false;
}

inline bool mpc(Search* search, int alpha, int beta, int depth, uint64_t legal, const bool is_end_search, int* v, bool *searching) {
    std::vector<bool*> searchings = {searching};
    return mpc(search, alpha, beta, depth, legal, is_end_search, v, searchings);
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
    int d0value = mid_evaluate_diff(search);
    if (d0value <= alpha - (error_search + error_0) / 2) {
        int pc_alpha = alpha - error_search;
        if (pc_alpha > -SCORE_MAX) {
            if (nega_alpha_ordering_nws(search, pc_alpha, search_depth, false, legal, false, searching) <= pc_alpha) {
                return true;
            }
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
                for (depth2 = 0; depth2 < HW2 - 3; ++depth2) {
                    mpc_error[mpc_level][n_discs][depth1][depth2] = ceil(SELECTIVITY_MPCT[mpc_level] * probcut_sigma(n_discs, depth1, depth2));
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
        for (int depth = 4; depth <= 16; ++depth) {
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
        for (int depth = 2; depth <= 26; ++depth) {
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