/*
    Egaroucid Project

    @file midsearch.hpp
        Search midgame
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transposition_table.hpp"
#include "transposition_cutoff.hpp"
#include "move_ordering.hpp"
#include "multi_probcut.hpp"
#include "thread_pool.hpp"
#include "util.hpp"
#include "stability_cutoff.hpp"
#include "endsearch.hpp"
#include "midsearch_nws.hpp"
#include "book.hpp"

inline int aspiration_search(Search *search, int alpha, int beta, int predicted_value, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, bool *searching);

/*
    @brief Get a value with last move with Nega-Alpha algorithm

    No move ordering. Just search it.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param skipped              already passed?
    @param searching            flag for terminating this search
    @return the value
*/
inline int nega_alpha_eval1(Search *search, int alpha, int beta, const bool skipped) {
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass();
            v = -nega_alpha_eval1(search, -beta, -alpha, true);
        search->pass();
        return v;
    }
    int g;
    Flip flip;
    for (int i = 0; i < N_STATIC_CELL_PRIORITY; ++i) {
        uint64_t l = legal & static_cell_priority[i];
        for (uint_fast8_t cell = first_bit(&l); l; cell = next_bit(&l)) {
            calc_flip(&flip, &search->board, cell);
            search->move(&flip);
                ++search->n_nodes;
                g = -mid_evaluate_diff(search);
            search->undo(&flip);
            if (v < g) {
                if (alpha < g) {
                    if (beta <= g) {
                        return g;
                    }
                    alpha = g;
                }
                v = g;
            }
        }
    }
    // for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
    //     calc_flip(&flip, &search->board, cell);
    //     search->move(&flip);
    //         ++search->n_nodes;
    //         g = -mid_evaluate_diff(search);
    //     search->undo(&flip);
    //     ++search->n_nodes;
    //     if (v < g) {
    //         if (alpha < g) {
    //             if (beta <= g) {
    //                 return g;
    //             }
    //             alpha = g;
    //         }
    //         v = g;
    //     }
    // }
    return v;
}


// inline int nega_alpha_simple(Search *search, int alpha, int beta, const int depth, const bool skipped) {
//     ++search->n_nodes;
// #if USE_SEARCH_STATISTICS
//     ++search->n_nodes_discs[search->n_discs];
// #endif
//     int v = -SCORE_INF;
//     uint64_t legal = search->board.get_legal();
//     if (legal == 0ULL) {
//         if (skipped) {
//             return end_evaluate(&search->board);
//         }
//         search->pass();
//             v = -nega_alpha_simple(search, -beta, -alpha, depth, true);
//         search->pass();
//         return v;
//     }
//     int g;
//     Flip flip;
//     for (int i = 0; i < N_STATIC_CELL_PRIORITY; ++i) {
//         uint64_t l = legal & static_cell_priority[i];
//         for (uint_fast8_t cell = first_bit(&l); l; cell = next_bit(&l)) {
//             calc_flip(&flip, &search->board, cell);
//             search->move(&flip);
//                 g = -nega_alpha_end_simple(search, -beta, -alpha, depth - 1, false);
//             search->undo(&flip);
//             ++search->n_nodes;
//             if (v < g) {
//                 if (alpha < g) {
//                     if (beta <= g) {
//                         return g;
//                     }
//                     alpha = g;
//                 }
//                 v = g;
//             }
//         }
//     }
//     return v;
// }


/*
    @brief Get a value with given depth with Negascout algorithm

    Search with move ordering for midgame

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                remaining depth
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param searching            flag for terminating this search
    @return the value
*/
int nega_scout(Search *search, int alpha, int beta, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, bool *searching) {
    if (!global_searching || !(*searching)) {
        return SCORE_UNDEFINED;
    }
    if (alpha + 1 == beta) {
        return nega_alpha_ordering_nws(search, alpha, depth, skipped, legal, is_end_search, searching);
    }
    if (is_end_search && search->n_discs == HW2 - 4) {
        return -last4(search, -beta, -alpha);
    }
    if (search->n_discs == HW2) {
        return end_evaluate(&search->board);
    }
    if (!is_end_search) {
        if (depth == 1) {
            return nega_alpha_eval1(search, alpha, beta, skipped);
        }
        if (depth == 0) {
            ++search->n_nodes;
            return mid_evaluate_diff(search);
        }
    }
    ++search->n_nodes;
    int first_alpha = alpha;
    int first_beta = beta;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
#if USE_END_SC
    if (is_end_search) {
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    }
#endif
    if (legal == LEGAL_UNDEFINED) {
        legal = search->board.get_legal();
    }
    int v = -SCORE_INF;
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass();
            v = -nega_scout(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    transposition_table.prefetch(hash_code);
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    if (transposition_cutoff(search, hash_code, depth, &alpha, &beta, &v, moves)) {
        return v;
    }
    int best_move = MOVE_UNDEFINED;
    const int canput = pop_count_ull(legal);
    // std::vector<Flip_value> move_list(canput);
    Flip_value move_list[MAX_N_BRANCHES];
    int idx = 0;
    int tt_moves_idx0 = -1;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent) {
            return SCORE_MAX;
        }
        if (cell == moves[0]) {
            tt_moves_idx0 = idx;
        }
        ++idx;
    }
    int n_etc_done = 0;
#if USE_MID_ETC
    if (depth >= MID_ETC_DEPTH) {
        if (etc(search, move_list, canput, depth, &alpha, &beta, &v, &n_etc_done)) {
            return v;
        }
    }
#endif
#if USE_MID_MPC
    if (search->mpc_level < MPC_100_LEVEL && depth >= USE_MPC_MIN_DEPTH) {
        if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching)) {
            return v;
        }
    }
#endif
    int g;
#if USE_ASPIRATION_NEGASCOUT
    if (beta - alpha >= 4 && depth >= 5) {
        int l = -HW2, u = HW2;
        transposition_table.get_bounds(search, hash_code, depth - 5, &l, &u);
        if (l == u && alpha < l && l < beta && !((l % 2) && is_end_search)) {
            return aspiration_search(search, alpha, beta, l, depth, skipped, legal, is_end_search, searching);
        }
    }
#endif
    bool serial_searched = false;
    if (tt_moves_idx0 != -1 && move_list[tt_moves_idx0].flip.flip) {
        search->move(&move_list[tt_moves_idx0].flip);
            g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[tt_moves_idx0].n_legal, is_end_search, searching);
        search->undo(&move_list[tt_moves_idx0].flip);
        if (v < g) {
            v = g;
            best_move = move_list[tt_moves_idx0].flip.pos;
            if (alpha < v) {
                alpha = v;
            }
        }
        serial_searched = true;
        move_list[tt_moves_idx0].flip.flip = 0;
        move_list[tt_moves_idx0].value = -INF;
    }
    if (alpha < beta) {
        move_list_evaluate(search, move_list, canput, moves, depth, alpha, beta, searching);
#if USE_YBWC_NEGASCOUT
        if (search->use_multi_thread && ((!is_end_search && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth - 1 >= YBWC_END_SPLIT_MIN_DEPTH))) {
            move_list_sort(move_list, canput);
            if (move_list[0].flip.flip) {
                if (!serial_searched) {
                    search->move(&move_list[0].flip);
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                    search->undo(&move_list[0].flip);
                    move_list[0].flip.flip = 0;
                    if (v < g) {
                        v = g;
                        best_move = move_list[0].flip.pos;
                        if (alpha < v) {
                            alpha = v;
                        }
                    }
                }
                if (alpha < beta) {
                    ybwc_search_young_brothers(search, &alpha, &beta, &v, &best_move, canput - n_etc_done - 1, hash_code, depth, is_end_search, move_list, canput, false, searching);
                }
            }
        } else{
#endif
            for (int move_idx = 0; move_idx < canput - n_etc_done && *searching; ++move_idx) {
                swap_next_best_move(move_list, move_idx, canput);
#if USE_MID_ETC
                if (move_list[move_idx].flip.flip == 0) {
                    break;
                }
#endif
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF) {
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (alpha < g && g < beta) {
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (v < g) {
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v) {
                        if (beta <= v) {
#if USE_KILLER_MOVE_MO
                            // Update all heuristics on beta cutoff
                            search->update_heuristics_on_cutoff(move_list[move_idx].flip.pos, depth);
#endif
                            break;
                        }
                        alpha = v;
                    }
                }
            }
#if USE_YBWC_NEGASCOUT
        }
#endif
    }
    if (*searching && global_searching) {
        transposition_table.reg(search, hash_code, depth, first_alpha, first_beta, v, best_move);
    }
    return v;
}




// int nega_scout_policy(Search *search, int alpha, int beta, const int depth, bool skipped, uint64_t legal, const bool is_end_search, bool *searching) {
//     if (!global_searching || !(*searching)) {
//         return SCORE_UNDEFINED;
//     }
//     ++search->n_nodes;
//     int first_alpha = alpha;
//     int first_beta = beta;
// #if USE_SEARCH_STATISTICS
//     ++search->n_nodes_discs[search->n_discs];
// #endif
//     if (legal == LEGAL_UNDEFINED) {
//         legal = search->board.get_legal();
//     }
//     int v = -SCORE_INF;
//     if (legal == 0ULL) {
//         if (skipped) {
//             return MOVE_NOMOVE;
//         }
//         search->pass();
//             int policy = nega_scout_policy(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
//         search->pass();
//         return policy;
//     }
//     uint32_t hash_code = search->board.hash();
//     transposition_table.prefetch(hash_code);
//     uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
//     transposition_table.get_moves_any_level(&search->board, hash_code, moves);
//     int best_move = MOVE_UNDEFINED;
//     const int canput = pop_count_ull(legal);
//     // std::vector<Flip_value> move_list(canput);
//     Flip_value move_list[MAX_N_BRANCHES];
//     int idx = 0;
//     int tt_moves_idx0 = -1;
//     for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
//         calc_flip(&move_list[idx].flip, &search->board, cell);
//         if (move_list[idx].flip.flip == search->board.opponent) {
//             return cell;
//         }
//         if (cell == moves[0]) {
//             tt_moves_idx0 = idx;
//         }
//         ++idx;
//     }
//     int n_etc_done = 0;
//     int g;
//     bool serial_searched = false;
//     if (tt_moves_idx0 != -1 && move_list[tt_moves_idx0].flip.flip) {
//         search->move(&move_list[tt_moves_idx0].flip);
//             g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[tt_moves_idx0].n_legal, is_end_search, searching);
//         search->undo(&move_list[tt_moves_idx0].flip);
//         if (v < g) {
//             v = g;
//             best_move = move_list[tt_moves_idx0].flip.pos;
//             if (alpha < v) {
//                 alpha = v;
//             }
//         }
//         serial_searched = true;
//         move_list[tt_moves_idx0].flip.flip = 0;
//         move_list[tt_moves_idx0].value = -INF;
//     }
//     if (alpha < beta) {
//         move_list_evaluate(search, move_list, canput, moves, depth, alpha, beta, searching);
// #if USE_YBWC_NEGASCOUT
//         if (search->use_multi_thread && ((!is_end_search && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth - 1 >= YBWC_END_SPLIT_MIN_DEPTH))) {
//             move_list_sort(move_list, canput);
//             if (move_list[0].flip.flip) {
//                 if (!serial_searched) {
//                     search->move(&move_list[0].flip);
//                         g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
//                     search->undo(&move_list[0].flip);
//                     move_list[0].flip.flip = 0;
//                     if (v < g) {
//                         v = g;
//                         best_move = move_list[0].flip.pos;
//                         if (alpha < v) {
//                             alpha = v;
//                         }
//                     }
//                 }
//                 if (alpha < beta) {
//                     ybwc_search_young_brothers(search, &alpha, &beta, &v, &best_move, canput - n_etc_done - 1, hash_code, depth, is_end_search, move_list, canput, false, searching);
//                 }
//             }
//         } else{
// #endif
//             for (int move_idx = 0; move_idx < canput - n_etc_done && *searching; ++move_idx) {
//                 swap_next_best_move(move_list, move_idx, canput);
// #if USE_MID_ETC
//                 if (move_list[move_idx].flip.flip == 0) {
//                     break;
//                 }
// #endif
//                 search->move(&move_list[move_idx].flip);
//                     if (v == -SCORE_INF) {
//                         g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
//                     } else{
//                         g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
//                         if (alpha < g && g < beta) {
//                             g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
//                         }
//                     }
//                 search->undo(&move_list[move_idx].flip);
//                 if (v < g) {
//                     v = g;
//                     best_move = move_list[move_idx].flip.pos;
//                     if (alpha < v) {
//                         if (beta <= v) {
// #if USE_KILLER_MOVE_MO
//                             search->update_heuristics_on_cutoff(move_list[move_idx].flip.pos, depth);
// #endif
//                             break;
//                         }
//                         alpha = v;
//                     }
//                 }
//             }
// #if USE_YBWC_NEGASCOUT
//         }
// #endif
//     }
//     if (*searching && global_searching) {
//         transposition_table.reg(search, hash_code, depth, first_alpha, first_beta, v, best_move);
//     }
//     return best_move;
// }





#if USE_ASPIRATION_NEGASCOUT
/*
    @brief aspiration search used in endgame search

    Used in PV node, if predicted value is available
    Based on MTD-f algorithm

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param predicted_value      value prediction
    @param depth                remaining depth
    @param is_end_search        search till the end?
    @param is_main_search       is this main search? (used for logging)
    @param best_move            previously calculated best move
    @param clogs                previously found clog moves
    @param legal                legal moves in bitboard
    @return pair of value and best move
*/
inline int aspiration_search(Search *search, int alpha, int beta, int predicted_value, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, bool *searching) {
    int pred_alpha = predicted_value - 1;
    int pred_beta = predicted_value + 1;
    int g = nega_scout(search, pred_alpha, pred_beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    if (pred_alpha < g && g < pred_beta) {
        return g;
    } else if (g <= pred_alpha) {
        if (g <= alpha) {
            return g;
        }
        beta = g;
    } else if (pred_beta <= g) {
        if (beta <= g) {
            return g;
        }
        alpha = g;
    }
    return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
}
#endif

/*
    @brief Wrapper of nega_scout

    This function is used in root node

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                remaining depth
    @param is_end_search        search till the end?
    @param is_main_search       is this main search? (used for logging)
    @param best_move            previously calculated best move
    @param clogs                previously found clog moves
    @param legal                legal moves in bitboard
    @return pair of value and best move
*/
std::pair<int, int> first_nega_scout_legal(Search *search, int alpha, int beta, const int depth, const bool is_end_search, const std::vector<Clog_result> clogs, uint64_t legal, uint64_t strt, bool *searching) {
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
    int g, v = -SCORE_INF, first_alpha = alpha;
    if (legal == 0ULL) {
        return std::make_pair(SCORE_UNDEFINED, -1);
    }
    bool is_all_legal = legal == search->board.get_legal();
    int best_move = MOVE_UNDEFINED;
    const int canput_all = pop_count_ull(legal);
    for (const Clog_result clog: clogs) {
        if (legal & (1ULL << clog.pos)) {
            if (v < clog.val) {
                v = clog.val;
                best_move = clog.pos;
                if (alpha < v) {
                    alpha = v;
                }
            }
            legal &= ~(1ULL << clog.pos);
        }
    }
    uint32_t hash_code = search->board.hash();
    if (alpha < beta && legal) {
        int pv_idx = 1;
        const int canput = pop_count_ull(legal);
        // std::vector<Flip_value> move_list(canput);
        Flip_value move_list[MAX_N_BRANCHES];
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            calc_flip(&move_list[idx].flip, &search->board, cell);
            if (move_list[idx].flip.flip == search->board.opponent) {
                return std::make_pair(SCORE_MAX, (int)cell);
            }
            ++idx;
        }
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
        transposition_table.get_moves_any_level(&search->board, hash_code, moves);
        move_list_evaluate(search, move_list, canput, moves, depth, alpha, beta, searching);
#if USE_YBWC_NEGASCOUT
        if (
            search->use_multi_thread && 
            ((!is_end_search && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth - 1 >= YBWC_END_SPLIT_MIN_DEPTH)) //&& 
            //((!is_end_search && depth - 1 <= YBWC_MID_SPLIT_MAX_DEPTH) || (is_end_search && depth - 1 <= YBWC_END_SPLIT_MAX_DEPTH))
        ) {
            move_list_sort(move_list, canput);
            if (move_list[0].flip.flip) {
                search->move(&move_list[0].flip);
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                search->undo(&move_list[0].flip);
                move_list[0].flip.flip = 0;
                if (v < g) {
                    v = g;
                    best_move = move_list[0].flip.pos;
                    if (alpha < v) {
                        alpha = v;
                    }
                }
                if (alpha < beta && *searching) {
                    ybwc_search_young_brothers(search, &alpha, &beta, &v, &best_move, canput - 1, hash_code, depth, is_end_search, move_list, canput, true, searching);
                }
            }
        } else{
#endif
            for (int move_idx = 0; move_idx < canput && *searching; ++move_idx) {
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF) {
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (alpha < g && g < beta) {
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                //std::cerr << move_idx << " " << idx_to_coord(move_list[move_idx].flip.pos) << " " << g << " " << alpha << "," << beta << std::endl;
                if (v < g) {
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v) {
                        if (beta <= v) {
#if USE_KILLER_MOVE_MO
                            search->update_heuristics_on_cutoff(move_list[move_idx].flip.pos, depth);
#endif
                            break;
                        }
                        alpha = v;
                    }
                }
            }
#if USE_YBWC_NEGASCOUT
        }
#endif
    }
    if (*searching && global_searching && is_all_legal) {
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    }
    return std::make_pair(v, best_move);
}

/*
    @brief Wrapper of nega_scout

    This function is used in root node

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                remaining depth
    @param is_end_search        search till the end?
    @param is_main_search       is this main search? (used for logging)
    @param best_move            previously calculated best move
    @param clogs                previously found clog moves
    @return pair of value and best move
*/
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, const int depth, const bool is_end_search, const std::vector<Clog_result> clogs, uint64_t strt, bool *searching) {
    return first_nega_scout_legal(search, alpha, beta, depth, is_end_search, clogs, search->board.get_legal(), strt, searching);
}

Analyze_result first_nega_scout_analyze(Search *search, int alpha, int beta, const int depth, const bool is_end_search, const std::vector<Clog_result> clogs, int clog_depth, uint_fast8_t played_move, uint64_t strt, bool *searching) {
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
    Analyze_result res;
    res.played_move = played_move;
    res.alt_score = -SCORE_INF;
    res.alt_move = MOVE_UNDEFINED;
    int g, first_alpha = alpha;
    uint64_t legal = search->board.get_legal();
    const int canput_all = pop_count_ull(legal);
    for (const Clog_result clog: clogs) {
        if (clog.pos == played_move) {
            res.played_score = clog.val;
            res.played_depth = clog_depth;
            res.played_probability = 100;
        } else if (res.alt_score < clog.val) {
            res.alt_score = clog.val;
            res.alt_move = clog.pos;
            res.alt_depth = clog_depth;
            res.alt_probability = 100;
            if (alpha < res.alt_score) {
                alpha = res.alt_score;
            }
        }
        legal &= ~(1ULL << clog.pos);
    }
    uint32_t hash_code = search->board.hash();
    if (alpha < beta && (legal & (1ULL << played_move))) {
        Flip flip;
        calc_flip(&flip, &search->board, played_move);
        search->move(&flip);
            if (book.contain(&search->board)) {
                res.played_depth = SEARCH_BOOK;
                res.played_score = -book.get(search->board).value;
            } else{
                res.played_depth = depth;
                res.played_probability = SELECTIVITY_PERCENTAGE[search->mpc_level];
                res.played_score = -nega_scout(search, -SCORE_MAX, SCORE_MAX, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            }
        search->undo(&flip);
        legal ^= 1ULL << played_move;
    }
    if (alpha < beta && legal) {
        int pv_idx = 1;
        const int canput = pop_count_ull(legal);
        // std::vector<Flip_value> move_list(canput);
        Flip_value move_list[MAX_N_BRANCHES];
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            calc_flip(&move_list[idx].flip, &search->board, cell);
            ++idx;
        }
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
        transposition_table.get_moves_any_level(&search->board, hash_code, moves);
        move_list_evaluate(search, move_list, canput, moves, depth, alpha, beta, searching);
#if USE_YBWC_NEGASCOUT_ANALYZE
        if (search->use_multi_thread && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH) {
            move_list_sort(move_list);
            bool book_used = false;
            search->move(&move_list[0].flip);
                if (book.contain(&search->board)) {
                    book_used = true;
                    g = -book.get(search->board).value;
                } else {
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                }
            search->undo(&move_list[0].flip);
            move_list[0].flip.flip = 0;
            if (res.alt_score < g) {
                res.alt_score = g;
                res.alt_move = move_list[0].flip.pos;
                if (book_used) {
                    res.alt_depth = SEARCH_BOOK;
                } else {
                    res.alt_depth = depth;
                    res.alt_probability = SELECTIVITY_PERCENTAGE[search->mpc_level];
                }
                if (alpha < res.alt_score) {
                    alpha = res.alt_score;
                }
            }
            if (alpha < beta) {
                ybwc_search_young_brothers(search, &alpha, &beta, &res.alt_score, &res.alt_move, hash_code, depth, is_end_search, move_list, true, searching);
            }
        } else{
#endif
            for (int move_idx = 0; move_idx < canput && *searching; ++move_idx) {
                swap_next_best_move(move_list, move_idx, canput);
                bool book_used = false;
                search->move(&move_list[move_idx].flip);
                    if (book.contain(&search->board)) {
                        book_used = true;
                        g = -book.get(search->board).value;
                    } else{
                        if (res.alt_score == -SCORE_INF) {
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        } else{
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            if (alpha < g && g < beta) {
                                g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (res.alt_score < g) {
                    res.alt_score = g;
                    res.alt_move = move_list[move_idx].flip.pos;
                    if (book_used) {
                        res.alt_depth = SEARCH_BOOK;
                    } else{
                        res.alt_depth = depth;
                        res.alt_probability = SELECTIVITY_PERCENTAGE[search->mpc_level];
                    }
                    if (alpha < res.alt_score) {
                        if (beta <= res.alt_score) {
                            break;
                        }
                        alpha = res.alt_score;
                    }
                }
            }
#if USE_YBWC_NEGASCOUT_ANALYZE
        }
#endif
    }
    if (res.alt_score == -SCORE_INF) {
        res.alt_move = -1;
        res.alt_score = -SCORE_INF;
        res.alt_depth = -1;
        res.alt_probability = 0;
    }
    if (*searching && global_searching) {
        int v, best_move;
        if (res.played_score >= res.alt_score) {
            v = res.played_score;
            best_move = res.played_move;
        } else{
            v = res.alt_score;
            best_move = res.alt_move;
        }
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    }
    return res;
}