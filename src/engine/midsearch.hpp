/*
    Egaroucid Project

    @file midsearch.hpp
        Search midgame
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
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
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "thread_pool.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "endsearch.hpp"
#include "midsearch_nws.hpp"
#include "etc.hpp"

inline int aspiration_search(Search *search, int alpha, int beta, int predicted_value, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

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
inline int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_eval1(search, -beta, -alpha, true);
        search->pass();
        return v;
    }
    int g;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->move(&flip);
            ++search->n_nodes;
            g = -mid_evaluate_diff(search);
        search->undo(&flip);
        ++search->n_nodes;
        if (v < g){
            if (alpha < g){
                if (beta <= g)
                    return g;
                alpha = g;
            }
            v = g;
        }
    }
    return v;
}

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
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (alpha + 1 == beta)
        return nega_alpha_ordering_nws(search, alpha, depth, skipped, legal, is_end_search, searching);
    if (is_end_search && search->n_discs == 60){
        return -last4(search, -beta, -alpha);
    }
    if (!is_end_search){
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, beta, skipped);
        if (depth == 0){
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
        if (is_end_search){
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED)
                return stab_res;
        }
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -SCORE_INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_scout(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    if (transposition_cutoff(search, hash_code, depth, &alpha, &beta, &v, moves)){
        return v;
    }
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    const int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent)
            return SCORE_MAX;
        ++idx;
    }
    int etc_done_idx = 0;
    #if USE_MID_ETC
        if (depth >= MID_ETC_DEPTH){
            if (etc(search, move_list, depth, &alpha, &beta, &v, &etc_done_idx)){
                return v;
            }
        }
    #endif
    #if USE_MID_MPC
        if (depth >= USE_MPC_DEPTH){
            if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching)){
                //transposition_table.reg(search, hash_code, depth, alpha, beta, v, TRANSPOSITION_TABLE_UNDEFINED);
                return v;
            }
        }
    #endif
    int g;
    #if USE_ASPIRATION_NEGASCOUT && false
        if (beta - alpha >= 4 && depth >= 5){
            int l = -HW2, u = HW2;
            transposition_table.get_value(search, depth - 5, hash_code, &l, &u);
            if (l == u && alpha < l && l < beta){
                return aspiration_search(search, alpha, beta, l, depth, skipped, legal, is_end_search, searching);
            }
        }
    #endif
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    #if USE_YBWC_NEGASCOUT
        if (search->use_multi_thread && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH){
            move_list_sort(move_list);
            if (move_list[0].flip.flip){
                search->move(&move_list[0].flip);
                    v = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                search->undo(&move_list[0].flip);
                move_list[0].flip.flip = 0;
                best_move = move_list[0].flip.pos;
                if (alpha < v){
                    alpha = v;
                }
                if (alpha < beta){
                    ybwc_search_young_brothers(search, &alpha, &beta, &v, &best_move, hash_code, depth, is_end_search, move_list, false, searching);
                }
            }
        } else{
    #endif
            for (int move_idx = 0; move_idx < canput - etc_done_idx && *searching; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                #if USE_MID_ETC
                    if (move_list[move_idx].flip.flip == 0ULL)
                        break;
                #endif
                if (search->need_to_see_tt_loop){
                    if (transposition_cutoff_bestmove(search, hash_code, depth, &alpha, &beta, &v, &best_move)){
                        break;
                    }
                }
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF){
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (alpha < g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    }
                search->undo(&move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v){
                        if (beta <= v)
                            break;
                        alpha = v;
                    }
                }
            }
    #if USE_YBWC_NEGASCOUT
        }
    #endif
    if (*searching && global_searching)
        transposition_table.reg(search, hash_code, depth, first_alpha, first_beta, v, best_move);
    return v;
}

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
inline int aspiration_search(Search *search, int alpha, int beta, int predicted_value, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (predicted_value < alpha || beta <= predicted_value)
        return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    int pred_alpha = predicted_value - 1;
    int pred_beta = predicted_value + 1;
    if ((predicted_value % 2) && is_end_search){
        --pred_alpha;
        ++pred_beta;
    }
    pred_alpha = std::max(pred_alpha, alpha);
    pred_beta = std::min(pred_beta, beta);
    if (pred_beta - pred_alpha > 0){
        int g = nega_scout(search, pred_alpha, pred_beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
        if (pred_alpha < g && g < pred_beta)
            return g;
    }
    return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
}

/*
    @brief Wrapper of nega_scout

    This function is used in root node

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
std::pair<int, int> first_nega_scout_legal(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const std::vector<Clog_result> clogs, uint64_t legal, uint64_t strt, bool *searching){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    //uint64_t legal = search->board.get_legal();
    int g, v = -SCORE_INF, first_alpha = alpha;
    if (legal == 0ULL)
        return std::make_pair(SCORE_UNDEFINED, -1);
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    const int canput_all = pop_count_ull(legal);
    for (const Clog_result clog: clogs){
        if (v < clog.val){
            v = clog.val;
            best_move = clog.pos;
            if (alpha < v){
                alpha = v;
            }
        }
        legal ^= 1ULL << clog.pos;
    }
    uint32_t hash_code = search->board.hash();
    if (alpha < beta && legal){
        int pv_idx = 1;
        const int canput = pop_count_ull(legal);
        std::vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&move_list[idx].flip, &search->board, cell);
            if (move_list[idx].flip.flip == search->board.opponent)
                return std::make_pair(SCORE_MAX, (int)cell);
            ++idx;
        }
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
        transposition_table.get_moves_any_level(&search->board, hash_code, moves);
        move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);

        #if USE_YBWC_NEGASCOUT
            if (search->use_multi_thread && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH){
                move_list_sort(move_list);
                search->move(&move_list[0].flip);
                    v = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                search->undo(&move_list[0].flip);
                move_list[0].flip.flip = 0;
                best_move = move_list[0].flip.pos;
                if (alpha < v){
                    alpha = v;
                }
                if (alpha < beta){
                    ybwc_search_young_brothers(search, &alpha, &beta, &v, &best_move, hash_code, depth, is_end_search, move_list, true, searching);
                }
            } else{
        #endif
                for (int move_idx = 0; move_idx < canput && *searching; ++move_idx){
                    swap_next_best_move(move_list, move_idx, canput);
                    search->move(&move_list[move_idx].flip);
                        if (v == -SCORE_INF){
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        } else{
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            if (alpha < g && g < beta){
                                g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            }
                        }
                    search->undo(&move_list[move_idx].flip);
                    if (v < g){
                        v = g;
                        best_move = move_list[move_idx].flip.pos;
                        if (alpha < v){
                            if (beta <= v)
                                break;
                            alpha = v;
                        }
                    }
                }
        #if USE_YBWC_NEGASCOUT
            }
        #endif
    }
    if (*searching && global_searching)
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
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
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const std::vector<Clog_result> clogs, uint64_t strt, bool *searching){
    return first_nega_scout_legal(search, alpha, beta, predicted_value, depth, is_end_search, clogs, search->board.get_legal(), strt, searching);
}

void first_nega_scout_hint_sub_thread(Search *search, int depth, bool is_end_search, uint64_t legal, bool *searching){
    ++search->n_nodes;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->move(&flip);
            nega_scout(search, -SCORE_MAX, SCORE_MAX, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
        search->undo(&flip);
    }
}

void first_nega_scout_hint(Search *search, int depth, int max_depth, bool is_end_search, uint64_t legal, bool *searching, double values[], int types[], int type, int n_display){
    ++search->n_nodes;
    std::vector<Value_policy> value_policies;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        Value_policy elem;
        elem.value = values[cell];
        elem.policy = cell;
        value_policies.emplace_back(elem);
    }
    std::sort(value_policies.begin(), value_policies.end());
    int n_threshold = n_display + 1 + ((int)value_policies.size() - n_display) * std::max(0, max_depth - depth) / max_depth;
    if (n_threshold > (int)value_policies.size()){
        n_threshold = (int)value_policies.size();
    }
    //std::cerr << depth << " " << max_depth << " " << n_threshold << " " << n_display << " " << value_policies.size() << std::endl;
    Flip flip;
    for (int i = 0; i < n_threshold && *searching && global_searching; ++i){
        calc_flip(&flip, &search->board, value_policies[i].policy);
        search->move(&flip);
            int g = -nega_scout(search, -SCORE_MAX, SCORE_MAX, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            if (-SCORE_MAX <= g && g <= SCORE_MAX && *searching && global_searching){
                if (values[value_policies[i].policy] == SCORE_UNDEFINED || is_end_search)
                    values[value_policies[i].policy] = g;
                else
                    values[value_policies[i].policy] = (0.9 * values[value_policies[i].policy] + 1.1 * g) / 2.0;
                types[value_policies[i].policy] = type;
            }
        search->undo(&flip);
    }
}
