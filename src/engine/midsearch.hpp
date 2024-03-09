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
            if (etc(search, move_list, depth, &alpha, &beta, &v, &etc_done_idx))
                return v;
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
    #if USE_ASPIRATION_NEGASCOUT
        if (beta - alpha > 2 && depth >= 5){
            int l = -HW2, u = HW2;
            transposition_table.get_value(search, depth - 5, hash_code, &l, &u);
            if (l == u && alpha < l && l < beta){
                return aspiration_search(search, alpha, beta, l, depth, skipped, legal, is_end_search, searching);
            }
        }
    #endif
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    #if USE_YBWC_NEGASCOUT
        if (
            search->use_multi_thread && 
            depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH
        ){
            int running_count = 0;
            std::vector<std::future<Parallel_task>> parallel_tasks;
            std::atomic<int> atomic_running_count = 0;
            std::vector<int> parallel_alphas;
            std::vector<int> parallel_idxes;
            std::vector<int> additional_search_windows;
            bool n_searching = true;
            int ybwc_idx;
            for (int move_idx = 0; move_idx < canput - etc_done_idx && *searching; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                #if USE_MID_ETC
                    if (move_list[move_idx].flip.flip == 0ULL)
                        break;
                #endif
                if (search->need_to_see_tt_loop){
                    if (transposition_cutoff(search, hash_code, depth, &alpha, &beta, &v, moves)){
                        return v;
                    }
                }
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF){
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        if (ybwc_split_nws(search, -alpha - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, move_idx, canput - etc_done_idx, running_count, parallel_tasks)){
                            ++running_count;
                            parallel_alphas.emplace_back(alpha);
                            parallel_idxes.emplace_back(move_idx);
                            additional_search_windows.emplace_back(SCORE_UNDEFINED);
                        } else{
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            if (alpha < g && g < beta){
                                g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v){
                        if (beta <= v){
                            n_searching = false;
                            break;
                        }
                        alpha = v;
                    }
                }
                if (running_count){
                    ybwc_get_end_tasks_negascout(search, parallel_tasks, parallel_alphas, additional_search_windows, &running_count, &g, &ybwc_idx);
                    if (g != SCORE_UNDEFINED && v < g){
                        v = g;
                        best_move = move_list[parallel_idxes[ybwc_idx]].flip.pos;
                        if (alpha < v){
                            if (beta <= v){
                                n_searching = false;
                                break;
                            }
                            alpha = v;
                        }
                    }
                    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
                        if (additional_search_windows[i] != SCORE_UNDEFINED){
                            additional_search_windows[i] = std::max(additional_search_windows[i], alpha);
                            search->move(&move_list[parallel_idxes[i]].flip);
                                g = -nega_scout(search, -beta, -additional_search_windows[i], depth - 1, false, move_list[parallel_idxes[i]].n_legal, is_end_search, searching);
                            search->undo(&move_list[parallel_idxes[i]].flip);
                            additional_search_windows[i] = SCORE_UNDEFINED;
                            if (v < g){
                                v = g;
                                best_move = move_list[parallel_idxes[i]].flip.pos;
                                if (alpha < v){
                                    if (beta <= v){
                                        n_searching = false;
                                        break;
                                    }
                                    alpha = v;
                                }
                            }
                        }
                    }
                }
            }
            if (running_count){
                ybwc_wait_all_negascout(search, parallel_tasks, parallel_alphas, additional_search_windows, &running_count, &g, &ybwc_idx, beta, searching, &n_searching);
                if (g != SCORE_UNDEFINED && v < g){
                    v = g;
                    best_move = move_list[parallel_idxes[ybwc_idx]].flip.pos;
                    if (alpha < v){
                        if (beta <= v)
                            n_searching = false;
                        alpha = v;
                    }
                }
                if (n_searching){
                    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
                        if (additional_search_windows[i] != SCORE_UNDEFINED){
                            additional_search_windows[i] = std::max(additional_search_windows[i], alpha);
                            search->move(&move_list[parallel_idxes[i]].flip);
                                g = -nega_scout(search, -beta, -additional_search_windows[i], depth - 1, false, move_list[parallel_idxes[i]].n_legal, is_end_search, searching);
                            search->undo(&move_list[parallel_idxes[i]].flip);
                            additional_search_windows[i] = SCORE_UNDEFINED;
                            if (v < g){
                                v = g;
                                best_move = move_list[parallel_idxes[i]].flip.pos;
                                if (alpha < v){
                                    if (beta <= v)
                                        break;
                                    alpha = v;
                                }
                            }
                        }
                    }
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
                    if (transposition_cutoff(search, hash_code, depth, &alpha, &beta, &v, moves)){
                        return v;
                    }
                }
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF)
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    else{
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
    // if (predicted_value < alpha || beta <= predicted_value)
    //     return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    // int g1 = nega_alpha_ordering_nws(search, predicted_value - 1, depth, false, LEGAL_UNDEFINED, is_end_search, searching); // search in [pred - 1, pred]
    // if (g1 < predicted_value) // when exact value < predicted value, exact value <= g1
    //     return nega_scout(search, alpha, g1, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    // int g2 = nega_alpha_ordering_nws(search, predicted_value, depth, false, LEGAL_UNDEFINED, is_end_search, searching); // search in [pred, pred + 1]
    // if (predicted_value < g2) // when predicted value < exact value, g2 <= exact value
    //     return nega_scout(search, g2, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    // return predicted_value;
    if (predicted_value < alpha || beta <= predicted_value)
        return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    int pred_alpha = predicted_value - 1;
    int pred_beta = predicted_value + 1;
    if (predicted_value % 2){
        --pred_alpha;
        ++pred_beta;
    }
    if (alpha <= pred_alpha && pred_beta <= beta){
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
std::pair<int, int> first_nega_scout_legal(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t legal, uint64_t strt, bool *searching){
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
        }
        legal ^= 1ULL << clog.pos;
    }
    alpha = std::max(alpha, v);
    uint32_t hash_code = search->board.hash();
    if (alpha < beta && legal){
        int lower = -SCORE_MAX, upper = SCORE_MAX;
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
        #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
            if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
                transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
        #else
            transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
        #endif
        int pv_idx = 1;
        Flip flip_best;
        //if (is_main_search){
        //    std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << 0 << "/" << canput_all << " best " << "??" << " [" << alpha << "," << beta << "] " << std::endl;
        //}
        for (uint_fast8_t i = 0; i < N_TRANSPOSITION_MOVES; ++i){
            if (moves[i] == TRANSPOSITION_TABLE_UNDEFINED)
                break;
            if (1 & (legal >> moves[i])){
                calc_flip(&flip_best, &search->board, moves[i]);
                search->move(&flip_best);
                    if (v == -SCORE_INF){
                        if (predicted_value == SCORE_UNDEFINED || !is_end_search)
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
                        else
                            g = -aspiration_search(search, -beta, -alpha, -predicted_value, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
                        if (alpha <= g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
                    }
                search->undo(&flip_best);
                if (v < g){
                    v = g;
                    best_move = moves[i];
                    if (alpha < v){
                        alpha = v;
                        if (beta <= v)
                            break;
                    }
                }
                if (is_main_search){
                    uint64_t elapsed = tim() - strt;
                    uint64_t nps = calc_nps(search->n_nodes, elapsed);
                    if (best_move != moves[i])
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value <= " << g << " time " << elapsed << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value = " << g << " time " << elapsed << std::endl;
                }
                legal ^= 1ULL << moves[i];
                ++pv_idx;
            }
        }
        if (alpha < beta && legal){
            const int canput = pop_count_ull(legal);
            std::vector<Flip_value> move_list(canput);
            int idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&move_list[idx].flip, &search->board, cell);
                if (move_list[idx].flip.flip == search->board.opponent)
                    return std::make_pair(SCORE_MAX, (int)cell);
                ++idx;
            }
            move_list_evaluate(search, move_list, depth, alpha, beta, searching);
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF){
                        if (predicted_value == SCORE_UNDEFINED || !is_end_search)
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        else
                            g = -aspiration_search(search, -beta, -alpha, -predicted_value, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (alpha <= g && g < beta)
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
                if (is_main_search){
                    uint64_t elapsed = tim() - strt;
                    uint64_t nps = calc_nps(search->n_nodes, elapsed);
                    if (best_move != move_list[move_idx].flip.pos)
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value <= " << g << " time " << elapsed << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value = " << g << " time " << elapsed << std::endl;
                }
                ++pv_idx;
            }
        }
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
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t strt, bool *searching){
    return first_nega_scout_legal(search, alpha, beta, predicted_value, depth, is_end_search, is_main_search, clogs, search->board.get_legal(), strt, searching);
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
    for (int i = 0; i < n_threshold; ++i){
        calc_flip(&flip, &search->board, value_policies[i].policy);
        search->move(&flip);
            int g = -nega_scout(search, -SCORE_MAX, SCORE_MAX, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            if (values[value_policies[i].policy] == SCORE_UNDEFINED || is_end_search)
                values[value_policies[i].policy] = g;
            else
                values[value_policies[i].policy] = (0.9 * values[value_policies[i].policy] + 1.1 * g) / 2.0;
            types[value_policies[i].policy] = type;
        search->undo(&flip);
    }
}
