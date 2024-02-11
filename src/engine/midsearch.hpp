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
inline int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_alpha_eval1(search, -beta, -alpha, true, searching);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    int g;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        eval_move(search, &flip);
        search->move(&flip);
            g = -mid_evaluate_diff(search);
        search->undo(&flip);
        eval_undo(search, &flip);
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

#if MID_FAST_DEPTH > 1
    /*
        @brief Get a value with last few moves with Nega-Alpha algorithm

        No move ordering. Just search it.

        @param search               search information
        @param alpha                alpha value
        @param beta                 beta value
        @param depth                remaining depth
        @param skipped              already passed?
        @param searching            flag for terminating this search
        @return the value
    */
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        if (alpha + 1 == beta)
            return nega_alpha_nws(search, alpha, depth, skipped, searching);
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, beta, skipped, searching);
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[search->n_discs];
        #endif
        if (depth == 0)
            return mid_evaluate_diff(search);
        int g, v = -SCORE_INF;
        uint64_t legal = search->board.get_legal();
        if (legal == 0ULL){
            if (skipped)
                return end_evaluate(&search->board);
            search->eval_feature_reversed ^= 1;
            search->board.pass();
                v = -nega_alpha(search, -beta, -alpha, depth, true, searching);
            search->board.pass();
            search->eval_feature_reversed ^= 1;
            return v;
        }
        Flip flip;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            eval_move(search, &flip);
            search->move(&flip);
                g = -nega_alpha(search, -beta, -alpha, depth - 1, false, searching);
            search->undo(&flip);
            eval_undo(search, &flip);
            if (v < g){
                v = g;
                if (alpha < v){
                    if (beta <= v)
                        break;
                    alpha = v;
                }
            }
        }
        return v;
    }
#endif

#if USE_NEGA_ALPHA_ORDERING
    /*
        @brief Get a value with given depth with Nega-Alpha algorithm

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
    int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        if (alpha + 1 == beta)
            return nega_alpha_ordering_nws(search, alpha, depth, skipped, legal, is_end_search, searching);
        if (is_end_search && depth <= MID_TO_END_DEPTH)
            return nega_alpha_end(search, alpha, beta, skipped, legal, searching);
        if (!is_end_search){
            #if MID_FAST_DEPTH > 1
                if (depth <= MID_FAST_DEPTH)
                    return nega_alpha(search, alpha, beta, depth, skipped, searching);
            #else
                if (depth == 1)
                    return nega_alpha_eval1(search, alpha, beta, skipped, searching);
                if (depth == 0)
                    return mid_evaluate_diff(search);
            #endif
        }
        ++search->n_nodes;
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
            search->eval_feature_reversed ^= 1;
            search->board.pass();
                v = -nega_alpha_ordering(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
            search->board.pass();
            search->eval_feature_reversed ^= 1;
            return v;
        }
        uint32_t hash_code = search->board.hash();
        int lower = -SCORE_MAX, upper = SCORE_MAX;
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
        #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
            if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
                transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
        #else
            transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
        #endif
        if (upper == lower)
            return upper;
        if (beta <= lower)
            return lower;
        if (upper <= alpha)
            return upper;
        if (alpha < lower)
            alpha = lower;
        if (upper < beta)
            beta = upper;
        #if USE_MID_MPC
            if (depth >= USE_MPC_DEPTH){
                if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                    return v;
            }
        #endif
        Flip flip_best;
        int best_move = TRANSPOSITION_TABLE_UNDEFINED;
        int first_alpha = alpha;
        int g;
        for (uint_fast8_t i = 0; i < N_TRANSPOSITION_MOVES; ++i){
            if (moves[i] == TRANSPOSITION_TABLE_UNDEFINED)
                break;
            if (1 & (legal >> moves[i])){
                calc_flip(&flip_best, &search->board, moves[i]);
                eval_move(search, &flip_best);
                search->move(&flip_best);
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                search->undo(&flip_best);
                eval_undo(search, &flip_best);
                if (v < g){
                    v = g;
                    best_move = moves[i];
                    if (alpha < v){
                        alpha = v;
                        if (beta <= v)
                            break;
                    }
                }
                legal ^= 1ULL << moves[i];
            }
        }
        if (alpha < beta && legal){
            const int canput = pop_count_ull(legal);
            std::vector<Flip_value> move_list(canput);
            int idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&move_list[idx].flip, &search->board, cell);
                if (move_list[idx].flip.flip == search->board.opponent)
                    return SCORE_MAX;
                ++idx;
            }
            move_list_evaluate(search, move_list, depth, alpha, beta, searching);
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
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
        }
        if (*searching && global_searching)
            transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
        return v;
    }
#endif



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
    #if USE_NEGA_ALPHA_END
        if (is_end_search && depth <= MID_TO_END_DEPTH)
            return nega_alpha_end(search, alpha, beta, skipped, legal, searching);
    #elif USE_NEGA_ALPHA_END_FAST
        if (is_end_search && search->n_discs >= HW2 - END_FAST_DEPTH)
            return nega_alpha_end_fast(search, alpha, beta, skipped, false, searching);
    #else
        if (is_end_search && search->n_discs == 60){
            return -last4(search, -beta, -alpha);
        }
    #endif
    if (!is_end_search){
        #if MID_FAST_DEPTH > 1
            if (depth <= MID_FAST_DEPTH)
                return nega_alpha(search, alpha, beta, depth, skipped, searching);
        #else
            if (depth == 1)
                return nega_alpha_eval1(search, alpha, beta, skipped, searching);
            if (depth == 0)
                return mid_evaluate_diff(search);
        #endif
    }
    ++search->n_nodes;
    int first_alpha = alpha;
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
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_scout(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    uint32_t hash_code = search->board.hash();
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
        if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
            transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    #else
        transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    #endif
    if (upper == lower)
        return upper;
    if (beta <= lower)
        return lower;
    if (upper <= alpha)
        return upper;
    if (alpha < lower)
        alpha = lower;
    if (upper < beta)
        beta = upper;
    #if USE_MID_MPC
        if (depth >= USE_MPC_DEPTH){
            if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                return v;
        }
    #endif
    int g;
    #if USE_ASPIRATION_NEGASCOUT
        if (search->mpc_level){
            int l = -HW2, u = HW2;
            --search->mpc_level;
                transposition_table.get(search, hash_code, std::max(0, depth - 2), &l, &u);
            ++search->mpc_level;
            if (l == u){
                g = nega_alpha_ordering_nws(search, l - 1, depth, false, legal, is_end_search, searching);
                if (g == l){
                    g = nega_alpha_ordering_nws(search, l, depth, false, legal, is_end_search, searching);
                    if (l == g)
                        return l;
                }
            }
        }
    #endif
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
    #if USE_MID_MPC && false
        if (depth >= USE_MPC_DEPTH + 1){
            if (enhanced_mpc(search, move_list, depth, alpha, beta, is_end_search, searching, &v))
                return v;
        }
    #endif
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    #if USE_YBWC_NEGASCOUT
        if (
            search->use_multi_thread && 
            #if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
                ((depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH && !is_end_search) || (depth - 1 >= YBWC_END_SPLIT_MIN_DEPTH && is_end_search))
            #else
                depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH
            #endif
        ){
            int running_count = 0;
            std::vector<std::future<Parallel_task>> parallel_tasks;
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
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF){
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    } else{
                        if (ybwc_split_nws(search, -alpha - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, move_idx, canput - etc_done_idx, running_count, false, parallel_tasks)){
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
                eval_undo(search, &move_list[move_idx].flip);
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
                            eval_move(search, &move_list[parallel_idxes[i]].flip);
                            search->move(&move_list[parallel_idxes[i]].flip);
                                g = -nega_scout(search, -beta, -additional_search_windows[i], depth - 1, false, move_list[parallel_idxes[i]].n_legal, is_end_search, searching);
                            search->undo(&move_list[parallel_idxes[i]].flip);
                            eval_undo(search, &move_list[parallel_idxes[i]].flip);
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
                            eval_move(search, &move_list[parallel_idxes[i]].flip);
                            search->move(&move_list[parallel_idxes[i]].flip);
                                g = -nega_scout(search, -beta, -additional_search_windows[i], depth - 1, false, move_list[parallel_idxes[i]].n_legal, is_end_search, searching);
                            search->undo(&move_list[parallel_idxes[i]].flip);
                            eval_undo(search, &move_list[parallel_idxes[i]].flip);
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
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF)
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (alpha < g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
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
    if (*searching && global_searching){
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    }
    return v;
}

/*
int nega_scout_lazy_smp(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    std::vector<std::future<int>> parallel_tasks;
    std::vector<Search> searches;
    int n_idle = thread_pool.get_n_idle();
    for (int i = 0; i < n_idle; ++i){
        Search n_search;
        n_search.init_board(&search->board);
        n_search.mpc_level = search->mpc_level;
        n_search.n_nodes = 0ULL;
        n_search.use_multi_thread = false;
        calc_features(&n_search);
        searches.emplace_back(n_search);
    }
    for (int i = 0; i < n_idle; ++i){
        bool pushed;
        parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[i], alpha, beta, depth, skipped, legal, is_end_search, searching)));
        if (!pushed){
            parallel_tasks.pop_back();
            break;
        }
        if (n_idle <= thread_pool.get_n_idle())
            break;
    }
    std::cerr << parallel_tasks.size() + 1 << " parallel" << std::endl;
    int res = nega_scout(search, alpha, beta, depth, skipped, legal, is_end_search, searching);
    uint64_t s = tim();
    for (std::future<int> &task: parallel_tasks)
        task.get();
    for (int i = 0; i < n_idle; ++i)
        search->n_nodes += searches[i].n_nodes;
    return res;
}
*/

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
int pv_aspiration_search(Search *search, int alpha, int beta, int predicted_value, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    /*
    if (predicted_value < alpha || beta <= predicted_value)
        return nega_scout(search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    int g1 = nega_alpha_ordering_nws(search, predicted_value - 1, depth, false, LEGAL_UNDEFINED, is_end_search, searching); // search in [pred - 1, pred]
    if (g1 < predicted_value) // when exact value < predicted value, exact value <= g1
        return nega_scout(search, alpha, g1, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    int g2 = nega_alpha_ordering_nws(search, predicted_value, depth, false, LEGAL_UNDEFINED, is_end_search, searching); // search in [pred, pred + 1]
    if (predicted_value < g2) // when predicted value < exact value, g2 <= exact value
        return nega_scout(search, g2, beta, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
    return predicted_value;
    */
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
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t legal, uint64_t strt){
    bool searching = true;
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
        for (uint_fast8_t i = 0; i < N_TRANSPOSITION_MOVES; ++i){
            if (moves[i] == TRANSPOSITION_TABLE_UNDEFINED)
                break;
            if (1 & (legal >> moves[i])){
                calc_flip(&flip_best, &search->board, moves[i]);
                eval_move(search, &flip_best);
                search->move(&flip_best);
                    if (v == -SCORE_INF){
                        if (predicted_value == SCORE_UNDEFINED || !is_end_search)
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                        else
                            g = -pv_aspiration_search(search, -beta, -alpha, -predicted_value, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                        if (alpha <= g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                    }
                search->undo(&flip_best);
                eval_undo(search, &flip_best);
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
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value <= " << g << " time " << elapsed << " nps " << nps << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value = " << g << " time " << elapsed << " nps " << nps << std::endl;
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
            move_list_evaluate(search, move_list, depth, alpha, beta, &searching);
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF){
                        if (predicted_value == SCORE_UNDEFINED || !is_end_search)
                            g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                        else
                            g = -pv_aspiration_search(search, -beta, -alpha, -predicted_value, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                        if (alpha <= g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
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
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value <= " << g << " time " << elapsed << " nps " << nps << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value = " << g << " time " << elapsed << " nps " << nps << std::endl;
                }
                ++pv_idx;
            }
        }
    }
    if (global_searching)
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
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int predicted_value, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t strt){
    return first_nega_scout(search, alpha, beta, predicted_value, depth, is_end_search, is_main_search, clogs, search->board.get_legal(), strt);
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
    @param legal                legal moves in bitboard
    @return value
*/
int first_nega_scout_value(Search *search, int alpha, int beta, int depth, bool is_end_search, const bool is_main_search, bool passed, uint64_t legal){
    bool searching = true;
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    //uint64_t legal = search->board.get_legal();
    int g, v = -SCORE_INF, first_alpha = alpha;
    if (legal == 0ULL){
        if (passed)
            return search->board.score_player();
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -first_nega_scout_value(search, -beta, -alpha, depth, is_end_search, is_main_search, true, search->board.get_legal());
        search->board.pass();
        search->eval_feature_reversed ^= 1;
    }
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    const int canput_all = pop_count_ull(legal);
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
        for (uint_fast8_t i = 0; i < N_TRANSPOSITION_MOVES; ++i){
            if (moves[i] == TRANSPOSITION_TABLE_UNDEFINED)
                break;
            if (1 & (legal >> moves[i])){
                calc_flip(&flip_best, &search->board, moves[i]);
                eval_move(search, &flip_best);
                search->move(&flip_best);
                    if (v == -SCORE_INF)
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                    else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                        if (alpha <= g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                    }
                search->undo(&flip_best);
                eval_undo(search, &flip_best);
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
                    if (best_move != moves[i])
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value <= " << g << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(moves[i]) << " value = " << g << std::endl;
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
                    return SCORE_MAX;
                ++idx;
            }
            move_list_evaluate(search, move_list, depth, alpha, beta, &searching);
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (v == -SCORE_INF)
                        g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                        if (alpha <= g && g < beta)
                            g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
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
                    if (best_move != move_list[move_idx].flip.pos)
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value <= " << g << std::endl;
                    else
                        std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " best " << idx_to_coord(best_move) << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].flip.pos) << " value = " << g << std::endl;
                }
                ++pv_idx;
            }
        }
    }
    if (global_searching)
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    return v;
}