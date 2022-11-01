/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
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
#include "transpose_table.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "thread_pool.hpp"
#include "ybwc.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "endsearch.hpp"
#include "midsearch_nws.hpp"

using namespace std;

#if MID_FAST_DEPTH > 1
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        ++search->n_nodes;
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, beta, skipped, searching);
        if (depth == 0)
            return mid_evaluate_diff(search);
        int g, v = -INF;
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

/*
int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end(search, alpha, beta, skipped, legal, searching);
    if (!is_end_search){
        //if (depth <= MID_FAST_DEPTH)
        //    return nega_alpha(search, alpha, beta, depth, skipped, searching);
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, beta, skipped, searching);
        if (depth == 0)
            return mid_evaluate_diff(search);
    }
    ++search->n_nodes;
    uint32_t hash_code = search->board.hash();
    int l = -INF, u = INF;
    if (depth >= USE_TT_DEPTH_THRESHOLD){
        parent_transpose_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
        if (u == l)
            return u;
        if (beta <= l)
            return l;
        if (u <= alpha)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    }
    int first_alpha = alpha;
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int g, v = -INF;
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
    #if USE_MID_MPC
        if (search->use_mpc){
            if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                return v;
        }
    #endif
    int best_move = child_transpose_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            if (*searching){
                alpha = max(alpha, g);
                v = g;
                legal ^= 1ULL << best_move;
            } else
                return SCORE_UNDEFINED;
        } else
            best_move = TRANSPOSE_TABLE_UNDEFINED;
    }
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
        if (search->use_multi_thread){
            int pv_idx = 0, split_count = 0;
            if (best_move != TRANSPOSE_TABLE_UNDEFINED)
                pv_idx = 1;
            vector<future<Parallel_task>> parallel_tasks;
            bool n_searching = true;
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                if (!(*searching))
                    break;
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split(search, &move_list[move_idx].flip, -beta, -alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, canput, pv_idx++, split_count, parallel_tasks)){
                        ++split_count;
                    } else{
                        g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (*searching){
                            alpha = max(alpha, g);
                            if (v < g){
                                v = g;
                                best_move = move_list[move_idx].flip.pos;
                            }
                            if (beta <= alpha){
                                search->undo(&move_list[move_idx].flip);
                                eval_undo(search, &move_list[move_idx].flip);
                                break;
                            }
                            if (split_count){
                                ybwc_get_end_tasks(search, parallel_tasks, &v, &best_move, &alpha);
                                if (beta <= alpha){
                                    search->undo(&move_list[move_idx].flip);
                                    eval_undo(search, &move_list[move_idx].flip);
                                    break;
                                }
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
            }
            if (split_count){
                if (beta <= alpha || !(*searching)){
                    n_searching = false;
                    ybwc_wait_all(search, parallel_tasks);
                } else
                    ybwc_wait_all(search, parallel_tasks, &v, &best_move, &alpha, beta, &n_searching);
            }
        } else{
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                if (!(*searching))
                    break;
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    if (*searching){
                        alpha = max(alpha, g);
                        if (v < g){
                            v = g;
                            best_move = move_list[move_idx].flip.pos;
                        }
                        if (beta <= alpha){
                            search->undo(&move_list[move_idx].flip);
                            eval_undo(search, &move_list[move_idx].flip);
                            break;
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
            }
        }
    }
    register_tt(search, depth, hash_code, first_alpha, v, best_move, l, u, alpha, beta, searching);
    return v;
}
*/

int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
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
    ++(search->n_nodes);
    #if USE_END_SC
        if (is_end_search){
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED)
                return stab_res;
        }
    #endif
    uint32_t hash_code = search->board.hash();
    int l = -INF, u = INF;
    if (depth >= USE_TT_DEPTH_THRESHOLD){
        parent_transpose_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
        if (u == l)
            return u;
        if (beta <= l)
            return l;
        if (u <= alpha)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    }
    int first_alpha = alpha;
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -INF;
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
    #if USE_MID_MPC
        if (search->use_mpc){
            if (!is_end_search || (is_end_search && depth <= USE_MPC_ENDSEARCH_DEPTH)){
                if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                    return v;
            }
        }
    #endif
    int best_move = child_transpose_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                v = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            alpha = max(alpha, v);
            legal ^= 1ULL << best_move;
        } else
            best_move = TRANSPOSE_TABLE_UNDEFINED;
    }
    int g;
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, &move_list[move_idx].flip);
            search->move(&move_list[move_idx].flip);
                if (v == -INF)
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
    }
    register_tt(search, depth, hash_code, v, best_move, l, u, first_alpha, beta, searching);
    return v;
}

pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, bool is_end_search, const bool is_main_search, int best_move){
    bool searching = true;
    ++(search->n_nodes);
    uint32_t hash_code = search->board.hash();
    uint64_t legal = search->board.get_legal();
    int first_alpha = alpha;
    int g, v = -INF;
    if (legal == 0ULL){
        pair<int, int> res;
        if (skipped){
            res.first = end_evaluate(&search->board);
            res.second = -1;
        } else{
            search->eval_feature_reversed ^= 1;
            search->board.pass();
                res = first_nega_scout(search, -beta, -alpha, depth, true, is_end_search, is_main_search, best_move);
            search->board.pass();
            search->eval_feature_reversed ^= 1;
            res.first = -res.first;
        }
        return res;
    }
    const int canput_all = pop_count_ull(legal);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                if (is_main_search)
                    cerr << 1 << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord(best_move) << " value " << g << endl;
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            v = g;
            alpha = max(alpha, g);
            legal ^= 1ULL << best_move;
        } else
            best_move = TRANSPOSE_TABLE_UNDEFINED;
    }
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        int mobility_idx = (v == -INF) ? 1 : 2;
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_ordering(search, move_list, depth, alpha, beta, is_end_search, &searching);
        for (const Flip_value &flip_value: move_list){
            eval_move(search, &flip_value.flip);
            search->move(&flip_value.flip);
                if (v == -INF)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, flip_value.n_legal, is_end_search, &searching);
                else{
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, flip_value.n_legal, is_end_search, &searching);
                    if (alpha <= g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, flip_value.n_legal, is_end_search, &searching);
                }
                if (is_main_search){
                    if (g <= alpha)
                        cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)flip_value.flip.pos) << " value " << g << " or lower" << endl;
                    else
                        cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)flip_value.flip.pos) << " value " << g << endl;
                }
                ++mobility_idx;
            search->undo(&flip_value.flip);
            eval_undo(search, &flip_value.flip);
            if (v < g){
                v = g;
                best_move = flip_value.flip.pos;
                if (alpha < v){
                    if (beta <= v)
                        break;
                    alpha = v;
                }
            }
        }
    }
    register_tt(search, depth, hash_code, v, best_move, first_alpha, beta, first_alpha, beta, &searching);
    return make_pair(v, best_move);
}
