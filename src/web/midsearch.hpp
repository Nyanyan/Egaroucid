/*
    Egaroucid Project

    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <vector>
#include <future>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "endsearch.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "util.hpp"
#include "stability.hpp"

using namespace std;

inline int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    int g, v = -INF;
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
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        eval_move(search, &flip);
        search->move(&flip);
            g = -mid_evaluate_diff(search);
        search->undo(&flip);
        eval_undo(search, &flip);
        ++search->n_nodes;
        alpha = max(alpha, g);
        v = max(v, g);
        if (beta <= alpha)
            break;
    }
    return v;
}

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
        if (*searching){
            alpha = max(alpha, g);
            v = max(v, g);
            if (beta <= alpha)
                break;
        } else
            return SCORE_UNDEFINED;
    }
    return v;
}

int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    //if (depth <= MID_FAST_DEPTH)
    //    return nega_alpha(search, alpha, beta, depth, skipped, searching);
    if (depth == 1)
        return nega_alpha_eval1(search, alpha, beta, skipped, searching);
    if (depth == 0)
        return mid_evaluate_diff(search);
    ++(search->n_nodes);
    int first_alpha = alpha;
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_alpha_ordering_nomemo(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    #if USE_MID_MPC
        if (search->use_mpc){
            if (mpc(search, alpha, beta, depth, legal, false, &v, searching))
                return v;
        }
    #endif
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    int best_move = child_transpose_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                g = -nega_alpha_ordering_nomemo(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, searching);
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
        //move_ordering(search, move_list, depth, alpha, beta, false, searching);
        move_list_evaluate(search, move_list, depth, alpha, beta, false, searching);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, &move_list[move_idx].flip);
            search->move(&move_list[move_idx].flip);
                g = -nega_alpha_ordering_nomemo(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, searching);
            search->undo(&move_list[move_idx].flip);
            eval_undo(search, &move_list[move_idx].flip);
            if (*searching){
                alpha = max(alpha, g);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                }
                if (beta <= alpha)
                    break;
            } else
                return SCORE_UNDEFINED;
        }
    }
    if (first_alpha < v)
        child_transpose_table.reg(&search->board, hash_code, best_move);
    return v;
}

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
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
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
    register_tt(search, depth, hash_code, first_alpha, v, best_move, l, u, alpha, beta, searching);
    return v;
}

int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end(search, alpha, beta, skipped, legal, searching);
    //if (!is_end_search && depth <= MID_FAST_DEPTH)
    //    return nega_alpha(search, alpha, beta, depth, skipped, searching);
    if (!is_end_search){
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, beta, skipped, searching);
        if (depth == 0)
            return mid_evaluate_diff(search);
    }
    ++(search->n_nodes);
    #if USE_END_SC
        if (is_end_search){
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED)
                return stab_res;
        }
    #endif
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
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
                g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
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
                    g = -nega_alpha_ordering(search, -alpha - 1, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                    if (alpha < g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                }
            search->undo(&move_list[move_idx].flip);
            eval_undo(search, &move_list[move_idx].flip);
            if (v < g){
                v = g;
                best_move = move_list[move_idx].flip.pos;
                alpha = max(alpha, g);
            }
            if (beta <= alpha)
                break;
        }
    }
    register_tt(search, depth, hash_code, first_alpha, v, best_move, l, u, alpha, beta, searching);
    return v;
}

pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, bool is_end_search, const bool is_main_search, int best_move){
    bool searching = true;
    ++(search->n_nodes);
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
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
        //move_ordering(search, move_list, depth, alpha, beta, is_end_search, &searching);
        //for (const Flip_value &flip_value: move_list){
        move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, &searching);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, &move_list[move_idx].flip);
            search->move(&move_list[move_idx].flip);
                if (v == -INF)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                else{
                    g = -nega_alpha_ordering(search, -alpha - 1, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    if (alpha < g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                }
                if (is_main_search){
                    if (g <= alpha)
                        cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)move_list[move_idx].flip.pos) << " value " << g << " or lower" << endl;
                    else
                        cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)move_list[move_idx].flip.pos) << " value " << g << endl;
                }
                ++mobility_idx;
            search->undo(&move_list[move_idx].flip);
            eval_undo(search, &move_list[move_idx].flip);
            if (v < g){
                v = g;
                best_move = move_list[move_idx].flip.pos;
                alpha = max(alpha, g);
            }
            if (beta <= alpha)
                break;
        }
    }
    register_tt(search, depth, hash_code, first_alpha, v, best_move, alpha, beta, alpha, beta, &searching);
    return make_pair(v, best_move);
}