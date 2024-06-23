/*
    Egaroucid Project

    @file midsearch_light.hpp
        Search midgame with light evaluation function
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
//#include "probcut.hpp"
#include "util.hpp"
//#include "etc.hpp"

inline void swap_next_best_move(std::vector<Flip_value> &move_list, const int strt, const int siz);
inline void move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, const bool *searching);
inline void move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, const bool *searching);

inline int nega_alpha_light_eval1(Search *search, int alpha, int beta, bool skipped){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass_light();
            v = -nega_alpha_light_eval1(search, -beta, -alpha, true);
        search->pass_light();
        return v;
    }
    int g;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->move_light(&flip);
            ++search->n_nodes;
            g = -mid_evaluate_light(search);
        search->undo_light(&flip);
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

int nega_alpha_light(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (depth == 1)
        return nega_alpha_light_eval1(search, alpha, beta, skipped);
    if (depth == 0){
        ++search->n_nodes;
        return mid_evaluate_light(search);
    }
    ++search->n_nodes;
    int first_alpha = alpha;
    int first_beta = beta;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -SCORE_INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass_light();
            v = -nega_alpha_light(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED, searching);
        search->pass_light();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    if (transposition_cutoff(search, hash_code, depth, &alpha, &beta, &v, moves)){
        return v;
    }
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
    /*
    #if USE_MID_ETC
        if (depth >= MID_ETC_DEPTH){
            if (etc(search, move_list, depth, &alpha, &beta, &v, &etc_done_idx)){
                return v;
            }
        }
    #endif
    */
    /*
    #if USE_MID_MPC
        if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching)){
            return v;
        }
    #endif
    */
    int g;
    //int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    move_list_evaluate(search, move_list, moves, depth, alpha, beta, searching);
    for (int move_idx = 0; move_idx < canput - etc_done_idx && *searching; ++move_idx){
        swap_next_best_move(move_list, move_idx, canput);
        #if USE_MID_ETC
            if (move_list[move_idx].flip.flip == 0)
                break;
        #endif
        search->move_light(&move_list[move_idx].flip);
            g = -nega_alpha_light(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, searching);
        search->undo_light(&move_list[move_idx].flip);
        if (v < g){
            v = g;
            //best_move = move_list[move_idx].flip.pos;
            if (alpha < v){
                if (beta <= v)
                    break;
                alpha = v;
            }
        }
    }
    /*
    if (*searching && global_searching){
        transposition_table.reg_best_move(search, hash_code, best_move);
    }
    */
    return v;
}

int nega_alpha_light_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (depth == 1)
        return nega_alpha_light_eval1(search, alpha, alpha + 1, skipped);
    if (depth == 0){
        ++search->n_nodes;
        return mid_evaluate_light(search);
    }
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -SCORE_INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass_light();
            v = -nega_alpha_light_nws(search, -alpha - 1, depth, true, LEGAL_UNDEFINED, searching);
        search->pass_light();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    if (transposition_cutoff_nws(search, hash_code, depth, alpha, &v, moves)){
        return v;
    }
    /*
    #if USE_MID_MPC
        if (mpc(search, alpha, alpha + 1, depth, legal, is_end_search, &v, searching)){
            return v;
        }
    #endif
    */
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    int g;
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
    /*
    #if USE_MID_ETC
        if (depth >= MID_ETC_DEPTH){
            if (etc_nws(search, move_list, depth, alpha, &v, &etc_done_idx)){
                return v;
            }
        }
    #endif
    */
    move_list_evaluate_nws(search, move_list, moves, depth, alpha, searching);
    for (int move_idx = 0; move_idx < canput - etc_done_idx && *searching; ++move_idx){
        swap_next_best_move(move_list, move_idx, canput);
        #if USE_MID_ETC
            if (move_list[move_idx].flip.flip == 0)
                break;
        #endif
        search->move_light(&move_list[move_idx].flip);
            g = -nega_alpha_light_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, searching);
        search->undo_light(&move_list[move_idx].flip);
        if (v < g){
            v = g;
            best_move = move_list[move_idx].flip.pos;
            if (alpha < v)
                break;
        }
    }
    /*
    if (*searching && global_searching)
        transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, best_move);
    */
    return v;
}
