/*
    Egaroucid Project

    @file endsearch_nws.hpp
        Search near endgame with NWS (Null Window Search)
        last2/3/4_nws imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2024
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "transposition_table.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "endsearch_common.hpp"
#include "parallel.hpp"
#include "ybwc.hpp"

#if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
    inline bool ybwc_split_end_nws(const Search *search, int alpha, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const int split_count, std::vector<std::future<Parallel_task>> &parallel_tasks);
    inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count);
    inline void ybwc_wait_all(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks);
    inline void ybwc_wait_all_nws(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count, int alpha, const bool *searching, bool *n_searching);
#endif

#if USE_SIMD
#include "endsearch_nws_simd.hpp"
#else
/*
    @brief Get a final min score with last 2 empties (NWS)

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param alpha                alpha value (beta value is alpha + 1)
    @param p0                   empty square 1/2
    @param p1                   empty square 2/2
    @param board                bitboard
    @return the final min score
*/
static int last2_nws(Search *search, int alpha, uint_fast8_t p0, uint_fast8_t p1, Board board) {
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    int v;
    Flip flip;
    //if ((bit_around[p0] & board.player) == 0)
    //    std::swap(p0, p1);
    if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) {
        v = last1(search, board.opponent ^ flip.flip, alpha, p1);

        if ((v > alpha) && (bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            int g = last1(search, board.opponent ^ flip.flip, alpha, p0);
            if (v > g)
                v = g;
        }
    }

    else if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1))
        v = last1(search, board.opponent ^ flip.flip, alpha, p0);

    else {	// pass
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[62];
        #endif
        alpha = -alpha - 1;
        if (flip.calc_flip(board.opponent, board.player, p0)) {
            v = last1(search, board.player ^ flip.flip, alpha, p1);

            if ((v > alpha) && (flip.calc_flip(board.opponent, board.player, p1))) {
                int g = last1(search, board.player ^ flip.flip, alpha, p0);
                if (v > g)
                    v = g;
            }
        }

        else if (flip.calc_flip(board.opponent, board.player, p1))
            v = last1(search, board.player ^ flip.flip, alpha, p0);

        else	// gameover
            v = end_evaluate(&board, 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final max score with last 3 empties (NWS)

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param sort3                parity sort (lower 2 bits for this node)
    @param p0                   empty square 1/3
    @param p1                   empty square 2/3
    @param p2                   empty square 3/3
    @param board                bitboard
    @return the final max score

    This board contains only 3 empty squares, so empty squares on each part will be:
        3 - 0 - 0 - 0
        2 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 0
    then the parities for squares will be:
        1 - 1 - 1
        1 - 0 - 0 > need to sort
        1 - 1 - 1
*/
static int last3_nws(Search *search, int alpha, int sort3, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, Board board) {
    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_END_PO
        switch (sort3 & 3){
            case 2:
                std::swap(p0, p2);
                break;
            case 1:
                std::swap(p0, p1);
                break;
        }
    #endif

    Flip flip;
    Board board2;
    int v = -SCORE_INF;
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[61];
        #endif
        if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) {
            board.move_copy(&flip, &board2);
            v = last2_nws(search, alpha, p1, p2, board2);
            if (alpha < v)
                return v * pol;
        }

        int g;
        if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            board.move_copy(&flip, &board2);
            g = last2_nws(search, alpha, p0, p2, board2);
            if (alpha < g)
                return g * pol;
            if (v < g)
                v = g;
        }

        if ((bit_around[p2] & board.opponent) && calc_flip(&flip, &board, p2)) {
            board.move_copy(&flip, &board2);
            g = last2_nws(search, alpha, p0, p1, board2);
            if (v < g)
                v = g;
            return v * pol;
        }

        if (v > -SCORE_INF)
            return v * pol;

        board.pass();
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate_odd(&board, 3);	// gameover
}

/*
    @brief Get a final min score with last 4 empties (NWS)

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @return the final min score

    This board contains only 4 empty squares, so empty squares on each part will be:
        4 - 0 - 0 - 0
        3 - 1 - 0 - 0
        2 - 2 - 0 - 0
        2 - 1 - 1 - 0 > need to sort
        1 - 1 - 1 - 1
    then the parities for squares will be:
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 1
*/
int last4_nws(Search *search, int alpha) {
    uint64_t empties = ~(search->board.player | search->board.opponent);
    uint_fast8_t p0 = first_bit(&empties);
    uint_fast8_t p1 = next_bit(&empties);
    uint_fast8_t p2 = next_bit(&empties);
    uint_fast8_t p3 = next_bit(&empties);

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_LAST4_SC
        int stab_res = stability_cut_last4_nws(search, alpha);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    #endif
    #if USE_END_PO
                // parity ordering optimization
                // I referred to http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm
        const int paritysort = parity_case[((p2 ^ p3) & 0x24) + ((((p1 ^ p3) & 0x24) * 2 + ((p0 ^ p3) & 0x24)) >> 2)];
        switch (paritysort) {
            case 8:     // case 1(p2) 1(p3) 2(p0 p1)
                std::swap(p0, p2);
                std::swap(p1, p3);
                break;
            case 7:     // case 1(p1) 1(p3) 2(p0 p2)
                std::swap(p0, p3);
                break;
            case 6:     // case 1(p1) 1(p2) 2(p0 p3)
                std::swap(p0, p2);
                break;
            case 5:     // case 1(p0) 1(p3) 2(p1 p2)
                std::swap(p1, p3);
                break;
            case 4:     // case 1(p0) 1(p2) 2(p1 p3)
                std::swap(p1, p2);
                break;
        }
        int sort3 = parity_ordering_last3[paritysort];     // for move sorting on 3 empties
    #else
        constexpr int sort3 = 0;
    #endif

    Flip flip;
    Board board3, board4;
    search->board.copy(&board4);
    int v = SCORE_INF;	// min stage
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[60];
        #endif
        if ((bit_around[p0] & board4.opponent) && calc_flip(&flip, &board4, p0)) {
            board4.move_copy(&flip, &board3);
            v = last3_nws(search, alpha, sort3, p1, p2, p3, board3);
            if (alpha >= v)
                return v * pol;
        }

        int g;
        if ((bit_around[p1] & board4.opponent) && calc_flip(&flip, &board4, p1)) {
            board4.move_copy(&flip, &board3);
            g = last3_nws(search, alpha, sort3 >> 4, p0, p2, p3, board3);
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
        }

        if ((bit_around[p2] & board4.opponent) && calc_flip(&flip, &board4, p2)) {
            board4.move_copy(&flip, &board3);
            g = last3_nws(search, alpha, sort3 >> 8, p0, p1, p3, board3);
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
        }

        if ((bit_around[p3] & board4.opponent) && calc_flip(&flip, &board4, p3)) {
            board4.move_copy(&flip, &board3);
            g = last3_nws(search, alpha, sort3 >> 12, p0, p1, p2, board3);
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        board4.pass();
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return -end_evaluate(&search->board, 4);	// gameover
}
#endif

/*
    @brief Get a final score with few empties (NWS)

    Only with parity-based ordering.
    imported from search_shallow of Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param searching            flag for terminating this search
    @return the final score
*/
int nega_alpha_end_fast_nws(Search *search, int alpha, bool skipped, const bool *searching) {
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_END_SC
        if (!skipped){
            int stab_res = stability_cut_nws(search, alpha);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
        }
    #endif
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            int v = -nega_alpha_end_fast_nws(search, -alpha - 1, true, searching);
        search->board.pass();
        return v;
    }

    Board board0;
    search->board.copy(&board0);
    int v = -SCORE_INF;
    uint_fast8_t parity0 = search->parity;
    int g;
    Flip flip;
    uint_fast8_t cell;
    uint64_t prioritymoves = legal;
    #if USE_END_PO
        prioritymoves &= parity_table[parity0];
        if (prioritymoves == 0) // all even
            prioritymoves = legal;
    #endif

    if (search->n_discs == 59)      // transfer to lastN, no longer uses n_discs, parity
        do {
            legal ^= prioritymoves;
            for (cell = first_bit(&prioritymoves); prioritymoves; cell = next_bit(&prioritymoves)) {
                calc_flip(&flip, &board0, cell);
                board0.move_copy(&flip, &search->board);
                g = last4_nws(search, alpha);
                if (alpha < g) {
                    board0.copy(&search->board);
                    return g;
                }
                if (v < g)
                    v = g;
            }
        } while ((prioritymoves = legal));

   else {
        ++search->n_discs;  // for next depth
        do {
            legal ^= prioritymoves;
            for (cell = first_bit(&prioritymoves); prioritymoves; cell = next_bit(&prioritymoves)) {
                calc_flip(&flip, &board0, cell);
                board0.move_copy(&flip, &search->board);
                search->parity = parity0 ^ cell_div4[cell];
                g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, searching);
                if (alpha < g) {
                    --search->n_discs;
                    search->parity = parity0;
                    board0.copy(&search->board);
                    return g;
                }
                if (v < g)
                    v = g;
            }
        } while ((prioritymoves = legal));
        --search->n_discs;
        search->parity = parity0;
    }
    board0.copy(&search->board);
    return v;
}

/*
    @brief Get a final score with some empties (NWS)

    Search with move ordering for endgame and transposition tables.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param searching            flag for terminating this search
    @return the final score
*/
int nega_alpha_end_simple_nws(Search *search, int alpha, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (search->n_discs >= HW2 - END_FAST_DEPTH)
        return nega_alpha_end_fast_nws(search, alpha, skipped, searching);
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_END_SC
        if (!skipped){
            int stab_res = stability_cut_nws(search, alpha);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
        }
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -SCORE_INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_simple_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        return v;
    }
    const int canput = pop_count_ull(legal);
    Flip_value move_list[END_SIMPLE_DEPTH];
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent)
            return SCORE_MAX;
        ++idx;
    }
    move_list_evaluate_end_nws(search, move_list, canput);
    int g;
    for (int move_idx = 0; move_idx < canput; ++move_idx){
        if (move_idx < 4)
            swap_next_best_move(move_list, move_idx, canput);
        search->move(&move_list[move_idx].flip);
            g = -nega_alpha_end_simple_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
        search->undo(&move_list[move_idx].flip);
        if (v < g){
            v = g;
            if (alpha < v)
                break;
        }
    }
    return v;
}

/*
    @brief Get a final score with some empties (NWS)

    Search with move ordering for endgame and transposition tables.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param searching            flag for terminating this search
    @return the final score
*/
int nega_alpha_end_nws(Search *search, int alpha, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (search->n_discs >= HW2 - END_SIMPLE_DEPTH)
        return nega_alpha_end_simple_nws(search, alpha, skipped, legal, searching);
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_END_SC
        if (!skipped){
            int stab_res = stability_cut_nws(search, alpha);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
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
            v = -nega_alpha_end_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    uint32_t hash_code = search->board.hash();
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    transposition_table.get(search, hash_code, HW2 - search->n_discs, &lower, &upper, moves);
    if (upper == lower)
        return upper;
    if (alpha < lower)
        return lower;
    if (upper <= alpha)
        return upper;
    Flip flip_best;
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
    move_list_evaluate_end_nws(search, move_list, canput, moves);
    #if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
        #if USE_ALL_NODE_PREDICTION
            const bool seems_to_be_all_node = predict_all_node(search, alpha, HW2 - search->n_discs, LEGAL_UNDEFINED, true, searching);
        #else
            constexpr bool seems_to_be_all_node = false;
        #endif
        if (search->use_multi_thread && search->n_discs <= HW2 - YBWC_END_SPLIT_MIN_DEPTH){
            int running_count = 0;
            std::vector<std::future<Parallel_task>> parallel_tasks;
            bool n_searching = true;
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                eval_move(search, &move_list[move_idx].flip);
                    if (ybwc_split_end_nws(search, -alpha - 1, move_list[move_idx].n_legal, &n_searching, move_list[move_idx].flip.pos, canput, move_idx, seems_to_be_all_node, running_count, parallel_tasks)){
                        ++running_count;
                    } else{
                        g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, move_idx > FAIL_HIGH_WISH_THRESHOLD_END_NWS, searching);
                        if (v < g){
                            v = g;
                            if (alpha < v){
                                best_move = move_list[move_idx].flip.pos;
                                eval_undo(search, &move_list[move_idx].flip);
                                search->undo(&move_list[move_idx].flip);
                                break;
                            }
                        }
                        if (running_count){
                            ybwc_get_end_tasks(search, parallel_tasks, &v, &best_move, &running_count);
                            if (alpha < v){
                                eval_undo(search, &move_list[move_idx].flip);
                                search->undo(&move_list[move_idx].flip);
                                break;
                            }
                        }
                    }
                eval_undo(search, &move_list[move_idx].flip);
                search->undo(&move_list[move_idx].flip);
            }
            if (running_count){
                if (alpha < v || !(*searching)){
                    n_searching = false;
                    ybwc_wait_all(search, parallel_tasks);
                } else
                    ybwc_wait_all_nws(search, parallel_tasks, &v, &best_move, &running_count, alpha, searching, &n_searching);
            }
        } else{
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                eval_move(search, &move_list[move_idx].flip);
                    g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, move_idx > FAIL_HIGH_WISH_THRESHOLD_END_NWS, searching);
                eval_undo(search, &move_list[move_idx].flip);
                search->undo(&move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v)
                        break;
                }
            }
        }
    #else
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            search->move(&move_list[move_idx].flip);
            eval_move(search, &move_list[move_idx].flip);
                g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
            eval_undo(search, &move_list[move_idx].flip);
            search->undo(&move_list[move_idx].flip);
            if (v < g){
                v = g;
                best_move = move_list[move_idx].flip.pos;
                if (alpha < v)
                    break;
            }
        }
    #endif
    if (*searching && global_searching){
        transposition_table.reg(search, hash_code, HW2 - search->n_discs, alpha, alpha + 1, v, best_move);
    }
    return v;
}