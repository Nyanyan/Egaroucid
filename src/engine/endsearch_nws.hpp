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


#if USE_SIMD
    #include "endsearch_nws_last_simd.hpp"
#else
    #include "endsearch_nws_last_generic.hpp"
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
        search->pass_noeval();
            int v = -nega_alpha_end_fast_nws(search, -alpha - 1, true, searching);
        search->pass_noeval();
        return v;
    }

    Board board0;
    search->board.copy(&board0);
    int v = -SCORE_INF;
    int g;
    Flip flip;
    uint_fast8_t cell;
    uint64_t prioritymoves = legal;
    #if USE_END_PO
        prioritymoves &= empty1_bb(search->board.player, search->board.opponent);
        if (prioritymoves == 0)
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
                g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, searching);
                if (alpha < g) {
                    --search->n_discs;
                    board0.copy(&search->board);
                    return g;
                }
                if (v < g)
                    v = g;
            }
        } while ((prioritymoves = legal));
        --search->n_discs;
    }
    board0.copy(&search->board);
    return v;
}

struct LocalTTEntry {
    uint64_t player;
    uint64_t opponent;
    int lower;
    int upper;

    bool cmp(Board *board) {
        return board->player == player && board->opponent == opponent;
    }

    void set_score(Board *board, int l, int u) {
        player = board->player;
        opponent = board->opponent;
        lower = l;
        upper = u;
    }
};

#define LOCAL_TT_SIZE 1024
#define LOCAL_TT_SIZE_BIT 10

static thread_local LocalTTEntry lttable[MID_TO_END_DEPTH - END_FAST_DEPTH][LOCAL_TT_SIZE];

inline uint32_t hash_bb(Board *board)
{
	return ((board->player * 0x9dda1c54cfe6b6e9ull) ^ (board->opponent * 0xa2e6c0300831e05aull)) >> (HW2 - LOCAL_TT_SIZE_BIT);
}

inline LocalTTEntry *get_ltt(Board *board, uint32_t n_discs)
{
	return lttable[HW2 - n_discs - END_FAST_DEPTH] + hash_bb(board);
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
        search->pass_noeval();
            v = -nega_alpha_end_simple_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->pass_noeval();
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
    move_list_evaluate_end_simple_nws(search, move_list, canput);
    int g;
    for (int move_idx = 0; move_idx < canput && *searching; ++move_idx){
        if (move_idx < 4)
            swap_next_best_move(move_list, move_idx, canput);
        search->move_noeval(&move_list[move_idx].flip);
            Board nboard = search->board;
            LocalTTEntry *tt = get_ltt(&nboard, search->n_discs);
            if (tt->cmp(&nboard)) {
                if (alpha < tt->lower) {
                    v = tt->lower;
                    search->undo_noeval(&move_list[move_idx].flip);
                    break;
                }
                if (tt->upper <= alpha) {
                    if (v < tt->upper) {
                        v = tt->upper;
                    }
                    search->undo_noeval(&move_list[move_idx].flip);
                    continue;
                }
            }
            g = -nega_alpha_end_simple_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
        search->undo_noeval(&move_list[move_idx].flip);
        if (v < g){
            v = g;
            if (alpha < v) {
                tt->set_score(&nboard, v, 64);
                break;
            }
        }
        tt->set_score(&nboard, -64, g);
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
        search->pass_endsearch();
            v = -nega_alpha_end_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->pass_endsearch();
        return v;
    }
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    /*
    uint32_t hash_code = search->board.hash();
    if (transposition_cutoff_nws(search, hash_code, HW2 - search->n_discs, alpha, &v, moves)){
        return v;
    }
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
    move_list_evaluate_end_nws(search, move_list, moves, searching);
    for (int move_idx = 0; move_idx < canput; ++move_idx){
        /*
        if (search->need_to_see_tt_loop){
            if (transposition_cutoff_nws(search, hash_code, HW2 - search->n_discs, alpha, &v, moves)){
                return v;
            }
        }
        */
        swap_next_best_move(move_list, move_idx, canput);
        search->move_endsearch(&move_list[move_idx].flip);
            Board nboard = search->board;
            LocalTTEntry *tt = get_ltt(&nboard, search->n_discs);
            if (tt->cmp(&nboard)) {
                if (alpha < tt->lower) {
                    best_move = move_list[move_idx].flip.pos;
                    v = tt->lower;
                    search->undo_endsearch(&move_list[move_idx].flip);
                    break;
                }
                if (tt->upper <= alpha) {
                    if (v < tt->upper) {
                        v = tt->upper;
                    }
                    search->undo_endsearch(&move_list[move_idx].flip);
                    continue;
                }
            }
            g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
        search->undo_endsearch(&move_list[move_idx].flip);
        if (v < g){
            v = g;
            best_move = move_list[move_idx].flip.pos;
            if (alpha < v) {
                tt->set_score(&nboard, v, 64);
                break;
            }
        }
        tt->set_score(&nboard, -64, g);
    }
    /*
    if (*searching && global_searching){
        transposition_table.reg(search, hash_code, HW2 - search->n_discs, alpha, alpha + 1, v, best_move);
    }
    */
    return v;
}
