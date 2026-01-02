/*
    Egaroucid Project

    @file midsearch_nws.hpp
        Search midgame with NWS (Null Window Search)
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
#include "endsearch.hpp"
#include "move_ordering.hpp"
#include "multi_probcut.hpp"
#include "thread_pool.hpp"
#include "ybwc.hpp"
#include "util.hpp"
#include "stability_cutoff.hpp"

inline bool mpc(Search* search, int alpha, int beta, int depth, uint64_t legal, const bool is_end_search, int* v, std::vector<bool*> &searchings);
inline bool mpc(Search* search, int alpha, int beta, const int depth, uint64_t legal, const bool is_end_search, int* v, const bool* searching);

/*
    @brief Get a value with last move with Nega-Alpha algorithm (NWS)

    No move ordering. Just search it.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @return the value
*/
inline int nega_alpha_eval1_nws(Search *search, int alpha, const bool skipped) {
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
            v = -nega_alpha_eval1_nws(search, -alpha - 1, true);
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
                    return g;
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
    //             return g;
    //         }
    //         v = g;
    //     }
    // }
    return v;
}



int nega_alpha_eval2_nws(Search *search, int alpha, const bool skipped, uint64_t legal, bool *searching) {
    if (!global_searching || !(*searching)) {
        return SCORE_UNDEFINED;
    }
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
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
            v = -nega_alpha_eval2_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    transposition_table.prefetch(hash_code);
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    if (transposition_cutoff_nws(search, hash_code, 2, alpha, &v, moves)) {
        return v;
    }
    int best_move = MOVE_UNDEFINED;
    int g;
    Flip flip;
    for (int i = 0; i < N_TRANSPOSITION_MOVES; ++i) {
        if (moves[i] != MOVE_UNDEFINED) {
            calc_flip(&flip, &search->board, moves[i]);
            search->move(&flip);
                g = -nega_alpha_eval1_nws(search, -alpha - 1, false);
            search->undo(&flip);
            if (!(*searching)) {
                return SCORE_UNDEFINED;
            }
            legal ^= 1ULL << moves[i];
            if (v < g) {
                v = g;
                best_move = moves[i];
                if (alpha < g) {
                    break;
                }
            }
        }
    }
    for (int i = 0; i < N_STATIC_CELL_PRIORITY && v <= alpha; ++i) {
        uint64_t l = legal & static_cell_priority[i];
        for (uint_fast8_t cell = first_bit(&l); l; cell = next_bit(&l)) {
            calc_flip(&flip, &search->board, cell);
            search->move(&flip);
                g = -nega_alpha_eval1_nws(search, -alpha - 1, false);
            search->undo(&flip);
            if (!(*searching)) {
                return SCORE_UNDEFINED;
            }
            if (v < g) {
                v = g;
                best_move = cell;
                if (alpha < g) {
                    break;
                }
            }
        }
    }
    // if (*searching && global_searching) {
    //     transposition_table.reg(search, hash_code, 2, alpha, alpha + 1, v, best_move);
    // }
    return v;
}


/*
    @brief Get a value with given depth with Nega-Alpha algorithm (NWS)

    Search with move ordering for midgame NWS (no YBWC)

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param depth                remaining depth
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param searching            flag for terminating this search
    @return the value
*/
int nega_alpha_ordering_nws_simple(Search *search, int alpha, const int depth, const bool skipped, uint64_t legal, bool *searching) {
    if (!global_searching || !(*searching)) {
        return SCORE_UNDEFINED;
    }
    if (depth == 2) {
        return nega_alpha_eval2_nws(search, alpha, skipped, legal, searching);
    }
    if (depth == 1) {
        return nega_alpha_eval1_nws(search, alpha, skipped);
    }
    if (depth == 0) {
        ++search->n_nodes;
        return mid_evaluate_diff(search);
    }
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
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
            v = -nega_alpha_ordering_nws_simple(search, -alpha - 1, depth, true, LEGAL_UNDEFINED, searching);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    transposition_table.prefetch(hash_code);
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    if (transposition_cutoff_nws(search, hash_code, depth, alpha, &v, moves)) {
        return v;
    }
#if USE_MID_MPC && MID_MPC_MIN_DEPTH <= MID_SIMPLE_ORDERING_DEPTH
    if (search->mpc_level < MPC_100_LEVEL && depth >= USE_MPC_MIN_DEPTH) {
        if (mpc(search, alpha, alpha + 1, depth, legal, false, &v, searching)) {
            return v;
        }
    }
#endif
    int best_move = MOVE_UNDEFINED;
    int g;
    const int canput = pop_count_ull(legal);
    Flip_value move_list[MAX_N_BRANCHES];
    // std::vector<Flip_value> move_list(canput);
    int idx = 0;
    int tt_moves_idx0 = -1;
    // int tt_moves_idx1 = -1;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent) {
            return SCORE_MAX;
        }
        if (cell == moves[0]) {
            tt_moves_idx0 = idx;
        }
        // else if (cell == moves[1]) {
        //     tt_moves_idx1 = idx;
        // }
        ++idx;
    }
#if USE_MID_ETC && MID_ETC_DEPTH_NWS <= MID_SIMPLE_ORDERING_DEPTH
    int n_etc_done = 0;
    if (depth >= MID_ETC_DEPTH_NWS) {
        if (etc_nws(search, move_list, canput, depth, alpha, &v, &n_etc_done)) {
            return v;
        }
    }
#endif
    if (tt_moves_idx0 != -1 && move_list[tt_moves_idx0].flip.flip) {
        search->move(&move_list[tt_moves_idx0].flip);
            g = -nega_alpha_ordering_nws_simple(search, -alpha - 1, depth - 1, false, move_list[tt_moves_idx0].n_legal, searching);
        search->undo(&move_list[tt_moves_idx0].flip);
        if (v < g) {
            v = g;
            best_move = move_list[tt_moves_idx0].flip.pos;
        }
        move_list[tt_moves_idx0].flip.flip = 0;
        move_list[tt_moves_idx0].value = -INF;
    }
    // if (tt_moves_idx1 != -1 && move_list[tt_moves_idx1].flip.flip && v <= alpha) {
    //     search->move(&move_list[tt_moves_idx1].flip);
    //         g = -nega_alpha_ordering_nws_simple(search, -alpha - 1, depth - 1, false, move_list[tt_moves_idx1].n_legal, searching);
    //     search->undo(&move_list[tt_moves_idx1].flip);
    //     if (v < g) {
    //         v = g;
    //         best_move = move_list[tt_moves_idx1].flip.pos;
    //     }
    //     move_list[tt_moves_idx1].flip.flip = 0;
    //     move_list[tt_moves_idx1].value = -INF;
    // }
    if (v <= alpha) {
        move_list_evaluate_nws(search, move_list, canput, moves, depth, alpha, searching);
#if USE_MID_ETC && MID_ETC_DEPTH_NWS <= MID_SIMPLE_ORDERING_DEPTH
        for (int move_idx = 0; move_idx < canput - n_etc_done && *searching; ++move_idx) {
#else
        for (int move_idx = 0; move_idx < canput && *searching; ++move_idx) {
#endif
            swap_next_best_move(move_list, move_idx, canput);
#if USE_MID_ETC && MID_ETC_DEPTH_NWS <= MID_SIMPLE_ORDERING_DEPTH
            if (move_list[move_idx].flip.flip == 0) {
                break;
            }
#endif
            search->move(&move_list[move_idx].flip);
                g = -nega_alpha_ordering_nws_simple(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, searching);
            search->undo(&move_list[move_idx].flip);
            if (v < g) {
                v = g;
                best_move = move_list[move_idx].flip.pos;
                if (alpha < v) {
#if USE_KILLER_MOVE_MO && USE_KILLER_MOVE_NWS_MO
                    search->update_heuristics_on_cutoff(move_list[move_idx].flip.pos, depth);
#endif
                    break;
                }
            }
        }
    }
    if (*searching && global_searching) {
        transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, best_move);
    }
    return v;
}


inline bool is_searching(std::vector<bool*> &searchings) {
    return std::all_of(searchings.begin(), searchings.end(), 
                       [](bool* elem) { return *elem; });
}



/*
    @brief Get a value with given depth with Nega-Alpha algorithm (NWS)

    Search with move ordering for midgame NWS
    Parallel search (YBWC: Young Brothers Wait Concept) used.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param depth                remaining depth
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param searching            flag for terminating this search
    @return the value
*/
int nega_alpha_ordering_nws(Search *search, int alpha, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, std::vector<bool*> &searchings) {
    if (!global_searching || !is_searching(searchings)) {
        return SCORE_UNDEFINED;
    }
    if (is_end_search) {
        if (depth <= MID_TO_END_DEPTH_MPC || (search->mpc_level == MPC_100_LEVEL && depth <= MID_TO_END_DEPTH)) {
            return nega_alpha_end_nws(search, alpha, skipped, legal);
        }
    } else {
        if (depth <= MID_SIMPLE_ORDERING_DEPTH) {
            return nega_alpha_ordering_nws_simple(search, alpha, depth, skipped, legal, searchings.back());
        }
    }
    int v = -SCORE_INF;
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[search->n_discs];
#endif
    if (legal == LEGAL_UNDEFINED) {
        legal = search->board.get_legal();
    }
    if (legal == 0ULL) {
        if (skipped) {
            return end_evaluate(&search->board);
        }
        search->pass();
            v = -nega_alpha_ordering_nws(search, -alpha - 1, depth, true, LEGAL_UNDEFINED, is_end_search, searchings);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    transposition_table.prefetch(hash_code);
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    if (transposition_cutoff_nws(search, hash_code, depth, alpha, &v, moves)) {
        return v;
    }
#if USE_MID_MPC
    if (search->mpc_level < MPC_100_LEVEL && depth >= USE_MPC_MIN_DEPTH) {
        if (mpc(search, alpha, alpha + 1, depth, legal, is_end_search, &v, searchings)) {
            return v;
        }
    }
#endif
    int best_move = MOVE_UNDEFINED;
    int g;
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
    if (depth >= MID_ETC_DEPTH_NWS) {
        if (etc_nws(search, move_list, canput, depth, alpha, &v, &n_etc_done)) {
            return v;
        }
    }
#endif
    bool serial_searched = false;
    if (tt_moves_idx0 != -1 && move_list[tt_moves_idx0].flip.flip) {
        search->move(&move_list[tt_moves_idx0].flip);
            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[tt_moves_idx0].n_legal, is_end_search, searchings);
        search->undo(&move_list[tt_moves_idx0].flip);
        if (v < g) {
            v = g;
            best_move = move_list[tt_moves_idx0].flip.pos;
        }
        serial_searched = true;
        move_list[tt_moves_idx0].flip.flip = 0;
        move_list[tt_moves_idx0].value = -INF;
    }
    if (v <= alpha) {
        move_list_evaluate_nws(search, move_list, canput, moves, depth, alpha, searchings.back());
#if USE_YBWC_NWS
        if (
            search->use_multi_thread && 
            ((!is_end_search && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth - 1 >= YBWC_END_SPLIT_MIN_DEPTH)) //&& 
            //((!is_end_search && depth - 1 <= YBWC_MID_SPLIT_MAX_DEPTH) || (is_end_search && depth - 1 <= YBWC_END_SPLIT_MAX_DEPTH))
        ) {
            move_list_sort(move_list, canput);
            //swap_next_best_move(move_list, 0, canput);
            if (move_list[0].flip.flip) {
                if (!serial_searched) {
                    search->move(&move_list[0].flip);
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[0].n_legal, is_end_search, searchings);
                    search->undo(&move_list[0].flip);
                    move_list[0].flip.flip = 0;
                    if (v < g) {
                        v = g;
                        best_move = move_list[0].flip.pos;
                    }
                }
                if (v <= alpha) {
                    ybwc_search_young_brothers_nws(search, alpha, &v, &best_move, canput - n_etc_done - 1, hash_code, depth, is_end_search, move_list, canput, searchings);
                }
            }
        } else{
#endif
            for (int move_idx = 0; move_idx < canput - n_etc_done && is_searching(searchings); ++move_idx) {
                swap_next_best_move(move_list, move_idx, canput);
#if USE_MID_ETC
                if (move_list[move_idx].flip.flip == 0) {
                    break;
                }
#endif
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                search->undo(&move_list[move_idx].flip);
                if (v < g) {
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v) {
#if USE_KILLER_MOVE_MO && USE_KILLER_MOVE_NWS_MO
                        search->update_heuristics_on_cutoff(move_list[move_idx].flip.pos, depth);
#endif
                        break;
                    }
                }
            }
#if USE_YBWC_NWS
        }
#endif
    }
    if (global_searching && is_searching(searchings)) {
        transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, best_move);
    }
    return v;
}

inline int nega_alpha_ordering_nws(Search *search, int alpha, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, bool *searching) {
    std::vector<bool*> searchings = {searching};
    return nega_alpha_ordering_nws(search, alpha, depth, skipped, legal, is_end_search, searchings);
}