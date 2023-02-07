/*
    Egaroucid Project

    @file midsearch.hpp
        Search midgame
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
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
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_alpha_eval1(search, -beta, -alpha, true);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    int g;
    uint64_t flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        flip = calc_flip(&search->board, cell);
        eval_move(search, flip, cell);
        search->move_cell(flip, cell);
            g = -mid_evaluate_diff(search);
        search->undo_cell(flip, cell);
        eval_undo(search, flip, cell);
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

#if MID_FAST_DEPTH > 1 // CANNOT BE COMPILED
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
            return nega_alpha_eval1(search, alpha, beta, skipped);
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

#if USE_NEGA_ALPHA_ORDERING // CANNOT BE COMPILED
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
                    return nega_alpha_eval1(search, alpha, beta, skipped);
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
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
                calc_flip(&move_list[idx++].flip, &search->board, cell);
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
        if (is_end_search && search->n_discs == 60)
            return last4_wrapper(search, alpha, beta);
    #endif
    if (!is_end_search){
        #if MID_FAST_DEPTH > 1
            if (depth <= MID_FAST_DEPTH)
                return nega_alpha(search, alpha, beta, depth, skipped, searching);
        #else
            if (depth == 1)
                return nega_alpha_eval1(search, alpha, beta, skipped);
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
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    int first_alpha = alpha;
    int g;
    const int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    Square *square;
    foreach_square(square, search->empty_list){
        if (1 & (legal >> square->cell)){
            move_list[idx].flip = calc_flip(&search->board, square->cell);
            move_list[idx++].square = square;
        }
    }
    #if USE_MID_ETC
        if (search->n_discs - search->strt_n_discs < MID_ETC_DEPTH){
            if (etc(search, move_list, depth, &alpha, &beta, &v))
                return v;
        }
    #endif
    move_list_evaluate(search, move_list, depth, alpha, beta, moves, searching);
    for (int move_idx = 0; move_idx < canput; ++move_idx){
        swap_next_best_move(move_list, move_idx, canput);
        eval_move(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
        search->move(move_list[move_idx].flip, move_list[move_idx].square);
            if (v == -SCORE_INF)
                g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
            else{
                g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                if (alpha < g && g < beta)
                    g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
            }
        search->undo(move_list[move_idx].flip, move_list[move_idx].square);
        eval_undo(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
        if (v < g){
            v = g;
            best_move = move_list[move_idx].square->cell;
            if (alpha < v){
                if (beta <= v)
                    break;
                alpha = v;
            }
        }
    }
    if (*searching && global_searching)
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    return v;
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
    @param legal                legal moves in bitboard
    @return pair of value and best move
*/
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs, uint64_t legal){
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
        const int canput = pop_count_ull(legal);
        std::vector<Flip_value> move_list(canput);
        int idx = 0;
        Square *square;
        foreach_square(square, search->empty_list){
            if (1 & (legal >> square->cell)){
                move_list[idx].flip = calc_flip(&search->board, square->cell);
                move_list[idx++].square = square;
            }
        }
        move_list_evaluate(search, move_list, depth, alpha, beta, moves, &searching);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
            search->move(move_list[move_idx].flip, move_list[move_idx].square);
                if (v == -SCORE_INF)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                else{
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    if (alpha <= g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                }
            search->undo(move_list[move_idx].flip, move_list[move_idx].square);
            eval_undo(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
            if (is_main_search){
                if (g <= alpha)
                    std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].square->cell) << " value " << g << " or lower" << std::endl;
                else
                    std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].square->cell) << " value " << g << std::endl;
            }
            if (v < g){
                v = g;
                best_move = move_list[move_idx].square->cell;
                if (alpha < v){
                    if (beta <= v)
                        break;
                    alpha = v;
                }
            }
            ++pv_idx;
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
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int depth, bool is_end_search, const bool is_main_search, const std::vector<Clog_result> clogs){
    return first_nega_scout(search, alpha, beta, depth, is_end_search, is_main_search, clogs, search->board.get_legal());
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
        const int canput = pop_count_ull(legal);
        std::vector<Flip_value> move_list(canput);
        int idx = 0;
        Square *square;
        foreach_square(square, search->empty_list){
            if (1 & (legal >> square->cell)){
                move_list[idx].flip = calc_flip(&search->board, square->cell);
                move_list[idx++].square = square;
            }
        }
        move_list_evaluate(search, move_list, depth, alpha, beta, moves, &searching);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
            search->move(move_list[move_idx].flip, move_list[move_idx].square);
                if (v == -SCORE_INF)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                else{
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                    if (alpha <= g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                }
            search->undo(move_list[move_idx].flip, move_list[move_idx].square);
            eval_undo(search, move_list[move_idx].flip, move_list[move_idx].square->cell);
            if (is_main_search){
                if (g <= alpha)
                    std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].square->cell) << " value " << g << " or lower" << std::endl;
                else
                    std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search->mpc_level] << "% " << pv_idx << "/" << canput_all << " [" << alpha << "," << beta << "] " << idx_to_coord(move_list[move_idx].square->cell) << " value " << g << std::endl;
            }
            if (v < g){
                v = g;
                best_move = move_list[move_idx].square->cell;
                if (alpha < v){
                    if (beta <= v)
                        break;
                    alpha = v;
                }
            }
            ++pv_idx;
        }
    }
    if (global_searching)
        transposition_table.reg(search, hash_code, depth, first_alpha, beta, v, best_move);
    return v;
}