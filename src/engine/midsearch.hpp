/*
    Egaroucid Project

    @file midsearch.hpp
        Search midgame
    @date 2021-2022
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
#include "null_move_pruning.hpp"
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
        ++search->n_nodes_mid_last;
    #endif
    int v = -INF;
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
            ++search->n_nodes_mid_last;
        #endif
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
            ++search->n_nodes_mid;
        #endif
        #if USE_END_SC
            if (is_end_search){
                int stab_res = stability_cut(search, &alpha, &beta);
                if (stab_res != SCORE_UNDEFINED)
                    return stab_res;
            }
        #endif
        int first_alpha = alpha;
        if (legal == LEGAL_UNDEFINED)
            legal = search->board.get_legal();
        int v = -INF;
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
        #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
            int l = -INF, u = INF;
            if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD){
                value_transposition_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
                if (u == l)
                    return u;
                if (beta <= l)
                    return l;
                if (u <= alpha)
                    return u;
                alpha = std::max(alpha, l);
                beta = std::min(beta, u);
            }
        #else
            int l, u;
            value_transposition_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
            if (u == l)
                return u;
            if (beta <= l)
                return l;
            if (u <= alpha)
                return u;
            alpha = std::max(alpha, l);
            beta = std::min(beta, u);
        #endif
        #if USE_MID_MPC
            if (search->use_mpc){
                if (search->n_discs <= USE_MPC_N_DISCS){
                    if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                        return v;
                }
                #if USE_MID_NMP
                    if (search->n_discs <= USE_NULL_MOVE_PRUNING_N_DISCS && depth <= USE_NULL_MOVE_PRUNING_DEPTH){
                        if (nmp(search, alpha, beta, depth, &v))
                            return v;
                    }
                #endif
            }
        #endif
        int best_move = best_move_transposition_table.get(&search->board, hash_code);
        if (best_move != TRANSPOSITION_TABLE_UNDEFINED){
            if (1 & (legal >> best_move)){
                Flip flip_best;
                calc_flip(&flip_best, &search->board, best_move);
                eval_move(search, &flip_best);
                search->move(&flip_best);
                    v = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
                search->undo(&flip_best);
                eval_undo(search, &flip_best);
                alpha = std::max(alpha, v);
                legal ^= 1ULL << best_move;
            } else
                best_move = TRANSPOSITION_TABLE_UNDEFINED;
        }
        int g;
        if (alpha < beta && legal){
            const int canput = pop_count_ull(legal);
            std::vector<Flip_value> move_list(canput);
            int idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
                calc_flip(&move_list[idx++].flip, &search->board, cell);
            move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
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
            register_tt(search, depth, hash_code, v, best_move, l, u, first_alpha, beta, searching);
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
    if (alpha + 1 == beta){
        bool mpc_used = false;
        return nega_alpha_ordering_nws(search, alpha, depth, skipped, legal, is_end_search, searching, &mpc_used);
    }
    #if USE_NEGA_ALPHA_END
        if (is_end_search && depth <= MID_TO_END_DEPTH)
            return nega_alpha_end(search, alpha, beta, skipped, legal, searching);
    #elif USE_NEGA_ALPHA_END_FAST
        if (is_end_search && search->n_discs >= HW2 - END_FAST_DEPTH)
            return nega_alpha_end_fast(search, alpha, beta, skipped, false, searching);
    #else
        if (is_end_search && search->n_discs == 60){
            uint64_t empties = ~(search->board.player | search->board.opponent);
            uint_fast8_t p0 = first_bit(&empties);
            uint_fast8_t p1 = next_bit(&empties);
            uint_fast8_t p2 = next_bit(&empties);
            uint_fast8_t p3 = next_bit(&empties);
            return last4(search, alpha, beta, p0, p1, p2, p3, false, searching);
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
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_mid;
    #endif
    #if USE_END_SC
        if (is_end_search){
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED)
                return stab_res;
        }
    #endif
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
    uint32_t hash_code = search->board.hash();
    #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
        int l = -INF, u = INF;
        if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD){
            value_transposition_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
            if (u == l)
                return u;
            if (beta <= l)
                return l;
            if (u <= alpha)
                return u;
            alpha = std::max(alpha, l);
            beta = std::min(beta, u);
        }
    #else
        int l, u;
        value_transposition_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
        if (u == l)
            return u;
        if (beta <= l)
            return l;
        if (u <= alpha)
            return u;
        alpha = std::max(alpha, l);
        beta = std::min(beta, u);
    #endif
    #if USE_MID_MPC
        if (search->use_mpc){
            if (search->n_discs <= USE_MPC_N_DISCS){
                if (mpc(search, alpha, beta, depth, legal, is_end_search, &v, searching))
                    return v;
            }
            #if USE_MID_NMP
                if (search->n_discs <= USE_NULL_MOVE_PRUNING_N_DISCS && depth <= USE_NULL_MOVE_PRUNING_DEPTH){
                    if (nmp(search, alpha, beta, depth, &v))
                        return v;
                }
            #endif
        }
    #endif
    int best_move = best_move_transposition_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSITION_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                v = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            alpha = std::max(alpha, v);
            legal ^= 1ULL << best_move;
        } else
            best_move = TRANSPOSITION_TABLE_UNDEFINED;
    }
    int g;
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        std::vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
        bool mpc_used;
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, &move_list[move_idx].flip);
            search->move(&move_list[move_idx].flip);
                if (v == -INF)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                else{
                    mpc_used = false;
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching, &mpc_used);
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

/*
    @brief Wrapper of nega_scout

    This function is used in root node

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                remaining depth
    @param skipped              already passed?
    @param is_end_search        search till the end?
    @param is_main_search       is this main search? (used for logging)
    @param best_move            previously calculated best move
    @param clogs                previously found clog moves
    @return pair of value and best move
*/
std::pair<int, int> first_nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, bool is_end_search, const bool is_main_search, int best_move, const std::vector<Clog_result> clogs){
    bool searching = true;
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_mid;
    #endif
    uint32_t hash_code = search->board.hash();
    uint64_t legal = search->board.get_legal();
    int first_alpha = alpha;
    int g, v = -INF;
    if (legal == 0ULL)
        return std::make_pair(SCORE_UNDEFINED, -1);
    int best_move_res = -1;
    const int canput_all = pop_count_ull(legal);
    for (const Clog_result clog: clogs){
        if (v < clog.val){
            v = clog.val;
            best_move_res = clog.pos;
        }
        legal ^= 1ULL << clog.pos;
    }
    alpha = std::max(alpha, v);
    bool pre_best_move_found = false;
    if (best_move != TRANSPOSITION_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                g = -nega_scout(search, -beta, -alpha, depth - 1, false, LEGAL_UNDEFINED, is_end_search, &searching);
                if (is_main_search)
                    std::cerr << 1 << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord(best_move) << " value " << g << std::endl;
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            if (v < g){
                v = g;
                best_move_res = best_move;
                if (alpha < v)
                    alpha = v;
            }
            legal ^= 1ULL << best_move;
            pre_best_move_found = true;
        }
    }
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        int mobility_idx = pre_best_move_found ? 2 : 1;
        std::vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, &searching);
        bool is_first_search = true;
        bool mpc_used;
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            swap_next_best_move(move_list, move_idx, canput);
            eval_move(search, &move_list[move_idx].flip);
            search->move(&move_list[move_idx].flip);
                if (!pre_best_move_found && is_first_search)
                    g = -nega_scout(search, -beta, -alpha, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                else{
                    mpc_used = false;
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching, &mpc_used);
                    if (alpha <= g && g < beta)
                        g = -nega_scout(search, -beta, -g, depth - 1, false, move_list[move_idx].n_legal, is_end_search, &searching);
                }
                if (is_main_search){
                    if (g <= alpha)
                        std::cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)move_list[move_idx].flip.pos) << " value " << g << " or lower" << std::endl;
                    else
                        std::cerr << mobility_idx << "/" << canput_all << " [" << alpha << "," << beta << "] mpct " << search->mpct << " " << idx_to_coord((int)move_list[move_idx].flip.pos) << " value " << g << std::endl;
                }
                ++mobility_idx;
            search->undo(&move_list[move_idx].flip);
            eval_undo(search, &move_list[move_idx].flip);
            if (v < g){
                v = g;
                best_move_res = move_list[move_idx].flip.pos;
                if (alpha < v){
                    if (beta <= v)
                        break;
                    alpha = v;
                }
            }
            is_first_search = false;
        }
    }
    register_tt(search, depth, hash_code, v, best_move_res, first_alpha, beta, first_alpha, beta, &searching);
    return std::make_pair(v, best_move_res);
}
