#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "endsearch.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif

using namespace std;

int nega_alpha(Search *search, int alpha, int beta, int depth);
inline bool mpc_higher(Search *search, int beta, int depth);
inline bool mpc_lower(Search *search, int alpha, int depth);

int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth){
    if (!global_searching)
        return -SCORE_UNDEFINED;
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth);
    ++(search->n_nodes);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_MPC
        if (MID_MPC_MIN_DEPTH <= depth && depth <= MID_MPC_MAX_DEPTH && search->use_mpc){
            if (mpc_higher(search, beta, depth))
                return beta;
            if (mpc_lower(search, alpha, depth))
                return alpha;
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_ordering_nomemo(search, -beta, -alpha, depth);
        search->undo_pass();
        alpha = max(alpha, v);
        return v;
    }
    const int canput = pop_count_ull(legal);
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering(search, move_list);
    for (const Mobility &mob: move_list){
        search->board.move(&mob);
            g = -nega_alpha_ordering_nomemo(search, -beta, -alpha, depth);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher(Search *search, int beta, int depth){
    int bound = beta + ceil(search->mpct * mpcsd[search->board.phase()][depth - MID_MPC_MIN_DEPTH]);
    if (bound > HW2)
        bound = HW2; //return false;
    return nega_alpha_ordering_nomemo(search, bound - 1, bound, mpcd[depth]) >= bound;
}

inline bool mpc_lower(Search *search, int alpha, int depth){
    int bound = alpha - ceil(search->mpct * mpcsd[search->board.phase()][depth - MID_MPC_MIN_DEPTH]);
    if (bound < -HW2)
        bound = -HW2; //return false;
    return nega_alpha_ordering_nomemo(search, bound, bound + 1, mpcd[depth]) <= bound;
}

int nega_alpha(Search *search, int alpha, int beta, int depth){
    if (!global_searching)
        return SCORE_UNDEFINED;
    ++(search->n_nodes);
    if (depth == 0)
        return mid_evaluate(&search->board);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    int g, v = -INF;
    unsigned long long legal = search->board.mobility_ull();
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha(search, -beta, -alpha, depth);
        search->undo_pass();
        alpha = max(alpha, v);
        return v;
    }
    Mobility mob;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &search->board, cell);
            search->board.move(&mob);
                g = -nega_alpha(search, -beta, -alpha, depth - 1);
            search->board.undo(&mob);
            alpha = max(alpha, g);
            v = max(v, g);
            if (beta <= alpha)
                break;
        }
    }
    return v;
}

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (depth >= HW2 - search->board.n && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end(search, alpha, beta);
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth);
    ++(search->n_nodes);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_TC
        int l, u, hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
        parent_transpose_table.get_now(&search->board, hash_code, &l, &u);
        if (u == l)
            return u;
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    #endif
    #if USE_MID_MPC
        if (MID_MPC_MIN_DEPTH <= depth && depth <= MID_MPC_MAX_DEPTH && search->use_mpc){
            if (mpc_higher(search, beta, depth)){
                #if USE_MID_TC
                    if (l < beta)
                        parent_transpose_table.reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_lower(search, alpha, depth)){
                #if USE_MID_TC
                    if (alpha < u)
                        parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                #endif
                return alpha;
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int first_alpha = alpha, g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_ordering(search, -beta, -alpha, depth);
        search->undo_pass();
        return v;
    }
    const int canput = pop_count_ull(legal);
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering(search, move_list);
    #if USE_MULTI_THREAD
        int pv_idx = 0, split_count = 0;
        vector<future<pair<int, unsigned long long>>> parallel_tasks;
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                if (ybwc_split(search, -beta, -alpha, depth - 1, mob.pos, pv_idx, canput, split_count, parallel_tasks)){
                    search->board.undo(&mob);
                    ++split_count;
                } else{
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1);
                    search->board.undo(&mob);
                    child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
                    alpha = max(alpha, g);
                    v = max(v, g);
                    if (beta <= alpha)
                        break;
                }
            ++pv_idx;
        }
        if (split_count){
            g = ybwc_wait(search, parallel_tasks);
            alpha = max(alpha, g);
            v = max(v, g);
        }
    #else
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1);
            search->board.undo(&mob);
            child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
            alpha = max(alpha, g);
            v = max(v, g);
            if (beta <= alpha)
                break;
        }
    #endif
    #if USE_MID_TC
        if (beta <= v)
            parent_transpose_table.reg(&search->board, hash_code, v, u);
        else if (v <= alpha)
            parent_transpose_table.reg(&search->board, hash_code, l, v);
        else
            parent_transpose_table.reg(&search->board, hash_code, v, v);
    #endif
    return v;
}

int nega_scout(Search *search, int alpha, int beta, int depth){
    if (!global_searching)
        return -INF;
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth);
    ++(search->n_nodes);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_TC
        int l, u, hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
        parent_transpose_table.get_now(&search->board, hash_code, &l, &u);
        if (u == l)
            return u;
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    #endif
    #if USE_MID_MPC
        if (MID_MPC_MIN_DEPTH <= depth && depth <= MID_MPC_MAX_DEPTH && search->use_mpc){
            if (mpc_higher(search, beta, depth)){
                #if USE_MID_TC
                    if (l < beta)
                        parent_transpose_table.reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_lower(search, alpha, depth)){
                #if USE_MID_TC
                    if (alpha < u)
                        parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                #endif
                return alpha;
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_scout(search, -beta, -alpha, depth);
        search->undo_pass();
        return v;
    }
    const int canput = pop_count_ull(legal);
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering(search, move_list);
    for (const Mobility &mob: move_list){
        search->board.move(&mob);
            if (v == -INF)
                g = -nega_scout(search, -beta, -alpha, depth - 1);
            else{
                g = -nega_alpha_ordering(search, -alpha - 1, -alpha, depth - 1);
                if (alpha < g)
                    g = -nega_scout(search, -beta, -g, depth - 1);
            }
        search->board.undo(&mob);
        child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
        alpha = max(alpha, g);
        v = max(v, g);
        if (beta <= alpha)
            break;
    }
    #if USE_MID_TC
        if (beta <= v)
            parent_transpose_table.reg(&search->board, hash_code, v, u);
        else if (v <= alpha)
            parent_transpose_table.reg(&search->board, hash_code, l, v);
        else
            parent_transpose_table.reg(&search->board, hash_code, v, v);
    #endif
    return v;
}


int mtd(Search *search, int l, int u, int depth){
    int g, beta;
    g = nega_alpha(search, l, u, 5);
    while (u > l){
        beta = max(l + 1, g);
        g = nega_alpha_ordering(search, beta - 1, beta, depth);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
}

inline Search_result tree_search(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst){
    long long strt = tim();
    int hash_code = b.hash() & TRANSPOSE_TABLE_MASK;
    Search search;
    search.board = b;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    unsigned long long legal = b.mobility_ull();
    vector<Mobility> move_list;
    for (const int &cell: search.vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search.board, cell));
    }
    Search_result res;
    int alpha, beta, g, former_alpha = -INF;
    if (b.n + max_depth < HW2){
        for (int depth = min(16, max(0, max_depth - 5)); depth <= max_depth; ++depth){
            alpha = -HW2;
            beta = HW2;
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            move_ordering(&search, move_list);
            for (const Mobility &mob: move_list){
                search.board.move(&mob);
                    g = -mtd(&search, -beta, -alpha, depth - 1);
                search.board.undo(&mob);
                child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                if (alpha < g){
                    alpha = g;
                    res.policy = mob.pos;
                }
            }
            if (depth == max_depth - 2)
                former_alpha = alpha;
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
    } else{
        if (b.n >= 60){

        } else{
            int depth = HW2 - b.n - 1;
            vector<double> pre_search_mpcts;
            pre_search_mpcts.emplace_back(0.5);
            if (search.mpct > 1.6 || !search.use_mpc)
                pre_search_mpcts.emplace_back(1.0);
            if (search.mpct > 2.0 || !search.use_mpc)
                pre_search_mpcts.emplace_back(1.5);
            search.use_mpc = true;
            for (double pre_search_mpct: pre_search_mpcts){
                alpha = -HW2;
                beta = HW2;
                search.mpct = pre_search_mpct;
                parent_transpose_table.ready_next_search();
                child_transpose_table.ready_next_search();
                move_ordering(&search, move_list);
                for (const Mobility &mob: move_list){
                    search.board.move(&mob);
                        g = -mtd(&search, -beta, -alpha, depth);
                    search.board.undo(&mob);
                    child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                    if (alpha < g){
                        alpha = g;
                        res.policy = mob.pos;
                    }
                }
                cerr << "endsearch time " << tim() - strt << " mpct " << pre_search_mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
            }
            alpha = -HW2;
            beta = HW2;
            search.use_mpc = use_mpc;
            search.mpct = mpct;
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            move_ordering(&search, move_list);
            for (const Mobility &mob: move_list){
                search.board.move(&mob);
                    g = -mtd(&search, -beta, -alpha, depth);
                search.board.undo(&mob);
                child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                if (alpha < g){
                    alpha = g;
                    res.policy = mob.pos;
                }
            }
            cerr << "endsearch time " << tim() - strt << " mpct " << use_mpc << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
    }
    res.depth = max_depth;
    res.nps = search.n_nodes * 1000 / max(1LL, tim() - strt);
    if (former_alpha != -INF)
        res.value = (former_alpha + alpha) / 2;
    else
        res.value = alpha;
    return res;
}

inline Search_result tree_search_value(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst){
    long long strt = tim();
    Search search;
    search.board = b;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    int g, former_g = -INF;
    for (int depth = min(16, max(0, max_depth - 5)); depth <= max_depth - 1; ++depth){
        parent_transpose_table.ready_next_search();
        child_transpose_table.ready_next_search();
        g = -mtd(&search, -HW2, HW2, depth);
        if (depth == max_depth - 2)
            former_g = g;
    }
    Search_result res;
    res.depth = max_depth;
    res.nps = search.n_nodes * 1000 / max(1LL, tim() - strt);
    if (former_g != -INF)
        res.value = (former_g + g) / 2;
    else
        res.value = g;
    return res;
}
