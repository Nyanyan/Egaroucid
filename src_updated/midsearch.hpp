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
    search->skipped = false;
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

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool is_end_search){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end(search, alpha, beta);
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth);
    ++(search->n_nodes);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    #if USE_MID_TC
        int l, u;
        parent_transpose_table.get(&search->board, hash_code, &l, &u);
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
        if (search->use_mpc){
            if (!is_end_search && MID_MPC_MIN_DEPTH <= depth && depth <= MID_MPC_MAX_DEPTH){
                if (mpc_higher(search, beta, depth)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_lower(search, alpha, depth)){
                    #if USE_MID_TC && false
                        if (alpha < u)
                            parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                    #endif
                    return alpha;
                }
            } else if (is_end_search && END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH){
                int val = mid_evaluate(&search->board);
                if (mpc_end_higher(search, beta, depth, val)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_end_lower(search, alpha, depth, val)){
                    #if USE_MID_TC && false
                        if (alpha < u)
                            parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                    #endif
                    return alpha;
                }
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_ordering(search, -beta, -alpha, depth, is_end_search);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering(search, move_list);
    #if USE_MULTI_THREAD
        const int canput = pop_count_ull(legal);
        int pv_idx = 0, split_count = 0;
        vector<future<pair<int, unsigned long long>>> parallel_tasks;
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                if (ybwc_split(search, -beta, -alpha, depth - 1, is_end_search, mob.pos, pv_idx, canput, split_count, parallel_tasks)){
                    search->board.undo(&mob);
                    ++split_count;
                } else{
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, is_end_search);
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
                g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, is_end_search);
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

int nega_scout(Search *search, int alpha, int beta, int depth, bool is_end_search){
    if (!global_searching)
        return -INF;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_scout_end(search, alpha, beta);
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth);
    ++(search->n_nodes);
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    #if USE_MID_TC
        int l, u;
        parent_transpose_table.get(&search->board, hash_code, &l, &u);
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
        if (search->use_mpc){
            if (!is_end_search && MID_MPC_MIN_DEPTH <= depth && depth <= MID_MPC_MAX_DEPTH){
                if (mpc_higher(search, beta, depth)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_lower(search, alpha, depth)){
                    #if USE_MID_TC && false
                        if (alpha < u)
                            parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                    #endif
                    return alpha;
                }
            } else if (is_end_search && END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH){
                int val = mid_evaluate(&search->board);
                if (mpc_end_higher(search, beta, depth, val)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_end_lower(search, alpha, depth, val)){
                    #if USE_MID_TC && false
                        if (alpha < u)
                            parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                    #endif
                    return alpha;
                }
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_scout(search, -beta, -alpha, depth, is_end_search);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering(search, move_list);
    for (const Mobility &mob: move_list){
        search->board.move(&mob);
            if (v == -INF)
                g = -nega_scout(search, -beta, -alpha, depth - 1, is_end_search);
            else{
                g = -nega_alpha_ordering(search, -alpha - 1, -alpha, depth - 1, is_end_search);
                if (alpha < g)
                    g = -nega_scout(search, -beta, -g, depth - 1, is_end_search);
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


int mtd(Search *search, int l, int u, int g, int depth, bool is_end_search){
    int beta;
    g = max(l, min(u, g));
    while (u > l){
        beta = max(l + 1, g);
        g = nega_alpha_ordering(search, beta - 1, beta, depth, is_end_search);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
}

int mtd_end(Search *search, int l, int u, int g, int depth, bool is_end_search){
    int beta;
    l /= 2;
    u /= 2;
    g = max(l, min(u, g / 2));
    while (u > l){
        beta = max(l + 1, g);
        g = nega_alpha_ordering(search, beta * 2 - 2, beta * 2, depth, is_end_search) / 2;
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g * 2;
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
    unsigned long long f_n_nodes = 0;
    unsigned long long legal = b.mobility_ull();
    vector<Mobility> move_list;
    for (const int &cell: search.vacant_list){
        if (1 & (legal >> cell)){
            move_list.emplace_back(calc_flip(&search.board, cell));
            move_list[move_list.size() - 1].value = 0;
        }
    }
    Search_result res;
    int alpha, beta, g, former_alpha = -INF;
    if (b.n + max_depth < HW2){
        for (int depth = min(16, max(0, max_depth - 5)); depth <= max_depth; ++depth){
            alpha = -HW2;
            beta = HW2;
            parent_transpose_table.init();
            child_transpose_table.ready_next_search();
            move_ordering(&search, move_list);
            //move_ordering_value(move_list);
            for (Mobility &mob: move_list){
                search.board.move(&mob);
                    g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, false);
                search.board.undo(&mob);
                mob.value = g;
                if (alpha < g){
                    child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                    alpha = g;
                    res.policy = mob.pos;
                }
            }
            if (depth == max_depth - 2)
                former_alpha = alpha;
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
    } else{
        if (b.n >= 60 && false){

        } else{
            int depth = HW2 - b.n;
            vector<double> pre_search_mpcts;
            pre_search_mpcts.emplace_back(0.5);
            if (search.mpct > 1.6 || !search.use_mpc)
                pre_search_mpcts.emplace_back(1.0);
            if (search.mpct > 2.0 || !search.use_mpc)
                pre_search_mpcts.emplace_back(1.5);
            //if (!search.use_mpc)
            //    pre_search_mpcts.emplace_back(2.0);
            search.use_mpc = true;
            for (double pre_search_mpct: pre_search_mpcts){
                f_n_nodes = search.n_nodes;
                alpha = -HW2;
                beta = HW2;
                search.mpct = pre_search_mpct;
                parent_transpose_table.init();
                child_transpose_table.ready_next_search();
                move_ordering_value(move_list);
                for (Mobility &mob: move_list){
                    search.board.move(&mob);
                        g = -nega_scout(&search, -beta, min(HW2, -alpha + 6), depth - 1, true);
                        //g = -mtd(&search, -beta, min(HW2, -alpha + 6), -mob.value, depth - 1, true);
                    search.board.undo(&mob);
                    mob.value = g;
                    if (alpha < g){
                        child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                        alpha = g;
                        res.policy = mob.pos;
                    }
                }
                cerr << "endsearch time " << tim() - strt << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes - f_n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
            }
            alpha = -HW2;
            beta = HW2;
            search.use_mpc = use_mpc;
            search.mpct = mpct;
            parent_transpose_table.init();
            child_transpose_table.ready_next_search();
            move_ordering_value(move_list);
            //bool first_search = true;
            for (Mobility &mob: move_list){
                f_n_nodes = search.n_nodes;
                search.board.move(&mob);
                    /*
                    if (first_search)
                        g = -nega_scout(&search, -beta, -alpha, depth - 1, true);
                    else{
                        g = -nega_alpha_ordering(&search, -alpha - 1, -alpha, depth - 1, true);
                        if (alpha < g)
                            g = -nega_scout(&search, -beta, -g, depth - 1, true);
                    }
                    */
                    g = -mtd_end(&search, -beta, -alpha, -mob.value, depth - 1, true);
                search.board.undo(&mob);
                cerr << "policy " << mob.pos << " value " << g << " expected " << mob.value << " time " << tim() - strt << " nodes " << search.n_nodes - f_n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
                mob.value = g;
                if (alpha < g){
                    child_transpose_table.reg(&search.board, hash_code, mob.pos, g);
                    alpha = g;
                    res.policy = mob.pos;
                }
                //first_search = false;
            }
            cerr << "endsearch time " << tim() - strt << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
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
