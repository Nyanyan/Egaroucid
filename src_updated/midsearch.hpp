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
#include "level.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha(Search *search, int alpha, int beta);
inline bool mpc_higher(Search *search, int beta);
inline bool mpc_lower(Search *search, int alpha);

int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta){
    if (!global_searching)
        return -SCORE_UNDEFINED;
    if (search->depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta);
    ++search->n_nodes;
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_MPC
        if (MID_MPC_MIN_DEPTH <= search->depth && search->depth <= MID_MPC_MAX_DEPTH){
            if (mpc_higher(search, beta))
                return beta;
            if (mpc_lower(search, alpha))
                return alpha;
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_ordering_nomemo(search, -beta, -alpha);
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
            g = -nega_alpha_ordering_nomemo(search, -beta, -alpha);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher(Search *search, int beta){
    int bound = beta + ceil(search->mpct * mpcsd[search->board.phase()][search->depth - MID_MPC_MIN_DEPTH]);
    if (bound > HW2)
        bound = HW2; //return false;
    return nega_alpha_ordering_nomemo(search, bound - 1, bound) >= bound;
}

inline bool mpc_lower(Search *search, int alpha){
    int bound = alpha - ceil(search->mpct * mpcsd[search->board.phase()][search->depth - MID_MPC_MIN_DEPTH]);
    if (bound < -HW2)
        bound = -HW2; //return false;
    return nega_alpha_ordering_nomemo(search, bound, bound + 1) <= bound;
}

int nega_alpha(Search *search, int alpha, int beta){
    if (!global_searching)
        return -SCORE_UNDEFINED;
    ++search->n_nodes;
    if (search->depth == 0)
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
            v = -nega_alpha(search, -beta, -alpha);
        search->undo_pass();
        alpha = max(alpha, v);
        return v;
    }
    Mobility mob;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &search->board, cell);
            search->board.move(&mob);
                g = -nega_alpha(search, -beta, -alpha);
            search->board.undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

int nega_alpha_ordering(Search *search, int alpha, int beta){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (search->depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta);
    ++search->n_nodes;
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_TC
        int l, u, hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
        search->parent_transpose_table->get_now(&search->board, hash_code, &l, &u);
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
        if (MID_MPC_MIN_DEPTH <= search->depth && search->depth <= MID_MPC_MAX_DEPTH){
            if (mpc_higher(search, beta)){
                #if USE_MID_MPC
                    if (l < beta)
                        search->parent_transpose_table->reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_lower(search, alpha)){
                #if USE_MID_MPC
                    if (alpha < u)
                        search->parent_transpose_table->reg(&search->board, hash_code, l, alpha);
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
            v = -nega_alpha_ordering(search, -beta, -alpha);
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
    #else
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                g = -nega_alpha(search, -beta, -alpha);
            search->board.undo(&mob);
            search->child_transpose_table->reg(&search->board, hash_code, mob.pos, g);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_MID_TC
                    if (l < alpha)
                        search->parent_transpose_table->reg(&search->board, hash_code, alpha, u);
                #endif
                return alpha;
            }
            v = max(v, g);
        }
    #endif
    /*
    #if USE_MULTI_THREAD
        int i;
        const int first_threshold = canput / mid_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -nega_alpha_ordering(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        vector<future<int>> future_tasks;
        unsigned long long *n_n_nodes = new unsigned long long[canput - first_threshold];
        int done_tasks = first_threshold;
        for (i = first_threshold; i < canput; ++i)
            n_n_nodes[i - first_threshold] = 0;
        int next_done_tasks, additional_done_tasks;
        while (done_tasks < canput){
            next_done_tasks = canput;
            future_tasks.clear();
            for (i = done_tasks; i < canput; ++i){
                if (thread_pool.n_idle() == 0){
                    next_done_tasks = i;
                    break;
                }
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, &n_n_nodes[i - first_threshold], vacant_lst)));
            }
            additional_done_tasks = 0;
            if (next_done_tasks < canput){
                g = -nega_alpha_ordering(&nb[next_done_tasks], false, depth - 1, -beta, -alpha,  use_mpc, mpct_in, n_nodes, vacant_lst);
                alpha = max(alpha, g);
                v = max(v, g);
                additional_done_tasks = 1;
            }
            for (i = done_tasks; i < next_done_tasks; ++i){
                g = -future_tasks[i - done_tasks].get();
                alpha = max(alpha, g);
                v = max(v, g);
                *n_nodes += n_n_nodes[i - first_threshold];
            }
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                delete[] n_n_nodes;
                return alpha;
            }
            done_tasks = next_done_tasks + additional_done_tasks;
        }
        delete[] nb;
        delete[] n_n_nodes;
    #else
        for (idx = 0; idx < canput; ++idx){
            g = -nega_alpha_ordering(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        delete[] nb;
    #endif
    */
    if (v <= alpha)
        search->parent_transpose_table->reg(&search->board, hash_code, l, v);
    else
        search->parent_transpose_table->reg(&search->board, hash_code, v, v);
    return v;
}

int nega_scout(Search *search, int alpha, int beta){
    if (!global_searching)
        return -INF;
    if (search->depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta);
    ++search->n_nodes;
    #if USE_MID_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_MID_TC
        int l, u, hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
        search->parent_transpose_table->get_now(&search->board, hash_code, &l, &u);
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
        if (MID_MPC_MIN_DEPTH <= search->depth && search->depth <= MID_MPC_MAX_DEPTH){
            if (mpc_higher(search, beta)){
                #if USE_MID_MPC
                    if (l < beta)
                        search->parent_transpose_table->reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_lower(search, alpha)){
                #if USE_MID_MPC
                    if (alpha < u)
                        search->parent_transpose_table->reg(&search->board, hash_code, l, alpha);
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
            v = -nega_scout(search, -beta, -alpha);
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
        int first_alpha = alpha;
    #else
        search->board.move(&move_list[0]);
        g = -nega_scout(search, -beta, -alpha);
        search->board.undo(&move_list[0]);
        search->child_transpose_table->reg(&search->board, hash_code, move_list[0].pos, g);
        alpha = max(alpha, g);
        if (beta <= alpha){
            #if USE_MID_TC
                if (l < alpha)
                    search->parent_transpose_table->reg(&search->board, hash_code, alpha, u);
            #endif
            return alpha;
        }
        v = max(v, g);
        for (int i = 1; i < canput; ++i){
            search->board.move(&move_list[i]);
                g = -nega_alpha_ordering(search, -alpha - 1, -alpha);
                if (alpha < g)
                    g = -nega_scout(search, -beta, -g);
            search->board.undo(&move_list[i]);
            search->child_transpose_table->reg(&search->board, hash_code, move_list[i].pos, g);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_MID_TC
                    if (l < alpha)
                        search->parent_transpose_table->reg(&search->board, hash_code, alpha, u);
                #endif
                return alpha;
            }
            v = max(v, g);
        }
    #endif
    /*
    #if USE_MULTI_THREAD
        int i;
        const int first_threshold = canput / mid_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -nega_scout(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        vector<future<int>> future_tasks;
        unsigned long long *n_n_nodes = new unsigned long long[canput - first_threshold];
        bool *re_search = new bool[canput - first_threshold];
        int done_tasks = first_threshold;
        for (i = first_threshold; i < canput; ++i){
            n_n_nodes[i - first_threshold] = 0;
            re_search[i - first_threshold] = false;
        }
        int next_done_tasks, additional_done_tasks, before_alpha;
        while (done_tasks < canput){
            before_alpha = alpha;
            next_done_tasks = canput;
            future_tasks.clear();
            for (i = done_tasks; i < canput; ++i){
                if (thread_pool.n_idle() == 0){
                    next_done_tasks = i;
                    break;
                }
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in, &n_n_nodes[i - first_threshold], vacant_lst)));
            }
            additional_done_tasks = 0;
            if (next_done_tasks < canput){
                g = -nega_alpha_ordering(&nb[next_done_tasks], false, depth - 1, -alpha - search_epsilon, -alpha,  use_mpc, mpct_in, n_nodes, vacant_lst);
                if (before_alpha < g)
                    re_search[next_done_tasks - first_threshold] = true;
                alpha = max(alpha, g);
                v = max(v, g);
                additional_done_tasks = 1;
            }
            for (i = done_tasks; i < next_done_tasks; ++i){
                g = -future_tasks[i - done_tasks].get();
                if (before_alpha < g)
                    re_search[i - first_threshold] = true;
                alpha = max(alpha, g);
                v = max(v, g);
                *n_nodes += n_n_nodes[i - first_threshold];
            }
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                delete[] n_n_nodes;
                delete[] re_search;
                return alpha;
            }
            for (i = done_tasks; i < next_done_tasks + additional_done_tasks; ++i){
                if (re_search[i - first_threshold]){
                    g = -nega_scout(&nb[i], false, depth - 1, -beta, -alpha,  use_mpc, mpct_in, n_nodes, vacant_lst);
                    alpha = max(alpha, g);
                    v = max(v, g);
                }
            }
            done_tasks = next_done_tasks + additional_done_tasks;
        }
        delete[] nb;
        delete[] n_n_nodes;
        delete[] re_search;
    #else
        for (idx = 0; idx < canput; ++idx){
            g = -nega_alpha_ordering(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        delete[] nb;
    #endif
    */
    if (v <= alpha)
        search->parent_transpose_table->reg(&search->board, hash_code, l, v);
    else
        search->parent_transpose_table->reg(&search->board, hash_code, v, v);
    return v;
}


int mtd(Search *search, int l, int u){
    int g, beta;
    g = nega_alpha(search, l, u);
    while (u > l){
        beta = max(l + 1, g);
        g = nega_alpha_ordering(search, beta - 1, beta);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
}

inline Search_result midsearch(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst, Parent_transpose_table *parent_transpose_table, Child_transpose_table *child_transpose_table){
    long long strt = tim();
    int hash_code = b.hash() & TRANSPOSE_TABLE_MASK;
    Search search;
    search.board = b;
    search.parent_transpose_table = parent_transpose_table;
    search.child_transpose_table = child_transpose_table;
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
    if (move_list.size() >= 2)
        move_ordering(&search, move_list);
    Search_result res;
    int alpha, beta, g, former_alpha = -INF;
    for (search.depth = min(16, max(0, max_depth - 5)); search.depth <= max_depth - 1; ++search.depth){
        alpha = -HW2;
        beta = HW2;
        search.parent_transpose_table->ready_next_search();
        search.child_transpose_table->ready_next_search();
        for (const Mobility &mob: move_list){
            search.board.move(&mob);
                g = -mtd(&search, alpha, beta);
            search.board.undo(&mob);
            search.child_transpose_table->reg(&search.board, hash_code, mob.pos, g);
            if (alpha < g){
                alpha = g;
                res.policy = mob.pos;
            }
        }
        if (search.depth == max_depth - 2)
            former_alpha = alpha;
    }
    res.depth = max_depth;
    res.nps = search.n_nodes * 1000 / max(1LL, tim() - strt);
    if (former_alpha != -INF)
        res.value = (former_alpha + alpha) / 2;
    else
        res.value = alpha;
    return res;
}

inline Search_result midsearch_value(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst, Parent_transpose_table *parent_transpose_table, Child_transpose_table *child_transpose_table){
    long long strt = tim();
    Search search;
    search.board = b;
    search.parent_transpose_table = parent_transpose_table;
    search.child_transpose_table = child_transpose_table;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    int g, former_g = -INF;
    for (search.depth = min(16, max(0, max_depth - 5)); search.depth <= max_depth - 1; ++search.depth){
        search.parent_transpose_table->ready_next_search();
        search.child_transpose_table->ready_next_search();
        g = -mtd(&search, -HW2, HW2);
        if (search.depth == max_depth - 2)
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
