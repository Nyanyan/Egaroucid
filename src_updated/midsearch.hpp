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
#include "transpose_table.hpp"
#include "endsearch.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif
#if USE_LOG
    #include "log.hpp"
#endif

using namespace std;

#define USE_DEFAULT_MPC -1.0
#define PRESEARCH_OFFSET 6
#define PARALLEL_SPLIT_DIV 6

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
    move_ordering(search, move_list, depth);
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

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end(search, alpha, beta, searching);
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
                if (mpc_end_higher(search, beta, val)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_end_lower(search, alpha, val)){
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
            v = -nega_alpha_ordering(search, -beta, -alpha, depth, is_end_search, searching);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    const int canput = pop_count_ull(legal);
    vector<Mobility> move_list(canput);
    int idx = 0;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            calc_flip(&move_list[idx++], &search->board, cell);
    }
    move_ordering(search, move_list, depth);
    #if USE_MULTI_THREAD
        int best_moves[N_BEST_MOVES] = {TRANSPOSE_TABLE_UNDEFINED, TRANSPOSE_TABLE_UNDEFINED, TRANSPOSE_TABLE_UNDEFINED};
        int pv_idx = 0, split_count = 0;
        vector<future<pair<int, unsigned long long>>> parallel_tasks;
        bool n_searching = true;
        #if MULTI_THREAD_EARLY_GETTING_MODE == 1
            int parallel_value;
        #endif
        for (const Mobility &mob: move_list){
            if (!(*searching))
                break;
            #if MULTI_THREAD_EARLY_GETTING_MODE == 1
                if (split_count){
                    parallel_value = child_transpose_table.get_best_value(&search->board, hash_code);
                    if (parallel_value != TRANSPOSE_TABLE_UNDEFINED && parallel_value > alpha){
                        alpha = parallel_value;
                        v = parallel_value;
                        if (beta <= alpha)
                            break;
                    }
                }
            #endif
            if (ybwc_split(search, &mob, -beta, -alpha, depth - 1, is_end_search, &n_searching, mob.pos, pv_idx++, canput, split_count, parallel_tasks)){
                ++split_count;
            } else{
                search->board.move(&mob);
                    g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, is_end_search, searching);
                search->board.undo(&mob);
                if (*searching){
                    alpha = max(alpha, g);
                    if (v < g){
                        v = g;
                        update_best_move(best_moves, mob.pos);
                        //child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
                    }
                    #if MULTI_THREAD_EARLY_GETTING_MODE == 2
                        if (split_count && beta > alpha && *searching){
                            g = ybwc_wait(search, parallel_tasks);
                            alpha = max(alpha, g);
                            v = max(v, g);
                        }
                    #endif
                    if (beta <= alpha)
                        break;
                }
            }
        }
        child_transpose_table.reg(&search->board, hash_code, best_moves, g);
        if (split_count){
            if (beta <= alpha || !(*searching)){
                n_searching = false;
                ybwc_wait_strict(search, parallel_tasks);
            } else{
                g = ybwc_wait_strict(search, parallel_tasks);
                alpha = max(alpha, g);
                v = max(v, g);
            }
        }
    #else
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                g = -nega_alpha_ordering(search, -beta, -alpha, depth - 1, is_end_search, searching);
            search->board.undo(&mob);
            alpha = max(alpha, g);
            if (v < g){
                v = g;
                child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
            }
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
    //if (is_end_search && depth <= MID_TO_END_DEPTH)
    //    return nega_scout_end(search, alpha, beta);
    if (!is_end_search && depth <= MID_FAST_DEPTH)
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
                if (mpc_end_higher(search, beta, val)){
                    #if USE_MID_TC && false
                        if (l < beta)
                            parent_transpose_table.reg(&search->board, hash_code, beta, u);
                    #endif
                    return beta;
                }
                if (mpc_end_lower(search, alpha, val)){
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
    const int canput = pop_count_ull(legal);
    vector<Mobility> move_list(canput);
    int idx = 0;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            calc_flip(&move_list[idx++], &search->board, cell);
    }
    move_ordering(search, move_list, depth);
    bool searching = true;
    for (const Mobility &mob: move_list){
        search->board.move(&mob);
            if (v == -INF)
                g = -nega_scout(search, -beta, -alpha, depth - 1, is_end_search);
            else{
                g = -nega_alpha_ordering(search, -alpha - 1, -alpha, depth - 1, is_end_search, &searching);
                if (alpha < g)
                    g = -nega_scout(search, -beta, -g, depth - 1, is_end_search);
            }
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (v < g){
            v = g;
            child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
        }
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
    bool searching = true;
    g = max(l, min(u, g));
    while (u > l){
        beta = max(l + 1, g);
        g = nega_alpha_ordering(search, beta - 1, beta, depth, is_end_search, &searching);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
}

/*
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
*/

inline Search_result tree_search(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst){
    long long strt = tim();
    long long sum_time = 0;
    int hash_code = b.hash() & TRANSPOSE_TABLE_MASK;
    Search search;
    search.board = b;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    unsigned long long f_n_nodes, f_n_nodes2;
    long long strt2, strt3;
    unsigned long long legal = b.mobility_ull();
    vector<Mobility> move_list;
    for (const int &cell: search.vacant_list){
        if (1 & (legal >> cell)){
            move_list.emplace_back(calc_flip(&search.board, cell));
            move_list[move_list.size() - 1].value = 0;
        }
    }
    Search_result res;
    int alpha = -HW2, beta = HW2, g, former_alpha = -INF;
    if (b.n + max_depth < HW2){
        for (int depth = min(16, max(0, max_depth - 5)); depth <= max_depth; ++depth){
            alpha = -HW2;
            beta = HW2;
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            move_ordering(&search, move_list, max_depth);
            //move_ordering_value(move_list);
            for (Mobility &mob: move_list){
                search.board.move(&mob);
                    g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, false);
                search.board.undo(&mob);
                //mob.value = g;
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
            if (!search.use_mpc)
                pre_search_mpcts.emplace_back(2.0);
            pre_search_mpcts.emplace_back(USE_DEFAULT_MPC);
            int pv_idx;
            vector<pair<Mobility*, future<pair<int, unsigned long long>>>> parallel_tasks;
            pair<int, unsigned long long> task_result;
            for (double pre_search_mpct: pre_search_mpcts){
                f_n_nodes = search.n_nodes;
                alpha = -HW2;
                beta = HW2;
                if (pre_search_mpct == USE_DEFAULT_MPC){
                    search.use_mpc = use_mpc;
                    search.mpct = mpct;
                } else{
                    search.use_mpc = true;
                    search.mpct = pre_search_mpct;
                }
                parent_transpose_table.ready_next_search();
                child_transpose_table.ready_next_search();
                move_ordering_value(move_list);
                pv_idx = 0;
                parallel_tasks.clear();
                strt3 = tim();
                for (Mobility &mob: move_list){
                    strt2 = tim();
                    f_n_nodes2 = search.n_nodes;
                    search.board.move(&mob);
                    #if USE_MULTI_THREAD
                        if (pre_search_mpct != USE_DEFAULT_MPC && pv_idx > 0 /* mob.value < move_list[0].value - 4 */ && thread_pool.n_idle()){
                            if (pre_search_mpct == USE_DEFAULT_MPC){
                                if (search.use_mpc)
                                    parallel_tasks.emplace_back(make_pair(&mob, thread_pool.push(parallel_negascout, search, -beta, -alpha, depth - 1, true)));
                                else
                                    parallel_tasks.emplace_back(make_pair(&mob, thread_pool.push(parallel_mtd, search, -beta, -alpha, -mob.value, depth - 1, true)));
                            } else
                                parallel_tasks.emplace_back(make_pair(&mob, thread_pool.push(parallel_negascout, search, -HW2, min(HW2, -alpha + PRESEARCH_OFFSET), depth - 1, true)));
                            search.board.undo(&mob);
                        } else{
                            if (pre_search_mpct == USE_DEFAULT_MPC){
                                if (search.use_mpc)
                                    g = -nega_scout(&search, -beta, -alpha, depth - 1, true);
                                else
                                    g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, true);
                            } else
                                g = -nega_scout(&search, -beta, min(HW2, -alpha + PRESEARCH_OFFSET), depth - 1, true);
                            if (pre_search_mpct == USE_DEFAULT_MPC)
                                cerr << "main searching time " << tim() - strt2 << " time from start " << tim() - strt << " policy " << mob.pos << " value " << g << " expected " << mob.value << " nodes " << search.n_nodes - f_n_nodes2 << " nps " << (search.n_nodes - f_n_nodes2) * 1000 / max(1LL, tim() - strt2) << endl;
                            #if USE_LOG
                                cout_div();
                            #endif
                            search.board.undo(&mob);
                            mob.value = g;
                            if (alpha < g){
                                alpha = g;
                                res.policy = mob.pos;
                            }
                        }
                    #else
                        if (pre_search_mpct == USE_DEFAULT_MPC){
                            if (search.use_mpc)
                                g = -nega_scout(&search, -beta, -alpha, depth - 1, true);
                            else
                                g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, true);
                        } else
                            g = -nega_scout(&search, -beta, min(HW2, -alpha + PRESEARCH_OFFSET), depth - 1, true);
                        if (pre_search_mpct == USE_DEFAULT_MPC)
                            cerr << "main searching time " << tim() - strt2 << " time from start " << tim() - strt << " policy " << mob.pos << " value " << g << " expected " << mob.value << " nodes " << search.n_nodes - f_n_nodes2 << " nps " << (search.n_nodes - f_n_nodes2) * 1000 / max(1LL, tim() - strt2) << endl;
                        #if USE_LOG
                            cout_div();
                        #endif
                        search.board.undo(&mob);
                        mob.value = g;
                        if (alpha < g){
                            alpha = g;
                            res.policy = mob.pos;
                        }
                    #endif
                    ++pv_idx;
                }
                for (pair<Mobility*, future<pair<int, unsigned long long>>> &parallel_task: parallel_tasks){
                    task_result = parallel_task.second.get();
                    if (pre_search_mpct == USE_DEFAULT_MPC)
                        cerr << "main parallel searching policy " << parallel_task.first->pos << " value " << task_result.first << " expected " << parallel_task.first->value << " nodes " << task_result.second << endl;
                    parallel_task.first->value = task_result.first;
                    search.n_nodes += task_result.second;
                    if (alpha < task_result.first){
                        alpha = task_result.first;
                        res.policy = parallel_task.first->pos;
                    }
                }
                #if USE_LOG
                    cout_div2();
                #endif
                cerr << "endsearch time " << tim() - strt3 << " time from start " << tim() - strt << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes - f_n_nodes << " nps " << (search.n_nodes - f_n_nodes) * 1000 / max(1LL, tim() - strt3) << endl;
                sum_time += tim() - strt3;
            }
            cerr << "endsearch overall time " << tim() - strt << " search time " << sum_time << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, sum_time) << endl;
        }
    }
    res.depth = max_depth;
    res.nps = search.n_nodes * 1000 / max(1LL, tim() - strt);
    if (former_alpha != -INF)
        res.value = (former_alpha + alpha) / 2;
    else
        res.value = alpha;
    //cerr << child_transpose_table.get_n_reg() << endl;
    #if MOVE_ORDERING_ADJUST
        cout << search.n_nodes << endl;
    #endif
    return res;
}
