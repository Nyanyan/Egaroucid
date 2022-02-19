#pragma once
#include <iostream>
#include <future>
#include "level.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

#define SEARCH_FINAL 100
#define SEARCH_BOOK -1
#define BOOK_CUT_THRESHOLD_DIV 2
#define USE_DEFAULT_MPC -1.0
#define PRESEARCH_OFFSET 6
#define PARALLEL_SPLIT_DIV 6

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
        for (int depth = min(18, max(1, max_depth - 5)); depth <= max_depth; ++depth){
            alpha = -HW2 - 1;
            beta = HW2;
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            search.tt_parent_idx = parent_transpose_table.now_idx();
            search.tt_child_idx = child_transpose_table.now_idx();
            move_ordering(&search, move_list, depth, alpha, beta, false);
            //move_ordering_value(move_list);
            for (Mobility &mob: move_list){
                search.board.move(&mob);
                    //g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, false);
                    g = -nega_scout(&search, -beta, -alpha, depth - 1, false);
                search.board.undo(&mob);
                //mob.value = g;
                if (alpha < g){
                    child_transpose_table.reg(search.tt_child_idx, &search.board, hash_code, mob.pos, g);
                    alpha = g;
                    res.policy = mob.pos;
                }
            }
            if (depth == max_depth - 1)
                former_alpha = alpha;
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
        if (former_alpha != -INF)
            alpha = (alpha + former_alpha) / 2;
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
            search.tt_parent_idx = parent_transpose_table.now_idx();
            search.tt_child_idx = child_transpose_table.now_idx();
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
                                g = -nega_scout(&search, -beta, -alpha, depth - 1, true) / 2 * 2;
                            else
                                g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, true);
                        } else
                            g = -mtd(&search, -beta, min(HW2, -alpha + PRESEARCH_OFFSET), -mob.value, depth - 1, true);
                            //g = -nega_scout(&search, -beta, min(HW2, -alpha + PRESEARCH_OFFSET), depth - 1, true);
                        if (pre_search_mpct == USE_DEFAULT_MPC)
                            cerr << "main searching time " << tim() - strt2 << " time from start " << tim() - strt << " mpct " << search.mpct << " policy " << mob.pos << " value " << g << " expected " << mob.value << " nodes " << search.n_nodes - f_n_nodes2 << " nps " << (search.n_nodes - f_n_nodes2) * 1000 / max(1LL, tim() - strt2) << endl;
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
                            g = -nega_scout(&search, -beta, -alpha, depth - 1, true) / 2 * 2;
                        else
                            g = -mtd(&search, -beta, -alpha, -mob.value, depth - 1, true);
                    } else
                        g = -nega_scout(&search, -beta, min(HW2, -alpha + PRESEARCH_OFFSET), depth - 1, true) / 2 * 2;
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
                task_result.first = task_result.first / 2 * 2;
                if (alpha < task_result.first){
                    alpha = task_result.first;
                    res.policy = parallel_task.first->pos;
                }
            }
            #if USE_LOG
                cout_div2();
            #endif
            cerr << "endsearch depth " << depth << " time " << tim() - strt3 << " time from start " << tim() - strt << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes - f_n_nodes << " nps " << (search.n_nodes - f_n_nodes) * 1000 / max(1LL, tim() - strt3) << endl;
            sum_time += tim() - strt3;
        }
        cerr << "endsearch depth " << depth << " overall time " << tim() - strt << " search time " << sum_time << " mpct " << search.mpct << " policy " << res.policy << " value " << alpha << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, sum_time) << endl;
    }
    res.depth = max_depth;
    res.nps = search.n_nodes * 1000 / max(1LL, tim() - strt);
    res.value = alpha;
    //cerr << child_transpose_table.get_n_reg() << endl;
    #if MOVE_ORDERING_ADJUST
        cout << search.n_nodes << endl;
    #endif
    return res;
}

inline int tree_search_value(Board b, int max_depth, bool use_mpc, double mpct, const vector<int> vacant_lst, bool show_log){
    long long strt = tim();
    int res = -INF, f_res = -INF;
    Search search;
    search.board = b;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    int g = 0;
    if (b.n + max_depth < HW2){
        for (int depth = min(18, max(0, max_depth - 5)); depth <= max_depth; ++depth){
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            search.tt_parent_idx = parent_transpose_table.now_idx();
            search.tt_child_idx = child_transpose_table.now_idx();
            //g = mtd(&search, -HW2, HW2, g, depth, false);
            g = nega_scout(&search, -HW2, HW2, depth, false);
            if (depth == max_depth - 1)
                f_res = g;
            else if (depth == max_depth)
                res = g;
            if (show_log)
                cerr << "midsearch time " << tim() - strt << " depth " << depth << " value " << g << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
        if (f_res != -INF)
            res = (res + f_res) / 2;
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
        pre_search_mpcts.emplace_back(USE_DEFAULT_MPC);
        vector<pair<Mobility*, future<pair<int, unsigned long long>>>> parallel_tasks;
        pair<int, unsigned long long> task_result;
        for (double pre_search_mpct: pre_search_mpcts){
            if (pre_search_mpct == USE_DEFAULT_MPC){
                search.use_mpc = use_mpc;
                search.mpct = mpct;
            } else{
                search.use_mpc = true;
                search.mpct = pre_search_mpct;
            }
            parent_transpose_table.ready_next_search();
            child_transpose_table.ready_next_search();
            search.tt_parent_idx = parent_transpose_table.now_idx();
            search.tt_child_idx = child_transpose_table.now_idx();
            if (search.use_mpc)
                g = nega_scout(&search, -HW2, HW2, depth, true) / 2 * 2;
            else
                g = mtd(&search, -HW2, HW2, g, depth, true);
            if (show_log)
                cerr << "endsearch depth " << depth << " time " << tim() - strt << " mpct " << search.mpct << " value " << g << " nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1LL, tim() - strt) << endl;
        }
        res = g;
    }
    return res;
}

int tree_search_value_no_id(Board b, int depth, bool use_mpc, double mpct, const vector<int> vacant_lst, int pre_searched_value){
    int res;
    Search search;
    search.board = b;
    search.skipped = false;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.vacant_list = vacant_lst;
    search.n_nodes = 0;
    search.tt_parent_idx = parent_transpose_table.now_idx();
    search.tt_child_idx = child_transpose_table.now_idx();
    if (b.n + depth < HW2){
        res = nega_scout(&search, -HW2, HW2, depth, false);
        //res = mtd(&search, -HW2, HW2, pre_searched_value, depth, false);
    } else{
        if (search.use_mpc)
            res = nega_scout(&search, -HW2, HW2, depth, true) / 2 * 2;
        else
            res = mtd(&search, -HW2, HW2, pre_searched_value, depth, true);
    }
    return res;
}

Search_result ai(Board b, int level, int book_error, const vector<int> vacant_lst){
    Search_result res;
    Book_value book_result = book.get_random(&b, book_error);
    if (book_result.policy != -1){
        cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = -1;
        res.nps = 0;
    }
    else if (level == 0){
        unsigned long long legal = b.mobility_ull();
        vector<int> move_lst;
        for (const int &cell: vacant_lst){
            if (1 & (legal >> cell))
                move_lst.emplace_back(cell);
        }
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = mid_evaluate(&b);
        res.depth = 0;
        res.nps = 0;
    } else{
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        cerr << "level status " << level << " " << b.n - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        res = tree_search(b, depth, use_mpc, mpct, vacant_lst);
    }
    return res;
}

bool ai_hint(Board b, int level, int max_level, int res[], int info[], const int pre_searched_values[], unsigned long long legal, vector<int> vacant_lst){
    Mobility mob;
    Board nb;
    future<int> val_future[HW2];
    int depth;
    bool use_mpc, is_mid_search;
    double mpct;
    get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    if (!is_mid_search && level != max_level && !use_mpc)
        return false;
    if (depth - 1 >= 0){
        parent_transpose_table.ready_next_search();
        child_transpose_table.ready_next_search();
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&mob, &b, i);
                b.move_copy(&mob, &nb);
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    val_future[i] = async(launch::async, tree_search_value_no_id, nb, depth - 1, use_mpc, mpct, vacant_lst, -pre_searched_values[i]);
                    if (!is_mid_search && !use_mpc)
                        info[i] = SEARCH_FINAL;
                    else
                        info[i] = level;
                } else
                    info[i] = SEARCH_BOOK;
            }
        }
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                if (res[i] == -INF){
                    res[i] = -val_future[i].get();
                }
            }
        }
    } else{
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&mob, &b, i);
                b.move_copy(&mob, &nb);
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    res[i] = -mid_evaluate(&nb);
                    info[i] = level;
                } else
                    info[i] = SEARCH_BOOK;
            }
        }
    }
    return true;
}

int ai_book(Board b, int level, int book_learn_accept, vector<int> vacant_lst){
    int cut_level = level / BOOK_CUT_THRESHOLD_DIV;
    int depth;
    bool use_mpc, is_mid_search;
    double mpct;
    int res, g;
    get_level(cut_level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    g = -tree_search_value(b, depth, use_mpc, mpct, vacant_lst, true);
    if (abs(g) == INF)
        return -INF;
    if (g < -book_learn_accept - 2)
        return -INF;
    get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    res = -tree_search_value(b, depth, use_mpc, mpct, vacant_lst, true);
    if (abs(res) == INF)
        return -INF;
    return res;
}