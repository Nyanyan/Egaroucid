/*
    Egaroucid Project

    @file lazy_smp.hpp
        Parallel search with Lazy SMP
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "thread_pool.hpp"

//#define LAZYSMP_ENDSEARCH_PRESEARCH_COE 0.75
#define LAZYSMP_ENDSEARCH_PRESEARCH_OFFSET 8

struct Lazy_SMP_task{
    uint_fast8_t mpc_level;
    int depth;
    bool is_end_search;
};

Search_result lazy_smp(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread){
    Search_result result;
    result.value = SCORE_UNDEFINED;
    result.nodes = 0;
    uint64_t strt = tim();
    int main_depth = 1;
    int main_mpc_level = mpc_level;
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    bool is_end_search = (depth == max_depth);
    if (is_end_search){
        main_mpc_level = MPC_74_LEVEL;
    }
    if (show_log){
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }
    std::vector<Search> searches(thread_pool.size() + 1);
    while (main_depth <= depth && main_mpc_level <= mpc_level){
        for (Search &search: searches){
            search.n_nodes = 0;
        }
        bool main_is_end_search = false;
        if (main_depth >= max_depth){
            main_is_end_search = true;
            main_depth = max_depth;
        }
        bool is_last_search = (main_depth == depth) && (main_mpc_level == mpc_level);
        std::vector<std::future<int>> parallel_tasks;
        std::vector<int> sub_depth_arr;
        bool sub_searching = true;
        int sub_depth = main_depth;
        int task_idx = 0;
        if (use_multi_thread && main_depth < depth){
            for (int sub_thread_idx = 0; sub_thread_idx < thread_pool.size() && sub_thread_idx < searches.size(); ++sub_thread_idx){
                int ntz = ntz_uint32(sub_thread_idx + 1);
                int sub_depth = main_depth + 1 + ntz;
                int sub_mpc_level = std::min(MPC_100_LEVEL, main_mpc_level + pop_count_ull((sub_thread_idx + 1) >> (ntz + 1)));
                bool sub_is_end_search = false;
                if (sub_depth >= max_depth){
                    sub_mpc_level = sub_depth - max_depth;
                    if (sub_mpc_level <= main_mpc_level){
                        sub_mpc_level = main_mpc_level + 1;
                    }
                    sub_depth = max_depth;
                    sub_is_end_search = true;
                }
                if (sub_mpc_level <= MPC_100_LEVEL){
                    //std::cerr << sub_thread_idx << " " << sub_depth << " " << SELECTIVITY_PERCENTAGE[sub_mpc_level] << std::endl;
                    searches[task_idx].init(&board, sub_mpc_level, false, true);
                    bool pushed = false;
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[task_idx], -SCORE_MAX, SCORE_MAX, sub_depth, false, LEGAL_UNDEFINED, sub_is_end_search, &sub_searching)));
                    sub_depth_arr.emplace_back(sub_depth);
                    ++task_idx;
                    if (!pushed){
                        parallel_tasks.pop_back();
                        sub_depth_arr.pop_back();
                        --task_idx;
                    }
                }
            }
        }
        int max_sub_search_depth = -1;
        int max_sub_search_mpc_level = 0;
        bool max_is_only_one = false;
        for (int i = 0; i < (int)parallel_tasks.size(); ++i){
            if (sub_depth_arr[i] > max_sub_search_depth){
                max_sub_search_depth = sub_depth_arr[i];
                max_sub_search_mpc_level = searches[i].mpc_level;
                max_is_only_one = true;
            } else if (sub_depth_arr[i] == max_sub_search_depth && max_sub_search_mpc_level < searches[i].mpc_level){
                max_sub_search_mpc_level = searches[i].mpc_level;
                max_is_only_one = true;
            } else if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_search_mpc_level){
                max_is_only_one = false;
            }
        }
        if (max_is_only_one){
            for (int i = 0; i < (int)parallel_tasks.size(); ++i){
                if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_search_mpc_level){
                    searches[i].need_to_see_tt_loop = false; // off the inside-loop tt lookup in the max level thread
                }
            }
        }
        Search main_search;
        main_search.init(&board, main_mpc_level, use_multi_thread, parallel_tasks.size() != 0);
        bool searching = true;
        std::pair<int, int> id_result = first_nega_scout_legal(&main_search, -SCORE_MAX, SCORE_MAX, result.value, main_depth, main_is_end_search, is_last_search && show_log, clogs, use_legal, strt, &searching);
        sub_searching = false;
        for (std::future<int> &task: parallel_tasks){
            task.get();
        }
        for (Search &search: searches){
            result.nodes += search.n_nodes;
        }
        result.nodes += main_search.n_nodes;
        if (result.value != SCORE_UNDEFINED && !main_is_end_search){
            double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
            result.value = round(n_value);
            id_result.first = result.value;
        } else{
            result.value = id_result.first;
        }
        result.policy = id_result.second;
        result.depth = main_depth;
        result.time = tim() - strt;
        result.nps = calc_nps(result.nodes, result.time);
        if (show_log){
            if (is_last_search){
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            if (main_is_end_search){
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (!is_end_search || main_depth < depth - LAZYSMP_ENDSEARCH_PRESEARCH_OFFSET){
            ++main_depth;
        } else{
            if (main_depth < depth){
                main_depth = depth;
                //main_mpc_level = mpc_level;
                main_mpc_level = MPC_74_LEVEL;
            } else{
                
                if (main_mpc_level == MPC_74_LEVEL && mpc_level > MPC_74_LEVEL){
                    main_mpc_level = mpc_level;
                } else{
                    ++main_mpc_level;
                }
                
                //++main_mpc_level;
            }
        }
    }
    result.is_end_search = false;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}


void lazy_smp_hint(Board board, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, int n_display, double values[], int hint_types[]){
    uint64_t strt = tim();
    int main_depth = 1;
    int main_mpc_level = mpc_level;
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    bool is_end_search = (depth == max_depth);
    if (is_end_search){
        main_mpc_level = MPC_74_LEVEL;
    }
    while (main_depth <= depth && main_mpc_level <= mpc_level){
        bool is_last_search_show_log = (main_depth == depth) && (main_mpc_level == mpc_level) && show_log;
        bool main_is_end_search = false;
        if (main_depth >= max_depth){
            main_is_end_search = true;
            main_depth = max_depth;
        }
        /*
        std::vector<Lazy_SMP_task> sub_tasks;
        int sub_depth = main_depth;
        if (use_multi_thread){
            for (int i = 1; i < thread_pool.get_n_idle() / 2; ++i){
                int sub_depth = main_depth + i;
                int sub_mpc_level = main_mpc_level;
                bool sub_is_end_search = false;
                if (sub_depth >= max_depth){
                    if (main_is_end_search){
                        sub_mpc_level = main_mpc_level + sub_depth - max_depth;
                    } else{
                        sub_mpc_level = sub_depth - max_depth;
                    }
                    sub_depth = max_depth;
                    sub_is_end_search = true;
                }
                if (sub_mpc_level <= MPC_100_LEVEL){
                    Lazy_SMP_task sub_task;
                    sub_task.mpc_level = sub_mpc_level;
                    sub_task.depth = sub_depth;
                    sub_task.is_end_search = sub_is_end_search;
                    sub_tasks.emplace_back(sub_task);
                }
            }
        }
        std::vector<std::future<void>> parallel_tasks;
        bool sub_searching = true;
        std::vector<Search> searches(sub_tasks.size());
        for (int i = 0; i < (int)sub_tasks.size(); ++i){
            bool pushed = false;
            searches[i].init(&board, sub_tasks[i].mpc_level, false, true);
            while (!pushed){
                parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&first_nega_scout_hint_sub_thread, &searches[i], sub_tasks[i].depth, sub_tasks[i].is_end_search, use_legal, &sub_searching)));
                if (!pushed){
                    parallel_tasks.pop_back();
                }
            }
        }
        */
        Search main_search;
        //main_search.init(&board, main_mpc_level, use_multi_thread, parallel_tasks.size() != 0);
        main_search.init(&board, main_mpc_level, use_multi_thread, true);
        bool searching = true;
        int hint_type = main_depth;
        if (main_is_end_search){ // endgame & this is last search
            hint_type = SELECTIVITY_PERCENTAGE[main_mpc_level];
        }
        uint64_t use_legal_copy = use_legal;
        first_nega_scout_hint(&main_search, main_depth, depth, main_is_end_search, use_legal, &searching, values, hint_types, hint_type, n_display);
        /*
        sub_searching = false;
        for (std::future<void> &task: parallel_tasks){
            task.get();
        }
        */
        if (is_last_search_show_log){
            std::cerr << "main ";
        } else{
            std::cerr << "pre ";
        }
        if (main_is_end_search){
            std::cerr << "end ";
        } else{
            std::cerr << "mid ";
        }
        //std::cerr << "depth " << main_depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " n_worker " << sub_tasks.size() << std::endl;
        std::cerr << "depth " << main_depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << std::endl;
        if (show_log){
            for (int y = 0; y < HW; ++y){
                for (int x = 0; x < HW; ++x){
                    int cell = HW2_M1 - y * 8 - x;
                    if (1 & (board.get_legal() >> cell))
                        std::cerr << round(values[cell]) << " ";
                    else
                        std::cerr << "  ";
                }
                std::cerr << std::endl;
            }
        }
        if (!is_end_search || main_depth < depth - LAZYSMP_ENDSEARCH_PRESEARCH_OFFSET){
            ++main_depth;
        } else{
            if (main_depth < depth){
                main_depth = depth;
                main_mpc_level = MPC_74_LEVEL;
            } else{
                if (main_mpc_level == MPC_74_LEVEL){
                    main_mpc_level = mpc_level;
                } else{
                    ++main_mpc_level;
                }
            }
        }
    }
}
