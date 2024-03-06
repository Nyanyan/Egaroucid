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

#define LAZYSMP_ENDSEARCH_PRESEARCH_COE 0.7

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
    while (main_depth <= depth && main_mpc_level <= mpc_level){
        bool is_last_search_show_log = (main_depth == depth) && (main_mpc_level == mpc_level) && show_log;
        bool main_is_end_search = false;
        if (main_depth >= max_depth){
            main_is_end_search = true;
            main_depth = max_depth;
        }
        std::vector<Lazy_SMP_task> sub_tasks;
        int sub_depth = main_depth;
        if (use_multi_thread){
            for (int i = 1; i < thread_pool.get_n_idle(); ++i){
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
        std::vector<std::future<std::pair<int, int>>> parallel_tasks;
        bool sub_searching = true;
        std::vector<Search> searches(sub_tasks.size());
        for (int i = 0; i < (int)sub_tasks.size(); ++i){
            bool pushed = false;
            searches[i].init(&board, sub_tasks[i].mpc_level, false);
            while (!pushed){
                parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&first_nega_scout_legal, &searches[i], -SCORE_MAX, SCORE_MAX, result.value, sub_tasks[i].depth, sub_tasks[i].is_end_search, false, clogs, use_legal, strt, &sub_searching)));
                if (!pushed){
                    parallel_tasks.pop_back();
                }
            }
        }
        std::pair<int, int> id_result;
        Search main_search;
        main_search.init(&board, main_mpc_level, use_multi_thread);
        bool searching = true;
        id_result = first_nega_scout_legal(&main_search, -SCORE_MAX, SCORE_MAX, result.value, main_depth, main_is_end_search, is_last_search_show_log, clogs, use_legal, strt, &searching);
        sub_searching = false;
        for (std::future<std::pair<int, int>> &task: parallel_tasks){
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
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " n_worker " << sub_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (!is_end_search || main_depth < round(LAZYSMP_ENDSEARCH_PRESEARCH_COE * depth)){
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
    result.is_end_search = false;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}
