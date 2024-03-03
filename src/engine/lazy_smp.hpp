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

#define N_PARALLEL_MAX 128
#define MAIN_THREAD_IDX 0

#define LAZYSMP_DEPTH_COE 3.0
#define LAZYSMP_MPC_COE 0.25
#define LAZYSMP_ENDSEARCH_PRESEARCH_COE 0.6


void lazy_smp_sub_thread(Search *search, int depth, bool is_end_search, const bool *searching){
    while (!(*searching));
    nega_scout(search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, is_end_search, searching);
}


Search_result lazy_smp(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs){
    Search searches[N_PARALLEL_MAX];
    for (int i = 0; i < N_PARALLEL_MAX; ++i){
        searches[i].init_board(&board);
        searches[i].n_nodes = 0ULL;
        searches[i].use_multi_thread = i == MAIN_THREAD_IDX; // combine with YBWC in main thread
    }
    Search_result result;
    result.value = SCORE_UNDEFINED;
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
        searches[MAIN_THREAD_IDX].mpc_level = main_mpc_level;
        bool main_is_end_search = false;
        if (main_depth >= max_depth){
            main_is_end_search = true;
            main_depth = max_depth;
        }

        bool sub_searching = false;
        std::vector<std::future<void>> parallel_tasks;
        for (int thread_idx = MAIN_THREAD_IDX + 1; thread_idx < N_PARALLEL_MAX && thread_idx - MAIN_THREAD_IDX - 1 < thread_pool.size(); ++thread_idx){
            int sub_depth = main_depth + thread_idx - MAIN_THREAD_IDX;
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
                bool pushed = false;
                searches[thread_idx].mpc_level = sub_mpc_level;
                while (!pushed){
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&lazy_smp_sub_thread, &searches[thread_idx], sub_depth, sub_is_end_search, &sub_searching)));
                    if (!pushed){
                        parallel_tasks.pop_back();
                    }
                }
            }
        }
        sub_searching = true;
        std::pair<int, int> id_result = first_nega_scout(&searches[MAIN_THREAD_IDX], -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, main_depth, main_is_end_search, is_last_search_show_log, clogs, strt);
        sub_searching = false;
        if (result.value != SCORE_UNDEFINED && !main_is_end_search){
            double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
            result.value = round(n_value);
            id_result.first = result.value;
        } else{
            result.value = id_result.first;
        }
        result.policy = id_result.second;
        for (std::future<void> &task: parallel_tasks)
            task.get();
        result.depth = main_depth;
        result.nodes = 0;
        for (int i = 0; i < N_PARALLEL_MAX; ++i){
            result.nodes += searches[i].n_nodes;
        }
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
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[searches[MAIN_THREAD_IDX].mpc_level] << "%" << " value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
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
