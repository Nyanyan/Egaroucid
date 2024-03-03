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

#define MID_TO_END_THRESHOLD_COE 1.0

Search_result lazy_smp_midsearch(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs){
    Search searches[N_PARALLEL_MAX];
    for (int i = 0; i < N_PARALLEL_MAX; ++i){
        searches[i].init_board(&board);
        searches[i].n_nodes = 0ULL;
        searches[i].use_multi_thread = false; // <= no need
        searches[i].mpc_level = mpc_level;
    }
    Search_result result;
    result.value = SCORE_UNDEFINED;
    uint64_t strt = tim();
    for (int main_thread_depth = 1; main_thread_depth <= depth; ++main_thread_depth){
        bool sub_thread_searching = true;
        std::vector<std::future<int>> parallel_tasks;
        for (int thread_idx = MAIN_THREAD_IDX + 1; thread_idx < N_PARALLEL_MAX && (int)parallel_tasks.size() < thread_pool.size() && thread_pool.get_n_idle(); ++thread_idx){
            int sub_thread_depth = main_thread_depth + (int)(3.0 * log(1.0 + thread_idx));
            bool sub_thread_is_end_search = false;
            int max_depth = HW2 - searches[thread_idx].n_discs;
            if (sub_thread_depth >= max_depth){
                searches[thread_idx].mpc_level = std::min(MPC_100_LEVEL, MPC_74_LEVEL + (sub_thread_depth - max_depth) / 4);
                sub_thread_depth = max_depth;
                sub_thread_is_end_search = true;
            }
            bool pushed = false;
            while (!pushed){
                parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[thread_idx], -SCORE_MAX, SCORE_MAX, sub_thread_depth, false, LEGAL_UNDEFINED, sub_thread_is_end_search, &sub_thread_searching)));
                if (!pushed){
                    parallel_tasks.pop_back();
                }
            }
        }
        bool is_main_search = (main_thread_depth == depth);
        if (is_main_search && show_log){
            std::cerr << "start main search" << std::endl;
        }
        std::pair<int, int> id_result = first_nega_scout(&searches[MAIN_THREAD_IDX], -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, main_thread_depth, false, is_main_search, clogs, strt);
        sub_thread_searching = false;
        if (main_thread_depth >= depth - 1){
            if (result.value == SCORE_UNDEFINED){
                result.value = id_result.first;
            } else{
                double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
                result.value = round(n_value);
                id_result.first = result.value;
            }
            result.policy = id_result.second;
        }
        for (std::future<int> &task: parallel_tasks)
            task.get();
        result.depth = main_thread_depth;
        result.nodes = 0;
        for (int i = 0; i < N_PARALLEL_MAX; ++i){
            result.nodes += searches[i].n_nodes;
        }
        result.time = tim() - strt;
        result.nps = calc_nps(result.nodes, result.time);
        if (show_log){
            if (is_main_search){
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[searches[MAIN_THREAD_IDX].mpc_level] << "%" << " value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
    }
    result.is_end_search = false;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}


Search_result lazy_smp_endsearch(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs){
    Search searches[N_PARALLEL_MAX];
    for (int i = 0; i < N_PARALLEL_MAX; ++i){
        searches[i].init_board(&board);
        searches[i].n_nodes = 0ULL;
        searches[i].use_multi_thread = false; // <= no need
    }
    Search_result result;
    result.value = SCORE_UNDEFINED;
    uint64_t strt = tim();
    int main_thread_depth = 1;
    uint_fast8_t main_thread_mpc_level = MPC_74_LEVEL;
    while (main_thread_depth <= depth && main_thread_mpc_level <= mpc_level){
        bool sub_thread_searching = true;
        std::vector<std::future<int>> parallel_tasks;
        for (int thread_idx = MAIN_THREAD_IDX + 1; thread_idx < N_PARALLEL_MAX && (int)parallel_tasks.size() < thread_pool.size() && thread_pool.get_n_idle(); ++thread_idx){
            int sub_thread_depth = main_thread_depth + (int)(3.0 * log(1.0 + thread_idx));
            bool sub_thread_is_end_search = false;
            int max_depth = HW2 - searches[thread_idx].n_discs;
            if (sub_thread_depth >= max_depth){
                searches[thread_idx].mpc_level = std::min(MPC_100_LEVEL, main_thread_mpc_level + (int)(0.4 * log(1.0 + thread_idx)));
                sub_thread_depth = max_depth;
                sub_thread_is_end_search = true;
            }
            std::cerr << thread_idx << " " << sub_thread_depth << " " << SELECTIVITY_PERCENTAGE[searches[thread_idx].mpc_level] << std::endl;
            bool pushed = false;
            while (!pushed){
                parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[thread_idx], -SCORE_MAX, SCORE_MAX, sub_thread_depth, false, LEGAL_UNDEFINED, sub_thread_is_end_search, &sub_thread_searching)));
                if (!pushed){
                    parallel_tasks.pop_back();
                }
            }
        }
        bool is_main_search = (main_thread_depth == depth) && (main_thread_mpc_level == mpc_level);
        bool is_end_search = (main_thread_depth == depth);
        if (is_main_search && show_log){
            std::cerr << "start main search" << std::endl;
        }
        searches[MAIN_THREAD_IDX].mpc_level = main_thread_mpc_level;
        std::pair<int, int> id_result = first_nega_scout(&searches[MAIN_THREAD_IDX], -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, main_thread_depth, is_end_search, is_main_search, clogs, strt);
        sub_thread_searching = false;
        if (main_thread_depth >= depth - 1){
            if (result.value == SCORE_UNDEFINED){
                result.value = id_result.first;
            } else{
                double n_value = (0.9 * result.value + 1.1 * id_result.first) / 2.0;
                result.value = round(n_value);
                id_result.first = result.value;
            }
            result.policy = id_result.second;
        }
        for (std::future<int> &task: parallel_tasks)
            task.get();
        result.depth = main_thread_depth;
        result.nodes = 0;
        for (int i = 0; i < N_PARALLEL_MAX; ++i){
            result.nodes += searches[i].n_nodes;
        }
        result.time = tim() - strt;
        result.nps = calc_nps(result.nodes, result.time);
        if (show_log){
            if (is_main_search){
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[searches[MAIN_THREAD_IDX].mpc_level] << "%" << " value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (main_thread_depth < depth * MID_TO_END_THRESHOLD_COE){
            ++main_thread_depth;
        } else{
            main_thread_depth = depth;
            ++main_thread_mpc_level;
        }
    }
    result.is_end_search = true;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}
