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

#define LAZYSMP_ENDSEARCH_PRESEARCH_OFFSET 10

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
    /*
    int main_depth = depth - 5;
    if (main_depth < 1){
        main_depth = 1;
    } else if (main_depth > 10){
        main_depth = 10;
    }
    */
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
    while (main_depth <= depth && main_mpc_level <= mpc_level && global_searching){
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
        int sub_max_mpc_level[61];
        bool sub_searching = true;
        int sub_depth = main_depth;
        if (use_multi_thread && !(is_end_search && main_depth == depth) && main_depth <= 10){
            int max_thread_size = thread_pool.size();
            for (int i = 0; i < main_depth - 14; ++i){
                max_thread_size *= 0.9;
            }
            sub_max_mpc_level[main_depth] = main_mpc_level + 1;
            for (int i = main_depth + 1; i < 61; ++i){
                sub_max_mpc_level[i] = MPC_74_LEVEL;
            }
            for (int sub_thread_idx = 0; sub_thread_idx < max_thread_size && sub_thread_idx < searches.size() && global_searching; ++sub_thread_idx){
                int ntz = ctz_uint32(sub_thread_idx + 1);
                int sub_depth = std::min(max_depth, main_depth + ntz);
                int sub_mpc_level = sub_max_mpc_level[sub_depth];
                bool sub_is_end_search = (sub_depth == max_depth);
                if (sub_mpc_level <= MPC_100_LEVEL){
                    //std::cerr << sub_thread_idx << " " << sub_depth << " " << SELECTIVITY_PERCENTAGE[sub_mpc_level] << std::endl;
                    searches[sub_thread_idx].init(&board, sub_mpc_level, false, true, true);
                    bool pushed = false;
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[sub_thread_idx], -SCORE_MAX, SCORE_MAX, sub_depth, false, LEGAL_UNDEFINED, sub_is_end_search, &sub_searching)));
                    sub_depth_arr.emplace_back(sub_depth);
                    ++sub_max_mpc_level[sub_depth];
                    if (!pushed){
                        parallel_tasks.pop_back();
                        sub_depth_arr.pop_back();
                    }
                }
            }
            int max_sub_search_depth = -1;
            int max_sub_main_mpc_level = 0;
            bool max_is_only_one = false;
            for (int i = 0; i < (int)parallel_tasks.size(); ++i){
                if (sub_depth_arr[i] > max_sub_search_depth){
                    max_sub_search_depth = sub_depth_arr[i];
                    max_sub_main_mpc_level = searches[i].mpc_level;
                    max_is_only_one = true;
                } else if (sub_depth_arr[i] == max_sub_search_depth && max_sub_main_mpc_level < searches[i].mpc_level){
                    max_sub_main_mpc_level = searches[i].mpc_level;
                    max_is_only_one = true;
                } else if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_main_mpc_level){
                    max_is_only_one = false;
                }
            }
            if (max_is_only_one){
                for (int i = 0; i < (int)parallel_tasks.size(); ++i){
                    if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_main_mpc_level){
                        searches[i].need_to_see_tt_loop = false; // off the inside-loop tt lookup in the max level thread
                    }
                }
            }
        }
        Search main_search;
        main_search.init(&board, main_mpc_level, use_multi_thread, parallel_tasks.size() != 0, !is_last_search);
        bool searching = true;
        std::pair<int, int> id_result;
        if (main_is_end_search && main_mpc_level == MPC_100_LEVEL){
            id_result = first_nega_scout_legal(&main_search, result.value - 1, result.value + 1, result.value, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
            if (show_log){
                std::cerr << "  main aspiration depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " window [" << result.value - 1 << ", " << result.value + 1 << "] value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " time " << tim() - strt << std::endl;
            }
            if (id_result.first != result.value){
                int alpha = -SCORE_MAX, beta = SCORE_MAX, expected_value = result.value;
                if (id_result.first < result.value){ // lower than expected
                    beta = result.value - 1;
                    expected_value = result.value - 2;
                } else{ // higher than expected
                    alpha = result.value + 1;
                    expected_value = result.value + 2;
                }
                for (main_mpc_level = MPC_93_LEVEL; main_mpc_level <= MPC_100_LEVEL; ++main_mpc_level){
                    main_search.mpc_level = main_mpc_level; // do presearch again
                    id_result = first_nega_scout_legal(&main_search, alpha, beta, expected_value, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
                    if (show_log){
                        std::cerr << "  again depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_search.mpc_level] << "%" << " window [" << alpha << ", " << beta << "] value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " time " << tim() - strt << std::endl;
                    }
                    expected_value = id_result.first;
                }
                /*
                main_mpc_level = MPC_100_LEVEL;
                main_search.mpc_level = main_mpc_level; // do presearch again
                id_result = first_nega_scout_legal(&main_search, alpha, beta, expected_value, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
                if (show_log){
                    std::cerr << "  again depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_search.mpc_level] << "%" << " window [" << alpha << ", " << beta << "] value " << id_result.first << " policy " << idx_to_coord(id_result.second) << " time " << tim() - strt << std::endl;
                }
                */
            }
        } else{
            id_result = first_nega_scout_legal(&main_search, -SCORE_MAX, SCORE_MAX, result.value, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
        }
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
            std::cerr << "depth " << result.depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << result.value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result.nodes << " time " << result.time << " NPS " << result.nps << std::endl;
        }
        if (!is_end_search || main_depth < depth - LAZYSMP_ENDSEARCH_PRESEARCH_OFFSET){
            if (main_depth <= 15 && main_depth < depth - 3){
                main_depth += 3;
            } else{
                ++main_depth;
            }
        } else{
            if (main_depth < depth){
                main_depth = depth;
                if (depth <= 30 && mpc_level >= MPC_88_LEVEL){
                    main_mpc_level = MPC_88_LEVEL;
                } else{
                    main_mpc_level = MPC_74_LEVEL;
                }
            } else{
                if (main_mpc_level < mpc_level){
                    if (
                        /*
                        (main_mpc_level >= MPC_74_LEVEL && mpc_level > MPC_74_LEVEL && depth <= 22) || 
                        (main_mpc_level >= MPC_88_LEVEL && mpc_level > MPC_88_LEVEL && depth <= 25) || 
                        (main_mpc_level >= MPC_93_LEVEL && mpc_level > MPC_93_LEVEL && depth <= 29) || 
                        (main_mpc_level >= MPC_98_LEVEL && mpc_level > MPC_98_LEVEL && depth <= 31) || 
                        (main_mpc_level >= MPC_99_LEVEL && mpc_level > MPC_99_LEVEL)
                        */
                        /*
                        (main_mpc_level >= MPC_74_LEVEL && mpc_level > MPC_74_LEVEL && depth <= 22) || 
                        (main_mpc_level >= MPC_88_LEVEL && mpc_level > MPC_88_LEVEL)
                        */
                        (main_mpc_level >= MPC_74_LEVEL && mpc_level > MPC_74_LEVEL)
                        ){
                        main_mpc_level = mpc_level;
                    } else{
                        ++main_mpc_level;
                    }
                } else{
                    break;
                }
            }
        }
    }
    result.is_end_search = false;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}
