/*
    Egaroucid Project

    @file ybwc.hpp
        Parallel search with YBWC (Young Brothers Wait Concept)
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include "setting.hpp"
#include "common.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "parallel.hpp"
#include "thread_pool.hpp"
#include "transposition_cutoff.hpp"

/*
    @brief YBWC parameters
*/
constexpr int YBWC_MID_SPLIT_MIN_DEPTH = 6;
//constexpr int YBWC_MID_SPLIT_MAX_DEPTH = 26;
constexpr int YBWC_END_SPLIT_MIN_DEPTH = 16;
//constexpr int YBWC_END_SPLIT_MAX_DEPTH = 29;
// constexpr int YBWC_N_ELDER_CHILD = 1;
constexpr int YBWC_N_YOUNGER_CHILD = 1;
constexpr int YBWC_MAX_SLAVES_PER_SPLIT = 7;
constexpr int YBWC_MAX_SLAVES_PER_SMALL_SPLIT = 3;
constexpr int YBWC_SHARED_NWS_MAX_DEPTH = 16;
// constexpr int YBWC_MAX_RUNNING_COUNT = 5;
constexpr int YBWC_NOT_PUSHED = -124;
constexpr int YBWC_PUSHED = 124;

#if USE_YBWC_SPLIT_STATISTICS
constexpr int YBWC_STATS_DEPTH_SIZE = HW2 + 1;
constexpr int YBWC_STATS_MOVE_BUCKET_SIZE = 3;
inline std::atomic<uint64_t> ybwc_split_attempt[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_idle_ok[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_move_ok[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_pushed[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_push_failed[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_attempt_by_move[YBWC_STATS_DEPTH_SIZE][YBWC_STATS_MOVE_BUCKET_SIZE];
inline std::atomic<uint64_t> ybwc_split_pushed_by_move[YBWC_STATS_DEPTH_SIZE][YBWC_STATS_MOVE_BUCKET_SIZE];

inline int ybwc_stats_move_bucket(const int n_remaining_moves) {
    if (n_remaining_moves <= 1) {
        return 0;
    }
    if (n_remaining_moves == 2) {
        return 1;
    }
    return 2;
}

inline void ybwc_split_stats_reset() {
    for (int i = 0; i < YBWC_STATS_DEPTH_SIZE; ++i) {
        ybwc_split_attempt[i] = 0;
        ybwc_split_idle_ok[i] = 0;
        ybwc_split_move_ok[i] = 0;
        ybwc_split_pushed[i] = 0;
        ybwc_split_push_failed[i] = 0;
        for (int j = 0; j < YBWC_STATS_MOVE_BUCKET_SIZE; ++j) {
            ybwc_split_attempt_by_move[i][j] = 0;
            ybwc_split_pushed_by_move[i][j] = 0;
        }
    }
}

inline void ybwc_split_stats_print() {
    std::cerr << "ybwc split stats depth attempt idle_ok move_ok pushed push_failed" << std::endl;
    for (int depth = 0; depth < YBWC_STATS_DEPTH_SIZE; ++depth) {
        uint64_t attempt = ybwc_split_attempt[depth].load();
        if (attempt == 0) {
            continue;
        }
        std::cerr << depth << " "
                  << attempt << " "
                  << ybwc_split_idle_ok[depth].load() << " "
                  << ybwc_split_move_ok[depth].load() << " "
                  << ybwc_split_pushed[depth].load() << " "
                  << ybwc_split_push_failed[depth].load()
                  << std::endl;
    }
    std::cerr << "ybwc split stats by_move depth rem1 rem2 rem3plus pushed1 pushed2 pushed3plus" << std::endl;
    for (int depth = 0; depth < YBWC_STATS_DEPTH_SIZE; ++depth) {
        uint64_t rem1 = ybwc_split_attempt_by_move[depth][0].load();
        uint64_t rem2 = ybwc_split_attempt_by_move[depth][1].load();
        uint64_t rem3 = ybwc_split_attempt_by_move[depth][2].load();
        if (rem1 + rem2 + rem3 == 0) {
            continue;
        }
        std::cerr << depth << " "
                  << rem1 << " "
                  << rem2 << " "
                  << rem3 << " "
                  << ybwc_split_pushed_by_move[depth][0].load() << " "
                  << ybwc_split_pushed_by_move[depth][1].load() << " "
                  << ybwc_split_pushed_by_move[depth][2].load()
                  << std::endl;
    }
}
#endif

int nega_alpha_ordering_nws(Search *search, int alpha, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, Searchings &searchings);
inline bool is_searching(Searchings &searchings);

inline bool ybwc_searching_flag(const Search_Stop_Token &searching) {
    return search_stop_token_is_searching(searching);
}

inline void ybwc_stop_searching(std::atomic_bool *searching, bool *searching_raw = nullptr) {
    searching->store(false, std::memory_order_relaxed);
    if (searching_raw != nullptr) {
        *searching_raw = false;
    }
}

inline int ybwc_poll_task(std::vector<std::future<Parallel_task>> &parallel_tasks, Parallel_task *task_result) {
    bool has_valid_task = false;
    for (std::future<Parallel_task> &task: parallel_tasks) {
        if (!task.valid()) {
            continue;
        }
        has_valid_task = true;
        if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            *task_result = task.get();
            return 1;
        }
    }
    return has_valid_task ? 0 : -1;
}

inline bool ybwc_wait_task_with_help(std::vector<std::future<Parallel_task>> &parallel_tasks, thread_id_t thread_id, bool use_help, Parallel_task *task_result) {
    while (true) {
        int task_state = ybwc_poll_task(parallel_tasks, task_result);
        if (task_state > 0) {
            return true;
        }
        if (task_state < 0) {
            return false;
        }
        if (!use_help || !thread_pool.try_execute_one(thread_id)) {
            std::this_thread::yield();
        }
    }
}

inline int ybwc_poll_void_task(std::vector<std::future<void>> &parallel_tasks) {
    bool has_valid_task = false;
    for (std::future<void> &task: parallel_tasks) {
        if (!task.valid()) {
            continue;
        }
        has_valid_task = true;
        if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            task.get();
            return 1;
        }
    }
    return has_valid_task ? 0 : -1;
}

inline void ybwc_wait_void_tasks_with_help(std::vector<std::future<void>> &parallel_tasks, thread_id_t thread_id, bool use_help) {
    while (true) {
        int task_state = ybwc_poll_void_task(parallel_tasks);
        if (task_state < 0) {
            return;
        }
        if (task_state == 0 && (!use_help || !thread_pool.try_execute_one(thread_id))) {
            std::this_thread::yield();
        }
    }
}

/*
    @brief Wrapper for parallel NWS (Null Window Search)

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @param n_discs              number of discs on the board
    @param parity               parity of the board
    @param mpc_level            MPC (Multi-ProbCut) probability level
    @param alpha                alpha value
    @param depth                remaining depth
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param policy               the last move
    @param searching            flag for terminating this search
    @return the result in Parallel_task structure
*/
Parallel_task ybwc_do_task_nws(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, bool is_presearch, thread_id_t thread_id, int parent_alpha, const int depth, uint64_t legal, const bool is_end_search, uint_fast8_t policy, int move_idx, Searchings searchings, std::atomic_bool *n_searching, bool *n_searching_raw) {
    Search search(player, opponent, n_discs, parity, mpc_level, (!is_end_search && depth > YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth > YBWC_END_SPLIT_MIN_DEPTH), is_presearch, thread_id);
    Parallel_task task;
    task.value = -nega_alpha_ordering_nws(&search, -parent_alpha - 1, depth, false, legal, is_end_search, searchings);
    if (!is_searching(searchings)) {
        task.value = SCORE_UNDEFINED;
    } else if (parent_alpha < task.value) {
        ybwc_stop_searching(n_searching, n_searching_raw);
    }
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    task.move_idx = move_idx;
    return task;
}



/*
    @brief Try to do parallel NWS (Null Window Search)

    @param search               searching information
    @param alpha                alpha value
    @param depth                remaining depth
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param searching          flag for terminating this search
    @param policy               the last move
    @param pv_idx               the priority of this move
    @param seems_to_be_all_node     this node seems to be ALL node?
    @param parallel_tasks       vector of splitted tasks
    @return task splitted?
*/
inline int ybwc_split_nws(Search *search, int parent_alpha, const int depth, uint64_t legal, const bool is_end_search, Searchings &searchings, std::atomic_bool *n_searching, bool *n_searching_raw, uint_fast8_t policy, const int n_remaining_moves, const int move_idx, const int running_count, std::vector<std::future<Parallel_task>> &parallel_tasks) {
    #if USE_YBWC_SPLIT_STATISTICS
        ++ybwc_split_attempt[depth];
        int move_bucket = ybwc_stats_move_bucket(n_remaining_moves);
        ++ybwc_split_attempt_by_move[depth][move_bucket];
    #endif
    bool idle_ok = thread_pool.get_n_idle() > 0;
    bool move_ok = n_remaining_moves >= YBWC_N_YOUNGER_CHILD;
    #if USE_YBWC_SPLIT_STATISTICS
        if (idle_ok) {
            ++ybwc_split_idle_ok[depth];
        }
        if (move_ok) {
            ++ybwc_split_move_ok[depth];
        }
    #endif
    if (
            idle_ok &&                                  // There is an idle thread
            n_remaining_moves >= YBWC_N_YOUNGER_CHILD    // This node is not the (some) youngest brother
    ) {
        // int v;
        // if (transposition_cutoff_nws(search, search->board.hash(), depth, -parent_alpha - 1, &v)) {
        //     return -v;
        // }
        // if (!is_end_search && search->mpc_level < MPC_100_LEVEL && depth >= USE_MPC_MIN_DEPTH) {
        //     if (mpc(search, -parent_alpha - 1, -parent_alpha, depth, legal, is_end_search, &v, searchings)) {
        //         return -v;
        //     }
        // }
        if (is_searching(searchings)) {
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(search->thread_id, &pushed, std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, search->is_presearch, search->thread_id, parent_alpha, depth, legal, is_end_search, policy, move_idx, searchings, n_searching, n_searching_raw)));
            if (pushed) {
                #if USE_YBWC_SPLIT_STATISTICS
                    ++ybwc_split_pushed[depth];
                    ++ybwc_split_pushed_by_move[depth][move_bucket];
                #endif
                return YBWC_PUSHED;
            } else {
                #if USE_YBWC_SPLIT_STATISTICS
                    ++ybwc_split_push_failed[depth];
                #endif
                parallel_tasks.pop_back();
            }
        }
    }
    return YBWC_NOT_PUSHED;
}

struct YBWC_NWS_Split_State {
    std::atomic<int> next_idx;
    std::atomic<uint64_t> n_nodes;
    std::mutex mtx;
    int best_value;
    int best_move;

    YBWC_NWS_Split_State(int first_idx, int value, int move)
        : next_idx(first_idx), n_nodes(0), best_value(value), best_move(move) {}
};

inline void ybwc_do_shared_nws_task_search(Search *local_search, int alpha, const int depth, const bool is_end_search, Flip_value *move_list, const int canput, Searchings searchings, std::atomic_bool *n_searching, bool *n_searching_raw, YBWC_NWS_Split_State *state, bool collect_nodes) {
    const int child_depth = depth - 1;
    const uint64_t n_nodes_before = local_search->n_nodes;
    while (is_searching(searchings)) {
        const int move_idx = state->next_idx.fetch_add(1, std::memory_order_relaxed);
        if (move_idx >= canput) {
            break;
        }
        if (!move_list[move_idx].flip.flip) {
            continue;
        }
        bool searched = false;
        local_search->move(&move_list[move_idx].flip);
            const int g = -nega_alpha_ordering_nws(local_search, -alpha - 1, child_depth, false, move_list[move_idx].n_legal, is_end_search, searchings);
        local_search->undo(&move_list[move_idx].flip);
        if (is_searching(searchings)) {
            searched = true;
            std::lock_guard<std::mutex> lock(state->mtx);
            if (state->best_value < g) {
                state->best_value = g;
                state->best_move = move_list[move_idx].flip.pos;
                if (alpha < g) {
                    ybwc_stop_searching(n_searching, n_searching_raw);
                }
            }
        }
        if (searched) {
            move_list[move_idx].flip.flip = 0;
        }
    }
    if (collect_nodes) {
        state->n_nodes.fetch_add(local_search->n_nodes - n_nodes_before, std::memory_order_relaxed);
    }
}

inline void ybwc_do_shared_nws_task(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, bool is_presearch, thread_id_t thread_id, int alpha, const int depth, const bool is_end_search, Flip_value *move_list, const int canput, Searchings searchings, std::atomic_bool *n_searching, bool *n_searching_raw, YBWC_NWS_Split_State *state) {
    const int child_depth = depth - 1;
    Search local_search(player, opponent, n_discs, parity, mpc_level, (!is_end_search && child_depth > YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && child_depth > YBWC_END_SPLIT_MIN_DEPTH), is_presearch, thread_id);
    ybwc_do_shared_nws_task_search(&local_search, alpha, depth, is_end_search, move_list, canput, searchings, n_searching, n_searching_raw, state, true);
}

inline bool ybwc_try_shared_nws_split(Search *search, int alpha, int *v, int *best_move, int n_available_moves, int depth, bool is_end_search, Flip_value *move_list, int canput, Searchings &searchings, std::atomic_bool *n_searching, bool *n_searching_raw) {
    if (is_end_search || depth > YBWC_SHARED_NWS_MAX_DEPTH || n_available_moves <= 1 || thread_pool.get_n_idle() <= 0) {
        return false;
    }
    const int max_slaves = thread_pool.size() >= 24 ? YBWC_MAX_SLAVES_PER_SPLIT : YBWC_MAX_SLAVES_PER_SMALL_SPLIT;
    int n_slaves = std::min(thread_pool.get_n_idle(), n_available_moves - 1);
    n_slaves = std::min(n_slaves, max_slaves);
    if (n_slaves <= 0) {
        return false;
    }

    YBWC_NWS_Split_State state(0, *v, *best_move);
    std::vector<std::future<void>> parallel_tasks;
    parallel_tasks.reserve(n_slaves);
    for (int i = 0; i < n_slaves; ++i) {
        bool pushed;
        parallel_tasks.emplace_back(thread_pool.push(search->thread_id, &pushed, std::bind(&ybwc_do_shared_nws_task, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, search->is_presearch, search->thread_id, alpha, depth, is_end_search, move_list, canput, searchings, n_searching, n_searching_raw, &state)));
        if (!pushed) {
            parallel_tasks.pop_back();
            break;
        }
    }
    if (parallel_tasks.empty()) {
        return false;
    }

    ybwc_do_shared_nws_task_search(search, alpha, depth, is_end_search, move_list, canput, searchings, n_searching, n_searching_raw, &state, false);
    ybwc_wait_void_tasks_with_help(parallel_tasks, search->thread_id, !is_end_search);
    search->n_nodes += state.n_nodes.load(std::memory_order_relaxed);
    *v = state.best_value;
    *best_move = state.best_move;
    return true;
}


#if USE_YBWC_NWS
inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, Searchings &searchings) {
    std::vector<std::future<Parallel_task>> parallel_tasks;
    std::atomic_bool n_searching(true);
    bool n_searching_raw = true;
    searchings.emplace_back(&n_searching, &n_searching_raw);
    int canput = (int)move_list.size();
    if (ybwc_try_shared_nws_split(search, alpha, v, best_move, n_available_moves, depth, is_end_search, move_list.data(), canput, searchings, &n_searching, &n_searching_raw)) {
        searchings.pop_back();
        return;
    }
    int running_count = 0;
    int g;
    bool searched;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(searchings); ++move_idx) {
        //swap_next_best_move(move_list, move_idx, canput);
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            searched = false;
            search->move(&move_list[move_idx].flip);
                int ybwc_split_state = ybwc_split_nws(search, alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, &n_searching_raw, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else {
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(searchings)) {
                        searched = true;
                        if (*v < g) {
                            *v = g;
                            *best_move = move_list[move_idx].flip.pos;
                            if (alpha < g) {
                                ybwc_stop_searching(&n_searching, &n_searching_raw);
                            }
                        }
                    }
                }
            search->undo(&move_list[move_idx].flip);
            if (searched) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    // thread_pool.start_idling();
    Parallel_task task_result;
#if USE_YBWC_SPLITTED_TASK_TERMINATION
    if (is_searching(searchings) && *v <= alpha && running_count >= 2 && ((is_end_search && depth >= 28) || (!is_end_search && depth >= 24))) {
        while (ybwc_poll_task(parallel_tasks, &task_result) > 0) {
            --running_count;
            search->n_nodes += task_result.n_nodes;
            if (task_result.value != SCORE_UNDEFINED) {
                if (*v < task_result.value) {
                    *v = task_result.value;
                    *best_move = move_list[task_result.move_idx].flip.pos;
                }
                move_list[task_result.move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
        if (is_searching(searchings) && *v <= alpha && running_count >= 2) {
            ybwc_stop_searching(&n_searching, &n_searching_raw); // terminate splitted tasks
            while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
                --running_count;
                search->n_nodes += task_result.n_nodes;
            }
            searchings.pop_back(); // pop n_searching
            if (is_searching(searchings)) {
                ybwc_search_young_brothers_nws(search, alpha, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, searchings);
            }
            return;
        }
    }
#endif
    while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
        --running_count;
        search->n_nodes += task_result.n_nodes;
        if (task_result.value != SCORE_UNDEFINED) {
            if (*v < task_result.value) {
                *v = task_result.value;
                *best_move = move_list[task_result.move_idx].flip.pos;
                // if (alpha < task_result.value) {
                //     n_searching = false;
                // }
            }
        }
    }
    // thread_pool.finish_idling();
    searchings.pop_back();
}




inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, Flip_value move_list[], int canput, Searchings &searchings) {
    std::vector<std::future<Parallel_task>> parallel_tasks;
    std::atomic_bool n_searching(true);
    bool n_searching_raw = true;
    searchings.emplace_back(&n_searching, &n_searching_raw);
    if (ybwc_try_shared_nws_split(search, alpha, v, best_move, n_available_moves, depth, is_end_search, move_list, canput, searchings, &n_searching, &n_searching_raw)) {
        searchings.pop_back();
        return;
    }
    int running_count = 0;
    int g;
    bool searched;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(searchings); ++move_idx) {
        //swap_next_best_move(move_list, move_idx, canput);
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            searched = false;
            search->move(&move_list[move_idx].flip);
                int ybwc_split_state = ybwc_split_nws(search, alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, &n_searching_raw, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else {
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(searchings)) {
                        searched = true;
                        if (*v < g) {
                            *v = g;
                            *best_move = move_list[move_idx].flip.pos;
                            if (alpha < g) {
                                ybwc_stop_searching(&n_searching, &n_searching_raw);
                            }
                        }
                    }
                }
            search->undo(&move_list[move_idx].flip);
            if (searched) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    // thread_pool.start_idling();
    Parallel_task task_result;
#if USE_YBWC_SPLITTED_TASK_TERMINATION
    if (is_searching(searchings) && *v <= alpha && running_count >= 2 && ((is_end_search && depth >= 28) || (!is_end_search && depth >= 24))) {
        while (ybwc_poll_task(parallel_tasks, &task_result) > 0) {
            --running_count;
            search->n_nodes += task_result.n_nodes;
            if (task_result.value != SCORE_UNDEFINED) {
                if (*v < task_result.value) {
                    *v = task_result.value;
                    *best_move = move_list[task_result.move_idx].flip.pos;
                }
                move_list[task_result.move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
        if (is_searching(searchings) && *v <= alpha && running_count >= 2) {
            ybwc_stop_searching(&n_searching, &n_searching_raw); // terminate splitted tasks
            while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
                --running_count;
                search->n_nodes += task_result.n_nodes;
            }
            searchings.pop_back(); // pop n_searching
            if (is_searching(searchings)) {
                ybwc_search_young_brothers_nws(search, alpha, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, canput, searchings);
            }
            return;
        }
    }
#endif
    while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
        --running_count;
        search->n_nodes += task_result.n_nodes;
        if (task_result.value != SCORE_UNDEFINED) {
            if (*v < task_result.value) {
                *v = task_result.value;
                *best_move = move_list[task_result.move_idx].flip.pos;
                // if (alpha < task_result.value) {
                //     n_searching = false;
                // }
            }
        }
    }
    // thread_pool.finish_idling();
    // while (!parallel_tasks.empty()) {
    //     bool progress = false;
    //     for (auto it = parallel_tasks.begin(); it != parallel_tasks.end();) {
    //         if (it->valid() && it->wait_for(std::chrono::microseconds(0)) == std::future_status::ready) {
    //             task_result = it->get();
    //             search->n_nodes += task_result.n_nodes;
    //             if (task_result.value != SCORE_UNDEFINED && *v < task_result.value) {
    //                 *v = task_result.value;
    //                 *best_move = move_list[task_result.move_idx].flip.pos;
    //                 if (alpha < task_result.value) {
    //                     n_searching = false;
    //                 }
    //             }
    //             it = parallel_tasks.erase(it);
    //             progress = true;
    //         } else {
    //             ++it;
    //         }
    //     }
    //     if (!progress) {
    //         std::this_thread::yield();
    //     }
    // }
    searchings.pop_back();
}
#endif

#if USE_YBWC_NEGASCOUT
void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, bool need_best_move, bool *searching) {
    std::vector<std::future<Parallel_task>> parallel_tasks;
    std::atomic_bool n_searching(true);
    bool n_searching_raw = true;
    Searchings searchings = {searching, Search_Stop_Token(&n_searching, &n_searching_raw)};
    int canput = (int)move_list.size();
    int running_count = 0;
    int g;
    std::vector<int> research_idxes;
    bool cutoff_found = false;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(searchings); ++move_idx) {
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            bool move_done = false;
            search->move(&move_list[move_idx].flip);
                int ybwc_split_state = ybwc_split_nws(search, *alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, &n_searching_raw, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else{
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(searchings)) {
                        if (*alpha < g) {
                            if (g >= *beta) {
                                if (*v < g) {
                                    *v = g;
                                    *best_move = move_list[move_idx].flip.pos;
                                }
                                cutoff_found = true;
                            } else {
                                research_idxes.emplace_back(move_idx);
                            }
                            ybwc_stop_searching(&n_searching, &n_searching_raw);
                        } else{
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                            }
                            move_done = true;
                        }
                    }
                }
            search->undo(&move_list[move_idx].flip);
            if (move_done) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    if (running_count) {
        // thread_pool.start_idling();
        Parallel_task task_result;
        while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
            --running_count;
            search->n_nodes += task_result.n_nodes;
            if (!cutoff_found && task_result.value != SCORE_UNDEFINED) {
                if (*alpha < task_result.value) {
                    if (task_result.value >= *beta) {
                        if (*v < task_result.value) {
                            *v = task_result.value;
                            *best_move = move_list[task_result.move_idx].flip.pos;
                        }
                        cutoff_found = true;
                    } else {
                        research_idxes.emplace_back(task_result.move_idx);
                    }
                } else {
                    if (*v < task_result.value) {
                        *v = task_result.value;
                        *best_move = move_list[task_result.move_idx].flip.pos;
                    }
                    move_list[task_result.move_idx].flip.flip = 0;
                    ++n_searched;
                }
            }
        }
        // thread_pool.finish_idling();
    }
    if (!cutoff_found && research_idxes.size() && *alpha < *beta && *searching) {
        for (const int &research_idx: research_idxes) {
            search->move(&move_list[research_idx].flip);
                g = -nega_scout(search, -(*beta), -(*alpha), depth - 1, false, move_list[research_idx].n_legal, is_end_search, searching);
            search->undo(&move_list[research_idx].flip);
            move_list[research_idx].flip.flip = 0;
            ++n_searched;
            if (*searching) {
                if (*v < g) {
                    *v = g;
                    *best_move = move_list[research_idx].flip.pos;
                }
                if (*alpha < g) {
                    *alpha = g;
                    if (*alpha >= *beta) {
                        break;
                    }
                }
            }
        }
        if (*alpha < *beta && *searching) {
            ybwc_search_young_brothers(search, alpha, beta, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, need_best_move, searching);
        }
    }
}




void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, Flip_value move_list[], int canput, bool need_best_move, bool *searching) {
    std::vector<std::future<Parallel_task>> parallel_tasks;
    std::atomic_bool n_searching(true);
    bool n_searching_raw = true;
    Searchings searchings = {searching, Search_Stop_Token(&n_searching, &n_searching_raw)};
    int running_count = 0;
    int g;
    std::vector<int> research_idxes;
    bool cutoff_found = false;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(searchings); ++move_idx) {
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            bool move_done = false;
            search->move(&move_list[move_idx].flip);
                int ybwc_split_state = ybwc_split_nws(search, *alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, &n_searching_raw, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else{
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(searchings)) {
                        if (*alpha < g) {
                            if (g >= *beta) {
                                if (*v < g) {
                                    *v = g;
                                    *best_move = move_list[move_idx].flip.pos;
                                }
                                cutoff_found = true;
                            } else {
                                research_idxes.emplace_back(move_idx);
                            }
                            ybwc_stop_searching(&n_searching, &n_searching_raw);
                        } else{
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                            }
                            move_done = true;
                        }
                    }
                }
            search->undo(&move_list[move_idx].flip);
            if (move_done) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    if (running_count) {
        // thread_pool.start_idling();
        Parallel_task task_result;
        while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search, &task_result)) {
            --running_count;
            search->n_nodes += task_result.n_nodes;
            if (!cutoff_found && task_result.value != SCORE_UNDEFINED) {
                if (*alpha < task_result.value) {
                    if (task_result.value >= *beta) {
                        if (*v < task_result.value) {
                            *v = task_result.value;
                            *best_move = move_list[task_result.move_idx].flip.pos;
                        }
                        cutoff_found = true;
                    } else {
                        research_idxes.emplace_back(task_result.move_idx);
                    }
                } else {
                    if (*v < task_result.value) {
                        *v = task_result.value;
                        *best_move = move_list[task_result.move_idx].flip.pos;
                    }
                    move_list[task_result.move_idx].flip.flip = 0;
                    ++n_searched;
                }
            }
        }
        // thread_pool.finish_idling();
    }
    if (!cutoff_found && research_idxes.size() && *alpha < *beta && *searching) {
        for (const int &research_idx: research_idxes) {
            search->move(&move_list[research_idx].flip);
                g = -nega_scout(search, -(*beta), -(*alpha), depth - 1, false, move_list[research_idx].n_legal, is_end_search, searching);
            search->undo(&move_list[research_idx].flip);
            move_list[research_idx].flip.flip = 0;
            ++n_searched;
            if (*searching) {
                if (*v < g) {
                    *v = g;
                    *best_move = move_list[research_idx].flip.pos;
                }
                if (*alpha < g) {
                    *alpha = g;
                    if (*alpha >= *beta) {
                        break;
                    }
                }
            }
        }
        if (*alpha < *beta && *searching) {
            ybwc_search_young_brothers(search, alpha, beta, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, canput, need_best_move, searching);
        }
    }
}
#endif
