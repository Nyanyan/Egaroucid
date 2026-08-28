/*
    Egaroucid Project

    @file ybwc.hpp
        Parallel search with YBWC (Young Brothers Wait Concept)
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <exception>
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "parallel.hpp"
#include "thread_pool.hpp"
#include "transposition_cutoff.hpp"
#include "ybwc_completion_group.hpp"

static_assert(MAX_N_BRANCHES <= 64);
using Ybwc_parallel_task_group = Ybwc_completion_group<Parallel_task, MAX_N_BRANCHES>;

/*
    @brief YBWC parameters
*/
constexpr int YBWC_MID_SPLIT_MIN_DEPTH = 6;
//constexpr int YBWC_MID_SPLIT_MAX_DEPTH = 26;
//constexpr int YBWC_END_SPLIT_MAX_DEPTH = 29;
// constexpr int YBWC_N_ELDER_CHILD = 1;
#if IS_GGS_TOURNAMENT
constexpr int YBWC_MID_N_YOUNGER_CHILD = 3;
#else
constexpr int YBWC_MID_N_YOUNGER_CHILD = 2;
#endif
constexpr int YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD = 3;
constexpr int YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH = 23;
constexpr int YBWC_END_N_YOUNGER_CHILD = 1;
#if IS_GGS_TOURNAMENT
constexpr int YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD = 3;
constexpr int YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH = 20;
#else
constexpr int YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD = 6;
constexpr int YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH = 16;
#endif
// constexpr int YBWC_MAX_RUNNING_COUNT = 5;
constexpr int YBWC_NOT_PUSHED = -124;
constexpr int YBWC_PUSHED = 124;

constexpr int MID_NWS_LMR_MIN_DEPTH = 8;
constexpr int MID_NWS_LMR_MIN_MOVE = 4;
inline int mid_nws_lmr_reduction(Search *search, const int depth, const int move_count, const bool is_end_search) {
    if (is_end_search || search->mpc_level >= MPC_100_LEVEL || depth < MID_NWS_LMR_MIN_DEPTH || move_count < MID_NWS_LMR_MIN_MOVE) {
        return 0;
    }
    return 1;
}

#if USE_YBWC_SPLIT_STATISTICS
constexpr int YBWC_STATS_DEPTH_SIZE = HW2 + 1;
constexpr int YBWC_STATS_MOVE_BUCKET_SIZE = 3;
inline std::atomic<uint64_t> ybwc_split_attempt[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_idle_ok[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_move_ok[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_pushed[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_push_failed[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_task_completed[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_task_cancelled[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_task_fail_high[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_task_nodes[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_cancelled_task_nodes[YBWC_STATS_DEPTH_SIZE];
inline std::atomic<uint64_t> ybwc_split_attempt_by_move[YBWC_STATS_DEPTH_SIZE][YBWC_STATS_MOVE_BUCKET_SIZE];
inline std::atomic<uint64_t> ybwc_split_pushed_by_move[YBWC_STATS_DEPTH_SIZE][YBWC_STATS_MOVE_BUCKET_SIZE];
inline std::atomic<uint64_t> ybwc_split_idle_sum;
inline std::atomic<int> ybwc_split_idle_min;
inline std::atomic<int> ybwc_split_idle_max;
inline std::atomic<int> ybwc_tasks_running;
inline std::atomic<int> ybwc_tasks_running_max;
inline std::atomic<uint64_t> ybwc_wait_help_executed;
inline std::atomic<uint64_t> ybwc_wait_yielded;

inline void ybwc_stats_update_max(std::atomic<int> *target, const int value) {
    int observed = target->load(std::memory_order_relaxed);
    while (
        observed < value &&
        !target->compare_exchange_weak(
            observed,
            value,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        )
    ) {}
}

inline void ybwc_stats_update_min(std::atomic<int> *target, const int value) {
    int observed = target->load(std::memory_order_relaxed);
    while (
        value < observed &&
        !target->compare_exchange_weak(
            observed,
            value,
            std::memory_order_relaxed,
            std::memory_order_relaxed
        )
    ) {}
}

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
    ybwc_split_idle_sum = 0;
    ybwc_split_idle_min = THREAD_SIZE_INF;
    ybwc_split_idle_max = 0;
    ybwc_tasks_running = 0;
    ybwc_tasks_running_max = 0;
    ybwc_wait_help_executed = 0;
    ybwc_wait_yielded = 0;
    for (int i = 0; i < YBWC_STATS_DEPTH_SIZE; ++i) {
        ybwc_split_attempt[i] = 0;
        ybwc_split_idle_ok[i] = 0;
        ybwc_split_move_ok[i] = 0;
        ybwc_split_pushed[i] = 0;
        ybwc_split_push_failed[i] = 0;
        ybwc_task_completed[i] = 0;
        ybwc_task_cancelled[i] = 0;
        ybwc_task_fail_high[i] = 0;
        ybwc_task_nodes[i] = 0;
        ybwc_cancelled_task_nodes[i] = 0;
        for (int j = 0; j < YBWC_STATS_MOVE_BUCKET_SIZE; ++j) {
            ybwc_split_attempt_by_move[i][j] = 0;
            ybwc_split_pushed_by_move[i][j] = 0;
        }
    }
}

inline void ybwc_split_stats_print() {
    uint64_t total_attempts = 0;
    for (int depth = 0; depth < YBWC_STATS_DEPTH_SIZE; ++depth) {
        total_attempts += ybwc_split_attempt[depth].load();
    }
    const int idle_min = total_attempts == 0
        ? 0
        : ybwc_split_idle_min.load(std::memory_order_relaxed);
    const double idle_mean = total_attempts == 0
        ? 0.0
        : (double)ybwc_split_idle_sum.load(std::memory_order_relaxed) /
            (double)total_attempts;
    std::cerr << "ybwc runtime stats attempts " << total_attempts
              << " idle_mean " << idle_mean
              << " idle_min " << idle_min
              << " idle_max " << ybwc_split_idle_max.load(std::memory_order_relaxed)
              << " max_running " << ybwc_tasks_running_max.load(std::memory_order_relaxed)
              << " wait_help " << ybwc_wait_help_executed.load(std::memory_order_relaxed)
              << " wait_yield " << ybwc_wait_yielded.load(std::memory_order_relaxed)
              << std::endl;
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
    std::cerr << "ybwc task stats depth completed cancelled fail_high nodes cancelled_nodes" << std::endl;
    for (int depth = 0; depth < YBWC_STATS_DEPTH_SIZE; ++depth) {
        const uint64_t completed = ybwc_task_completed[depth].load();
        const uint64_t cancelled = ybwc_task_cancelled[depth].load();
        if (completed + cancelled == 0) {
            continue;
        }
        std::cerr << depth << " "
                  << completed << " "
                  << cancelled << " "
                  << ybwc_task_fail_high[depth].load() << " "
                  << ybwc_task_nodes[depth].load() << " "
                  << ybwc_cancelled_task_nodes[depth].load()
                  << std::endl;
    }
}
#endif

int nega_alpha_ordering_nws(Search *search, int alpha, const int depth, Nws_node_hint node_hint, uint64_t legal, const bool is_end_search, const Search_cancellation_context &cancellation);
int nega_scout_node(Search *search, int alpha, int beta, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, Search_node_type node_type, bool *searching);

inline int ybwc_poll_task(Ybwc_parallel_task_group &parallel_tasks, Parallel_task *task_result) {
    return parallel_tasks.try_pop(task_result) ? 1 : 0;
}

inline bool ybwc_wait_task_with_help(Ybwc_parallel_task_group &parallel_tasks, thread_id_t thread_id, bool use_help, Parallel_task *task_result) {
    while (true) {
        if (ybwc_poll_task(parallel_tasks, task_result) > 0) {
            return true;
        }
        const bool helped = use_help && thread_pool.try_execute_one(thread_id);
#if USE_YBWC_SPLIT_STATISTICS
        if (helped) {
            ybwc_wait_help_executed.fetch_add(1, std::memory_order_relaxed);
        } else {
            ybwc_wait_yielded.fetch_add(1, std::memory_order_relaxed);
        }
#endif
        if (!helped) {
            parallel_tasks.wait_for_ready();
        }
    }
}

inline bool ybwc_use_endsearch_move(const Search *search, const int child_depth, const bool is_end_search) {
    return is_end_search && (child_depth <= MID_TO_END_DEPTH_MPC || (search->mpc_level == MPC_100_LEVEL && child_depth <= MID_TO_END_DEPTH));
}

inline void ybwc_move_child(Search *search, const Flip *flip, const bool use_endsearch_move) {
    if (use_endsearch_move) {
        search->move_endsearch(flip);
    } else {
        search->move(flip);
    }
}

inline void ybwc_undo_child(Search *search, const Flip *flip, const bool use_endsearch_move) {
    if (use_endsearch_move) {
        search->undo_endsearch(flip);
    } else {
        search->undo(flip);
    }
}

inline int ybwc_end_split_min_depth(const uint_fast8_t mpc_level) {
    return mpc_level == MPC_74_LEVEL
        ? YBWC_SELECTIVE_END_SPLIT_MIN_DEPTH
        : YBWC_END_SPLIT_MIN_DEPTH;
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
    @param cancellation         linked flags for terminating this search
    @return the result in Parallel_task structure
*/
Parallel_task ybwc_do_task_nws(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, bool is_presearch, bool use_dim0_mpc_eval, thread_id_t thread_id, int mid_split_task_limit, int parent_alpha, const int depth, const Nws_node_hint node_hint, uint64_t legal, const bool is_end_search, uint_fast8_t policy, int move_idx, Search_cancellation_context cancellation) {
#if USE_YBWC_SPLIT_STATISTICS
    const int running_tasks = ybwc_tasks_running.fetch_add(1, std::memory_order_relaxed) + 1;
    ybwc_stats_update_max(&ybwc_tasks_running_max, running_tasks);
#endif
    Search search(
        player,
        opponent,
        n_discs,
        parity,
        mpc_level,
        (!is_end_search && depth > YBWC_MID_SPLIT_MIN_DEPTH) ||
            (is_end_search && depth > ybwc_end_split_min_depth(mpc_level)),
        is_presearch,
        thread_id
    );
    search.use_dim0_mpc_eval = use_dim0_mpc_eval;
    search.mid_split_task_limit = mid_split_task_limit;
    Parallel_task task;
    task.value = -nega_alpha_ordering_nws(&search, -parent_alpha - 1, depth, node_hint, legal, is_end_search, cancellation);
    const bool cancelled = !is_searching(cancellation);
    if (cancelled) {
        task.value = SCORE_UNDEFINED;
    } else if (parent_alpha < task.value) {
        search_cancellation_store(cancellation.current, false);
    }
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    task.move_idx = move_idx;
#if USE_YBWC_SPLIT_STATISTICS
    const int stats_depth = std::clamp(depth, 0, YBWC_STATS_DEPTH_SIZE - 1);
    ybwc_task_nodes[stats_depth].fetch_add(task.n_nodes, std::memory_order_relaxed);
    if (cancelled) {
        ybwc_task_cancelled[stats_depth].fetch_add(1, std::memory_order_relaxed);
        ybwc_cancelled_task_nodes[stats_depth].fetch_add(task.n_nodes, std::memory_order_relaxed);
    } else {
        ybwc_task_completed[stats_depth].fetch_add(1, std::memory_order_relaxed);
        if (parent_alpha < task.value) {
            ybwc_task_fail_high[stats_depth].fetch_add(1, std::memory_order_relaxed);
        }
    }
    ybwc_tasks_running.fetch_sub(1, std::memory_order_relaxed);
#endif
    return task;
}



/*
    @brief Try to do parallel NWS (Null Window Search)

    @param search               searching information
    @param alpha                alpha value
    @param depth                remaining depth
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param cancellation         linked flags for terminating this search
    @param policy               the last move
    @param pv_idx               the priority of this move
    @param seems_to_be_all_node     this node seems to be ALL node?
    @param parallel_tasks       completion group of splitted tasks
    @return task splitted?
*/
inline int ybwc_split_nws(Search *search, int parent_alpha, const int depth, const Nws_node_hint node_hint, uint64_t legal, const bool is_end_search, const Search_cancellation_context &cancellation, uint_fast8_t policy, const int n_remaining_moves, const int move_idx, const int running_count, Ybwc_parallel_task_group &parallel_tasks) {
    #if USE_YBWC_SPLIT_STATISTICS
        ++ybwc_split_attempt[depth];
        int move_bucket = ybwc_stats_move_bucket(n_remaining_moves);
        ++ybwc_split_attempt_by_move[depth][move_bucket];
    #endif
    const int n_idle = thread_pool.get_n_idle();
    bool idle_ok = n_idle > 0;
    const int n_younger_child = is_end_search
        ? (YBWC_END_MIN_REMAINING_MOVES > 0
            ? YBWC_END_MIN_REMAINING_MOVES
            : (depth <= YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH ? YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD : YBWC_END_N_YOUNGER_CHILD))
        : (depth <= YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH ? YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD : YBWC_MID_N_YOUNGER_CHILD);
    bool move_ok = n_remaining_moves >= n_younger_child;
    #if USE_YBWC_SPLIT_STATISTICS
        ybwc_split_idle_sum.fetch_add((uint64_t)std::max(0, n_idle), std::memory_order_relaxed);
        ybwc_stats_update_min(&ybwc_split_idle_min, n_idle);
        ybwc_stats_update_max(&ybwc_split_idle_max, n_idle);
        if (idle_ok) {
            ++ybwc_split_idle_ok[depth];
        }
        if (move_ok) {
            ++ybwc_split_move_ok[depth];
        }
    #endif
    if (
            idle_ok &&                                  // There is an idle thread
            n_remaining_moves >= n_younger_child         // This node is not the (some) youngest brother
    ) {
        // int v;
        // if (transposition_cutoff_nws(search, search->board.hash(), depth, -parent_alpha - 1, &v)) {
        //     return -v;
        // }
        // if (!is_end_search && search->mpc_level < MPC_100_LEVEL && depth >= USE_MPC_MIN_DEPTH) {
        //     if (mpc(search, -parent_alpha - 1, -parent_alpha, depth, legal, is_end_search, &v, cancellation)) {
        //         return -v;
        //     }
        // }
        if (is_searching(cancellation)) {
            bool pushed;
            const int task_limit = is_end_search ? YBWC_END_MAX_SPLIT_TASKS : search->mid_split_task_limit;
            const Nws_node_hint task_node_hint = Nws_node_hint::no_static_eval();
            const std::size_t task_slot = parallel_tasks.reserve_slot();
            auto search_task = std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, search->is_presearch, search->use_dim0_mpc_eval, search->thread_id, search->mid_split_task_limit, parent_alpha, depth, task_node_hint, legal, is_end_search, policy, move_idx, cancellation);
            auto completion_task = [search_task = std::move(search_task), completion_group = &parallel_tasks, task_slot]() noexcept {
                try {
                    completion_group->publish(task_slot, search_task());
                } catch (...) {
                    std::terminate();
                }
            };
            if (task_limit == THREAD_SIZE_INF) {
                std::future<void> ignored_future = thread_pool.push(search->thread_id, &pushed, completion_task);
                (void)ignored_future;
            } else {
                std::future<void> ignored_future = thread_pool.push(search->thread_id, task_limit, &pushed, completion_task);
                (void)ignored_future;
            }
            if (pushed) {
                parallel_tasks.mark_submitted(task_slot);
                #if USE_YBWC_SPLIT_STATISTICS
                    ++ybwc_split_pushed[depth];
                    ++ybwc_split_pushed_by_move[depth][move_bucket];
                #endif
                return YBWC_PUSHED;
            } else {
                #if USE_YBWC_SPLIT_STATISTICS
                    ++ybwc_split_push_failed[depth];
                #endif
            }
        }
    }
    return YBWC_NOT_PUSHED;
}



#if USE_YBWC_NWS
inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, const Search_cancellation_context &parent_cancellation) {
    alignas(std::atomic_ref<bool>::required_alignment) bool n_searching = true;
    const Search_cancellation_context cancellation{&n_searching, &parent_cancellation};
    Ybwc_parallel_task_group parallel_tasks;
    int canput = (int)move_list.size();
    int running_count = 0;
    int g;
    bool searched;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(cancellation); ++move_idx) {
        //swap_next_best_move(move_list, move_idx, canput);
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            searched = false;
            const Nws_node_hint child_hint = child_nws_hint(move_list[move_idx], is_end_search);
            const bool use_endsearch_move = ybwc_use_endsearch_move(search, depth - 1, is_end_search);
            ybwc_move_child(search, &move_list[move_idx].flip, use_endsearch_move);
                int ybwc_split_state = ybwc_split_nws(search, alpha, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else {
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        const int lmr_reduction = mid_nws_lmr_reduction(search, depth, n_moves_seen, is_end_search);
                        if (lmr_reduction > 0) {
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1 - lmr_reduction, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                            if (alpha < g && is_searching(cancellation)) {
                                g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                            }
                        } else {
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                        }
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(cancellation)) {
                        searched = true;
                        if (*v < g) {
                            *v = g;
                            *best_move = move_list[move_idx].flip.pos;
                            if (alpha < g) {
                                search_cancellation_store(&n_searching, false);
                            }
                        }
                    }
                }
            ybwc_undo_child(search, &move_list[move_idx].flip, use_endsearch_move);
            if (searched) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    // thread_pool.start_idling();
    Parallel_task task_result;
#if USE_YBWC_SPLITTED_TASK_TERMINATION
    if (is_searching(cancellation) && *v <= alpha && running_count >= 2 && ((is_end_search && depth >= 28) || (!is_end_search && depth >= 24))) {
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
        if (is_searching(cancellation) && *v <= alpha && running_count >= 2) {
            search_cancellation_store(&n_searching, false); // terminate splitted tasks
            while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
                --running_count;
                search->n_nodes += task_result.n_nodes;
            }
            if (is_searching(parent_cancellation)) {
                ybwc_search_young_brothers_nws(search, alpha, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, parent_cancellation);
            }
            return;
        }
    }
#endif
    while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
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
}




inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, Flip_value move_list[], int canput, const Search_cancellation_context &parent_cancellation) {
    alignas(std::atomic_ref<bool>::required_alignment) bool n_searching = true;
    const Search_cancellation_context cancellation{&n_searching, &parent_cancellation};
    Ybwc_parallel_task_group parallel_tasks;
    int running_count = 0;
    int g;
    bool searched;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && is_searching(cancellation); ++move_idx) {
        //swap_next_best_move(move_list, move_idx, canput);
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            searched = false;
            const Nws_node_hint child_hint = child_nws_hint(move_list[move_idx], is_end_search);
            const bool use_endsearch_move = ybwc_use_endsearch_move(search, depth - 1, is_end_search);
            ybwc_move_child(search, &move_list[move_idx].flip, use_endsearch_move);
                int ybwc_split_state = ybwc_split_nws(search, alpha, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else {
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        const int lmr_reduction = mid_nws_lmr_reduction(search, depth, n_moves_seen, is_end_search);
                        if (lmr_reduction > 0) {
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1 - lmr_reduction, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                            if (alpha < g && is_searching(cancellation)) {
                                g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                            }
                        } else {
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                        }
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (is_searching(cancellation)) {
                        searched = true;
                        if (*v < g) {
                            *v = g;
                            *best_move = move_list[move_idx].flip.pos;
                            if (alpha < g) {
                                search_cancellation_store(&n_searching, false);
                            }
                        }
                    }
                }
            ybwc_undo_child(search, &move_list[move_idx].flip, use_endsearch_move);
            if (searched) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    // thread_pool.start_idling();
    Parallel_task task_result;
#if USE_YBWC_SPLITTED_TASK_TERMINATION
    if (is_searching(cancellation) && *v <= alpha && running_count >= 2 && ((is_end_search && depth >= 28) || (!is_end_search && depth >= 24))) {
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
        if (is_searching(cancellation) && *v <= alpha && running_count >= 2) {
            search_cancellation_store(&n_searching, false); // terminate splitted tasks
            while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
                --running_count;
                search->n_nodes += task_result.n_nodes;
            }
            if (is_searching(parent_cancellation)) {
                ybwc_search_young_brothers_nws(search, alpha, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, canput, parent_cancellation);
            }
            return;
        }
    }
#endif
    while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
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
}
#endif

#if USE_YBWC_NEGASCOUT
void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, Search_node_type node_type, bool need_best_move, bool *searching) {
    alignas(std::atomic_ref<bool>::required_alignment) bool n_searching = true;
    const Search_cancellation_context root_cancellation{searching, nullptr};
    const Search_cancellation_context cancellation{&n_searching, &root_cancellation};
    Ybwc_parallel_task_group parallel_tasks;
    int canput = (int)move_list.size();
    int running_count = 0;
    int g;
    std::vector<int> research_idxes;
    bool cutoff_found = false;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && search_cancellation_load(searching) && search_cancellation_load(&n_searching); ++move_idx) {
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            bool move_done = false;
            const Nws_node_hint child_hint = child_nws_hint(move_list[move_idx], is_end_search);
            const bool use_endsearch_move = ybwc_use_endsearch_move(search, depth - 1, is_end_search);
            ybwc_move_child(search, &move_list[move_idx].flip, use_endsearch_move);
                int ybwc_split_state = ybwc_split_nws(search, *alpha, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else{
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (search_cancellation_load(searching) && search_cancellation_load(&n_searching)) {
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
                            search_cancellation_store(&n_searching, false);
                        } else{
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                            }
                            move_done = true;
                        }
                    }
                }
            ybwc_undo_child(search, &move_list[move_idx].flip, use_endsearch_move);
            if (move_done) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    if (running_count) {
        // thread_pool.start_idling();
        Parallel_task task_result;
        while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
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
                g = -nega_scout_node(search, -(*beta), -(*alpha), depth - 1, false, move_list[research_idx].n_legal, is_end_search, search_child_node_type(node_type), searching);
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
            ybwc_search_young_brothers(search, alpha, beta, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, node_type, need_best_move, searching);
        }
    }
}




void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, Flip_value move_list[], int canput, Search_node_type node_type, bool need_best_move, bool *searching) {
    alignas(std::atomic_ref<bool>::required_alignment) bool n_searching = true;
    const Search_cancellation_context root_cancellation{searching, nullptr};
    const Search_cancellation_context cancellation{&n_searching, &root_cancellation};
    Ybwc_parallel_task_group parallel_tasks;
    int running_count = 0;
    int g;
    std::vector<int> research_idxes;
    bool cutoff_found = false;
    int n_searched = 0;
    int n_moves_seen = 0;
    for (int move_idx = 0; move_idx < canput && search_cancellation_load(searching) && search_cancellation_load(&n_searching); ++move_idx) {
        if (move_list[move_idx].flip.flip) {
            ++n_moves_seen;
            bool move_done = false;
            const Nws_node_hint child_hint = child_nws_hint(move_list[move_idx], is_end_search);
            const bool use_endsearch_move = ybwc_use_endsearch_move(search, depth - 1, is_end_search);
            ybwc_move_child(search, &move_list[move_idx].flip, use_endsearch_move);
                int ybwc_split_state = ybwc_split_nws(search, *alpha, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                if (ybwc_split_state == YBWC_PUSHED) {
                    ++running_count;
                } else{
                    if (ybwc_split_state == YBWC_NOT_PUSHED) {
                        g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, child_hint, move_list[move_idx].n_legal, is_end_search, cancellation);
                    } else{
                        g = ybwc_split_state;
                        ++search->n_nodes;
                    }
                    if (search_cancellation_load(searching) && search_cancellation_load(&n_searching)) {
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
                            search_cancellation_store(&n_searching, false);
                        } else{
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                            }
                            move_done = true;
                        }
                    }
                }
            ybwc_undo_child(search, &move_list[move_idx].flip, use_endsearch_move);
            if (move_done) {
                move_list[move_idx].flip.flip = 0;
                ++n_searched;
            }
        }
    }
    if (running_count) {
        // thread_pool.start_idling();
        Parallel_task task_result;
        while (running_count > 0 && ybwc_wait_task_with_help(parallel_tasks, search->thread_id, !is_end_search || YBWC_END_WAIT_HELP, &task_result)) {
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
                g = -nega_scout_node(search, -(*beta), -(*alpha), depth - 1, false, move_list[research_idx].n_legal, is_end_search, search_child_node_type(node_type), searching);
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
            ybwc_search_young_brothers(search, alpha, beta, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, canput, node_type, need_best_move, searching);
        }
    }
}
#endif
