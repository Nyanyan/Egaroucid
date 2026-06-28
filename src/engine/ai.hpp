/*
    Egaroucid Project

    @file ai.hpp
        Main algorithm of Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include <iomanip>
#include <sstream>
#include "level.hpp"
#include "setting.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"
#include "clogsearch.hpp"
#include "time_management.hpp"

constexpr int AI_TYPE_BOOK = 1000;

constexpr int IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET = 8;
constexpr int IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT = 8;
constexpr int PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT = 4;

constexpr int PONDER_START_SELFPLAY_DEPTH = 17;
constexpr double PONDER_UCB_COE = 0.6;
#if IS_GGS_TOURNAMENT
constexpr bool PONDER_USE_SELFPLAY = false;
#else
constexpr bool PONDER_USE_SELFPLAY = true;
#endif

constexpr int AI_TL_EARLY_BREAK_THRESHOLD = 5;
#if IS_GGS_TOURNAMENT
constexpr bool AI_TL_USE_EARLY_BREAK = false;
#else
constexpr bool AI_TL_USE_EARLY_BREAK = true;
#endif
constexpr int AI_TL_MID_VERIFY_MIN_DEPTH = 26;
#if IS_GGS_TOURNAMENT
    #ifndef GGS_TOURNAMENT_MID_VERIFY_93_MIN_DEPTH
        #define GGS_TOURNAMENT_MID_VERIFY_93_MIN_DEPTH 30
    #endif
constexpr int AI_TL_GGS_MID_VERIFY_93_MIN_DEPTH = GGS_TOURNAMENT_MID_VERIFY_93_MIN_DEPTH;
#endif
constexpr int AI_TL_EARLY_BREAK_VERIFY_MIN_DEPTH = 29;
#if IS_GGS_TOURNAMENT
    #ifndef GGS_TOURNAMENT_POLICY_CHANGE_VERIFY_MIN_DEPTH
        #define GGS_TOURNAMENT_POLICY_CHANGE_VERIFY_MIN_DEPTH 28
    #endif
constexpr int AI_TL_POLICY_CHANGE_VERIFY_MIN_DEPTH = GGS_TOURNAMENT_POLICY_CHANGE_VERIFY_MIN_DEPTH;
    #ifndef GGS_TOURNAMENT_ENDGAME_INITIAL_MPC_LEVEL
        #define GGS_TOURNAMENT_ENDGAME_INITIAL_MPC_LEVEL MPC_74_LEVEL
    #endif
constexpr int AI_TL_ENDGAME_INITIAL_MPC_LEVEL = GGS_TOURNAMENT_ENDGAME_INITIAL_MPC_LEVEL;
    #ifndef GGS_TOURNAMENT_VERIFY_TIMEOUT_KEEP_MARGIN
        #define GGS_TOURNAMENT_VERIFY_TIMEOUT_KEEP_MARGIN 2
    #endif
constexpr int AI_TL_VERIFY_TIMEOUT_KEEP_MARGIN = GGS_TOURNAMENT_VERIFY_TIMEOUT_KEEP_MARGIN;
    #ifndef GGS_TOURNAMENT_VERIFY_TIMEOUT_POSITIVE_KEEP_MARGIN
        #define GGS_TOURNAMENT_VERIFY_TIMEOUT_POSITIVE_KEEP_MARGIN 2
    #endif
constexpr int AI_TL_GGS_VERIFY_TIMEOUT_POSITIVE_KEEP_MARGIN = GGS_TOURNAMENT_VERIFY_TIMEOUT_POSITIVE_KEEP_MARGIN;
#else
constexpr int AI_TL_POLICY_CHANGE_VERIFY_MIN_DEPTH = 24;
constexpr int AI_TL_ENDGAME_INITIAL_MPC_LEVEL = MPC_74_LEVEL;
constexpr int AI_TL_VERIFY_TIMEOUT_KEEP_MARGIN = 0;
#endif
constexpr uint_fast8_t AI_TL_POLICY_CHANGE_VERIFY_MPC_LEVEL = MPC_93_LEVEL;
#if IS_GGS_TOURNAMENT
constexpr uint64_t AI_TL_GGS_POLICY_CHANGE_VERIFY_MAX_TIME = 9000ULL;
constexpr double AI_TL_GGS_POLICY_CHANGE_VERIFY_TIME_COE = 0.40;
constexpr int AI_TL_GGS_DEFENSIVE_ALT_VERIFY_EARLY_MAX_N_EMPTY = 48;
constexpr int AI_TL_GGS_DEFENSIVE_ALT_VERIFY_EARLY_MIN_DEPTH = 30;
constexpr int AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_DEPTH = 32;
constexpr int AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MAX_N_EMPTY = 48;
constexpr uint_fast8_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_MPC_LEVEL = MPC_74_LEVEL;
constexpr uint_fast8_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_SEARCH_MPC_LEVEL = MPC_88_LEVEL;
constexpr double AI_TL_GGS_DEFENSIVE_ALT_VERIFY_VALUE_MIN = -18.0;
constexpr double AI_TL_GGS_DEFENSIVE_ALT_VERIFY_VALUE_MAX = -3.0;
constexpr uint64_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_TIME_LEFT = 1800ULL;
constexpr uint64_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_MAIN_TIME = 14000ULL;
constexpr uint64_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MAX_TIME = 7000ULL;
constexpr uint64_t AI_TL_GGS_DEFENSIVE_ALT_VERIFY_RESERVE_TIME = 2600ULL;
constexpr double AI_TL_GGS_DEFENSIVE_ALT_VERIFY_TIME_COE = 0.25;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_MIN_N_EMPTY = 36;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_MAX_N_EMPTY = 43;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_MIN_DEPTH = 29;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_VALUE_MIN = -12;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_VALUE_MAX = -6;
constexpr int AI_TL_GGS_BAD_STABLE_SWITCH_MAX_GAIN = 0;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_MIN_N_EMPTY = 29;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_MAX_N_EMPTY = 48;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_MIN_DEPTH = 25;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_MIN_PREV_PROBABILITY = 88;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_VALUE_MIN = 0;
constexpr int AI_TL_GGS_HIGH_CONF_STICK_MAX_GAIN = 2;
constexpr int AI_TL_GGS_NARROW_ALT_VERIFY_MIN_N_EMPTY = 29;
constexpr int AI_TL_GGS_NARROW_ALT_VERIFY_MAX_N_EMPTY = 48;
constexpr int AI_TL_GGS_NARROW_ALT_VERIFY_MIN_DEPTH = 30;
constexpr int AI_TL_GGS_NARROW_ALT_VERIFY_VALUE_MIN = -2;
constexpr int AI_TL_GGS_NARROW_ALT_VERIFY_VALUE_MAX = 8;
constexpr uint_fast8_t AI_TL_GGS_NARROW_ALT_VERIFY_MIN_MPC_LEVEL = MPC_74_LEVEL;
constexpr uint_fast8_t AI_TL_GGS_NARROW_ALT_VERIFY_SEARCH_MPC_LEVEL = MPC_88_LEVEL;
constexpr uint64_t AI_TL_GGS_NARROW_ALT_VERIFY_MIN_MAIN_TIME = 8000ULL;
constexpr uint64_t AI_TL_GGS_NARROW_ALT_VERIFY_MIN_TIME_LEFT = 2200ULL;
constexpr uint64_t AI_TL_GGS_NARROW_ALT_VERIFY_MAX_TIME = 5200ULL;
constexpr double AI_TL_GGS_NARROW_ALT_VERIFY_TIME_COE = 0.28;
constexpr int AI_TL_GGS_ALT_VERIFY_RETRY_DEPTH_GAP = 3;
#endif

constexpr double AI_TL_ADDITIONAL_SEARCH_THRESHOLD = 1.5;

struct Ponder_elem {
    Flip flip;
    double value;
    int count;
    int level;
    int depth;
    uint_fast8_t mpc_level;
    bool is_endgame_search;
    bool is_complete_search;
    std::string response_board_key;
    Search_result response;
};

std::vector<Ponder_elem> ai_ponder(Board board, bool show_log, thread_id_t thread_id, bool *searching);
std::vector<Ponder_elem> ai_get_values(Board board, bool show_log, uint64_t time_limit, thread_id_t thread_id);
std::pair<int, int> ponder_selfplay(Board board_start, int root_depth, uint_fast8_t root_mpc_level, bool show_log, bool use_multi_thread, bool *searching);
std::vector<Ponder_elem> ai_align_move_levels(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id, int aligned_min_level);
std::vector<Ponder_elem> ai_additional_selfplay(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, double threshold, uint64_t time_limit, thread_id_t thread_id);
Search_result ai_legal_window(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal);
Search_result selfplay_and_analyze_search_result(Board board, int level, bool show_log, thread_id_t thread_id, bool *searching);

inline uint64_t get_this_search_time_limit(uint64_t time_limit, uint64_t elapsed) {
    if (time_limit <= elapsed) {
        return 0;
    }
    return time_limit - elapsed;
}

inline int get_ai_tl_mid_verify_mpc_level(int depth) {
    if (depth < AI_TL_MID_VERIFY_MIN_DEPTH) {
        return MPC_74_LEVEL;
    }
#if IS_GGS_TOURNAMENT
    return depth >= AI_TL_GGS_MID_VERIFY_93_MIN_DEPTH ? MPC_93_LEVEL : MPC_88_LEVEL;
#else
    return depth >= 31 ? MPC_93_LEVEL : MPC_88_LEVEL;
#endif
}

inline int get_ai_tl_early_break_mpc_level(int depth) {
    if (depth < AI_TL_EARLY_BREAK_VERIFY_MIN_DEPTH) {
        return MPC_74_LEVEL;
    }
    return MPC_88_LEVEL;
}

inline uint_fast8_t get_ai_tl_policy_change_verify_mpc_level(int depth, uint_fast8_t mpc_level) {
    if (depth < AI_TL_POLICY_CHANGE_VERIFY_MIN_DEPTH) {
        return mpc_level;
    }
    return std::max<uint_fast8_t>(mpc_level, AI_TL_POLICY_CHANGE_VERIFY_MPC_LEVEL);
}

#if IS_GGS_TOURNAMENT
inline bool ai_tl_ggs_is_defensive_alt_verify_candidate(
    int max_depth,
    int main_depth,
    uint_fast8_t main_mpc_level,
    bool main_is_complete_search,
    bool main_is_end_search,
    uint64_t time_limit,
    bool defensive_alt_verify_done,
    int value,
    int policy,
    uint64_t legal
) {
    const int min_depth =
        max_depth >= 44 && max_depth <= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_EARLY_MAX_N_EMPTY ?
            AI_TL_GGS_DEFENSIVE_ALT_VERIFY_EARLY_MIN_DEPTH :
            AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_DEPTH;
    return
        !defensive_alt_verify_done &&
        !main_is_complete_search &&
        !main_is_end_search &&
        max_depth <= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MAX_N_EMPTY &&
        main_depth >= min_depth &&
        main_mpc_level >= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_MPC_LEVEL &&
        main_mpc_level < MPC_100_LEVEL &&
        time_limit >= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_MAIN_TIME &&
        AI_TL_GGS_DEFENSIVE_ALT_VERIFY_VALUE_MIN <= value &&
        value <= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_VALUE_MAX &&
        is_valid_policy(policy) &&
        (legal & (1ULL << policy)) &&
        (legal & ~(1ULL << policy));
}

inline bool ai_tl_ggs_is_narrow_alt_verify_candidate(
    int max_depth,
    int main_depth,
    uint_fast8_t main_mpc_level,
    bool main_is_complete_search,
    uint64_t time_limit,
    bool narrow_alt_verify_done,
    int value,
    int policy,
    uint64_t legal
) {
    return
        !narrow_alt_verify_done &&
        !main_is_complete_search &&
        AI_TL_GGS_NARROW_ALT_VERIFY_MIN_N_EMPTY <= max_depth &&
        max_depth <= AI_TL_GGS_NARROW_ALT_VERIFY_MAX_N_EMPTY &&
        main_depth >= AI_TL_GGS_NARROW_ALT_VERIFY_MIN_DEPTH &&
        main_mpc_level >= AI_TL_GGS_NARROW_ALT_VERIFY_MIN_MPC_LEVEL &&
        main_mpc_level < MPC_100_LEVEL &&
        time_limit >= AI_TL_GGS_NARROW_ALT_VERIFY_MIN_MAIN_TIME &&
        AI_TL_GGS_NARROW_ALT_VERIFY_VALUE_MIN <= value &&
        value <= AI_TL_GGS_NARROW_ALT_VERIFY_VALUE_MAX &&
        is_valid_policy(policy) &&
        (legal & (1ULL << policy)) &&
        (legal & ~(1ULL << policy));
}

inline bool ai_tl_ggs_can_retry_alt_verify(int last_verify_depth, int main_depth) {
    return last_verify_depth < 0 || main_depth >= last_verify_depth + AI_TL_GGS_ALT_VERIFY_RETRY_DEPTH_GAP;
}
#endif

struct Lazy_smp_worker_result {
    Search_result result;
    int worker_idx;
    uint64_t nodes;
    bool completed_target;

    Lazy_smp_worker_result()
        : worker_idx(0), nodes(0), completed_target(false) {}
};

constexpr int LAZY_SMP_MIN_MID_DEPTH = 23;
constexpr int LAZY_SMP_MIN_END_DEPTH = 30;
#if IS_GGS_TOURNAMENT
    #ifndef GGS_TOURNAMENT_USE_MID_ROOT_LAZY_SMP
        #define GGS_TOURNAMENT_USE_MID_ROOT_LAZY_SMP true
    #endif
#else
    #define GGS_TOURNAMENT_USE_MID_ROOT_LAZY_SMP false
#endif
constexpr bool USE_MID_ROOT_LAZY_SMP = GGS_TOURNAMENT_USE_MID_ROOT_LAZY_SMP;
constexpr int LAZY_SMP_MAX_MID_HELPER_THREADS = THREAD_SIZE_DEFAULT;
constexpr int LAZY_SMP_MAX_END_HELPER_THREADS = 2;
constexpr int LAZY_SMP_END_HELPER_MAX_ROOT_MOVES = 8;
#if IS_GGS_TOURNAMENT
    #ifndef GGS_TOURNAMENT_LAZY_SMP_MIN_PURE_MID_THREADS
        #define GGS_TOURNAMENT_LAZY_SMP_MIN_PURE_MID_THREADS 5
    #endif
    #ifndef GGS_TOURNAMENT_LAZY_SMP_MAX_PURE_MID_DEPTH
        #define GGS_TOURNAMENT_LAZY_SMP_MAX_PURE_MID_DEPTH 32
    #endif
#else
    #define GGS_TOURNAMENT_LAZY_SMP_MIN_PURE_MID_THREADS 12
    #define GGS_TOURNAMENT_LAZY_SMP_MAX_PURE_MID_DEPTH 23
#endif
constexpr int LAZY_SMP_MIN_PURE_MID_THREADS = GGS_TOURNAMENT_LAZY_SMP_MIN_PURE_MID_THREADS;
constexpr int LAZY_SMP_MAX_PURE_MID_DEPTH = GGS_TOURNAMENT_LAZY_SMP_MAX_PURE_MID_DEPTH;

inline int lazy_smp_depth_bias(int worker_idx) {
    if (worker_idx <= 0) {
        return 0;
    }
    return std::min(3, ctz_uint32((uint32_t)worker_idx) + 1);
}

inline uint_fast8_t lazy_smp_worker_mpc(uint_fast8_t mpc_level, int worker_idx, bool is_end_search, bool is_target_search) {
    if (is_end_search || is_target_search || worker_idx <= 0) {
        return mpc_level;
    }
    if ((worker_idx & 3) == 0 && mpc_level < MPC_100_LEVEL) {
        return mpc_level + 1;
    }
    return mpc_level;
}

inline bool lazy_smp_has_target_result(const Lazy_smp_worker_result &worker_result) {
    return worker_result.completed_target && worker_result.result.value != SCORE_UNDEFINED && 0 <= worker_result.result.policy && worker_result.result.policy <= MOVE_PASS;
}

inline int get_root_wipeout_move(Board *board, uint64_t legal) {
    if (legal == 0ULL) {
        return MOVE_UNDEFINED;
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, board, cell);
        if (flip.flip == board->opponent) {
            return (int)cell;
        }
    }
    return MOVE_UNDEFINED;
}

inline Search_result lazy_smp_select_result(const std::vector<Lazy_smp_worker_result> &worker_results, const Search_result &fallback_result) {
    int min_value = SCORE_INF;
    bool has_target = false;
    for (const Lazy_smp_worker_result &worker_result: worker_results) {
        if (lazy_smp_has_target_result(worker_result)) {
            min_value = std::min(min_value, worker_result.result.value);
            has_target = true;
        }
    }
    if (!has_target) {
        return fallback_result;
    }

    int64_t policy_votes[MOVE_PASS + 1] = {};
    for (const Lazy_smp_worker_result &worker_result: worker_results) {
        if (!lazy_smp_has_target_result(worker_result)) {
            continue;
        }
        const Search_result &candidate = worker_result.result;
        policy_votes[candidate.policy] += (int64_t)(candidate.value - min_value + HW2 + 1) * std::max(1, candidate.depth);
    }

    const Lazy_smp_worker_result *best_worker = nullptr;
    for (const Lazy_smp_worker_result &worker_result: worker_results) {
        if (!lazy_smp_has_target_result(worker_result)) {
            continue;
        }
        if (best_worker == nullptr) {
            best_worker = &worker_result;
            continue;
        }
        const Search_result &candidate = worker_result.result;
        const Search_result &best = best_worker->result;
        const int64_t candidate_votes = policy_votes[candidate.policy];
        const int64_t best_votes = policy_votes[best.policy];
        if (
            candidate_votes > best_votes ||
            (candidate_votes == best_votes && candidate.depth > best.depth) ||
            (candidate_votes == best_votes && candidate.depth == best.depth && candidate.probability > best.probability) ||
            (candidate_votes == best_votes && candidate.depth == best.depth && candidate.probability == best.probability && candidate.value > best.value)
        ) {
            best_worker = &worker_result;
        }
    }
    return best_worker == nullptr ? fallback_result : best_worker->result;
}

inline void lazy_smp_advance_depth(int *main_depth, int *main_mpc_level, int depth, uint_fast8_t target_mpc_level, bool is_end_search) {
    if (is_end_search && *main_depth >= depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET) {
        if (*main_depth < depth) {
            *main_depth = depth;
            if (depth <= 27 && target_mpc_level >= MPC_88_LEVEL) {
                *main_mpc_level = MPC_88_LEVEL;
            } else {
                *main_mpc_level = MPC_74_LEVEL;
            }
        } else {
            if (*main_mpc_level < target_mpc_level) {
                if (
                    (*main_mpc_level >= MPC_74_LEVEL && target_mpc_level > MPC_74_LEVEL && depth <= 22) ||
                    (*main_mpc_level >= MPC_88_LEVEL && target_mpc_level > MPC_88_LEVEL && depth <= 25) ||
                    (*main_mpc_level >= MPC_93_LEVEL && target_mpc_level > MPC_93_LEVEL && depth <= 29) ||
                    (*main_mpc_level >= MPC_98_LEVEL && target_mpc_level > MPC_98_LEVEL)
                ) {
                    *main_mpc_level = target_mpc_level;
                } else {
                    ++(*main_mpc_level);
                }
            } else {
                *main_depth = depth + 1;
            }
        }
    } else {
        if (*main_depth <= 15 && *main_depth < depth - 3) {
            *main_depth += 3;
        } else {
            ++(*main_depth);
        }
    }
}

inline Lazy_smp_worker_result lazy_smp_worker(
    Board board,
    int alpha,
    int beta,
    int depth,
    uint_fast8_t mpc_level,
    std::vector<Clog_result> clogs,
    uint64_t use_legal,
    int worker_idx,
    thread_id_t thread_id,
    uint64_t strt,
    bool *searching,
    bool show_log,
    bool use_tree_parallel,
    bool preserve_main_presearch_importance
) {
    Lazy_smp_worker_result worker_result;
    worker_result.worker_idx = worker_idx;
    Search_result *result = &worker_result.result;
    result->value = SCORE_UNDEFINED;

    int main_depth = 1;
    int main_mpc_level = mpc_level;
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    const bool is_end_search = (depth == max_depth);
    if (is_end_search) {
        main_mpc_level = MPC_74_LEVEL;
    }
    const int depth_bias = lazy_smp_depth_bias(worker_idx);

    while (main_depth <= depth && main_mpc_level <= mpc_level && global_searching && *searching) {
        int worker_depth = main_depth;
        bool target_iteration = (main_depth == depth) && (main_mpc_level == mpc_level);
        if (worker_idx > 0 && target_iteration && is_end_search) {
            break;
        }
        if (worker_idx > 0 && !target_iteration) {
            worker_depth = std::min(depth, main_depth + depth_bias);
        }
        bool worker_is_end_search = false;
        if (worker_depth >= max_depth) {
            worker_is_end_search = true;
            worker_depth = max_depth;
        }
        uint_fast8_t worker_mpc_level = lazy_smp_worker_mpc((uint_fast8_t)main_mpc_level, worker_idx, worker_is_end_search, target_iteration);
        bool worker_is_target = (worker_depth == depth) && (worker_mpc_level == mpc_level);

        Search main_search(&board, worker_mpc_level, use_tree_parallel, !worker_is_target);
        main_search.thread_id = thread_id;
        main_search.lazy_smp_worker_idx = use_tree_parallel ? ((worker_idx == 0 && preserve_main_presearch_importance) ? -1 : worker_idx) : 0;
        std::pair<int, int> id_result = first_nega_scout_legal(&main_search, alpha, beta, worker_depth, worker_is_end_search, clogs, use_legal, strt, searching);
        result->nodes += main_search.n_nodes;

        if (*searching) {
            if (result->value != SCORE_UNDEFINED && !worker_is_end_search) {
                double n_value = (0.9 * result->value + 1.1 * id_result.first) / 2.0;
                result->value = round(n_value);
            } else {
                result->value = id_result.first;
            }
            result->policy = id_result.second;
            result->depth = worker_depth;
            result->is_end_search = worker_is_end_search;
            result->probability = SELECTIVITY_PERCENTAGE[worker_mpc_level];
            if (worker_is_target) {
                worker_result.completed_target = true;
            }
        }

        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        if (show_log) {
            if (worker_is_target) {
                std::cerr << "main ";
            } else {
                std::cerr << "pre ";
            }
            if (worker_is_end_search) {
                std::cerr << "end ";
            } else {
                std::cerr << "mid ";
            }
            std::cerr << "depth " << result->depth << "@" << SELECTIVITY_PERCENTAGE[worker_mpc_level] << "%"
                      << " value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second)
                      << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
        }
        if (worker_result.completed_target && worker_idx > 0) {
            break;
        }
        lazy_smp_advance_depth(&main_depth, &main_mpc_level, depth, mpc_level, is_end_search);
    }
    worker_result.nodes = result->nodes;
    return worker_result;
}

void iterative_deepening_search(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, thread_id_t thread_id, Search_result *result, bool *searching) {
    uint64_t strt = tim();
    int max_helper_threads = thread_pool.get_max_thread_size(thread_id);
    if (max_helper_threads == THREAD_SIZE_DEFAULT) {
        max_helper_threads = thread_pool.size();
    }
    const int n_usable_threads = use_multi_thread ? std::min(thread_pool.size(), max_helper_threads) + 1 : 1;
    if (show_log) {
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }

    const int max_depth = HW2 - board.n_discs();
    const bool is_end_search = depth >= max_depth;
#if USE_YBWC_SPLIT_STATISTICS
    if (is_end_search) {
        ybwc_split_stats_reset();
    }
#endif
    const uint64_t root_legal = use_legal == LEGAL_UNDEFINED ? board.get_legal() : use_legal;
    const int wipeout_move = get_root_wipeout_move(&board, root_legal);
    if (wipeout_move != MOVE_UNDEFINED) {
        *result = Search_result();
        result->value = SCORE_MAX;
        result->policy = wipeout_move;
        result->depth = std::min(depth, max_depth);
        result->is_end_search = is_end_search;
        result->probability = SELECTIVITY_PERCENTAGE[mpc_level];
        result->nodes = 1;
        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        return;
    }
    const int root_legal_count = pop_count_ull(root_legal);
    const bool candidate_pure_mid_lazy_smp = USE_MID_ROOT_LAZY_SMP && !is_end_search && depth >= LAZY_SMP_MIN_MID_DEPTH && depth <= LAZY_SMP_MAX_PURE_MID_DEPTH && n_usable_threads >= LAZY_SMP_MIN_PURE_MID_THREADS;
    const int max_lazy_helpers = is_end_search ? LAZY_SMP_MAX_END_HELPER_THREADS : std::min(candidate_pure_mid_lazy_smp ? LAZY_SMP_MAX_MID_HELPER_THREADS : 0, n_usable_threads - 1);
    const int n_lazy_workers = use_multi_thread ? std::min(n_usable_threads, max_lazy_helpers + 1) : 1;
    const bool use_pure_mid_lazy_smp = candidate_pure_mid_lazy_smp && n_lazy_workers >= LAZY_SMP_MIN_PURE_MID_THREADS;
    const bool use_root_lazy_smp = USE_LAZY_SMP && n_lazy_workers >= 2 && (
        use_pure_mid_lazy_smp ||
        (is_end_search && depth >= LAZY_SMP_MIN_END_DEPTH && root_legal_count <= LAZY_SMP_END_HELPER_MAX_ROOT_MOVES)
    );
    const bool use_tree_parallel = use_multi_thread && !use_pure_mid_lazy_smp;

    std::vector<std::future<Lazy_smp_worker_result>> helper_tasks;
    bool helper_searching = true;
    if (use_root_lazy_smp) {
        for (int worker_idx = 1; worker_idx < n_lazy_workers && global_searching && *searching; ++worker_idx) {
            bool pushed = false;
            helper_tasks.emplace_back(thread_pool.push(thread_id, &pushed, std::bind(
                lazy_smp_worker, board, alpha, beta, depth, mpc_level, clogs, use_legal, worker_idx, thread_id, strt, &helper_searching, false, false, false
            )));
            if (!pushed) {
                helper_tasks.pop_back();
                break;
            }
        }
    }

    Lazy_smp_worker_result main_worker_result = lazy_smp_worker(board, alpha, beta, depth, mpc_level, clogs, use_legal, 0, thread_id, strt, searching, show_log, use_tree_parallel, use_root_lazy_smp && is_end_search);
    helper_searching = false;
    std::vector<Lazy_smp_worker_result> worker_results;
    worker_results.emplace_back(main_worker_result);
    *result = main_worker_result.result;
    uint64_t helper_nodes = 0;
    int completed_helpers = 0;
    for (std::future<Lazy_smp_worker_result> &task: helper_tasks) {
        Lazy_smp_worker_result helper_result = task.get();
        helper_nodes += helper_result.nodes;
        completed_helpers += helper_result.completed_target;
        worker_results.emplace_back(helper_result);
        if (!main_worker_result.completed_target && helper_result.completed_target) {
            *result = helper_result.result;
        }
    }
    if (use_root_lazy_smp) {
        *result = lazy_smp_select_result(worker_results, *result);
    }
    result->nodes += helper_nodes;
    result->time = tim() - strt;
    result->nps = calc_nps(result->nodes, result->time);
    if (show_log && use_root_lazy_smp) {
        std::cerr << "lazy smp workers " << (helper_tasks.size() + 1)
                  << " completed_helpers " << completed_helpers
                  << " helper_n_nodes " << helper_nodes
                  << " whole_n_nodes " << result->nodes
                  << " whole_time " << result->time
                  << " whole_NPS " << result->nps << std::endl;
    }
#if USE_YBWC_SPLIT_STATISTICS
    if (is_end_search) {
        ybwc_split_stats_print();
    }
#endif
}

void iterative_deepening_search_time_limit(Board board, int alpha, int beta, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, thread_id_t thread_id, Search_result *result, uint64_t time_limit, bool *searching) {
    const int n_usable_threads = thread_pool.get_max_thread_size(thread_id);
    uint64_t strt = tim();
    result->value = SCORE_UNDEFINED;
    uint64_t fallback_legal = use_legal == LEGAL_UNDEFINED ? board.get_legal() : use_legal;
    if (fallback_legal) {
        uint64_t preferred = fallback_legal & 0x8100000000000081ULL;
        uint64_t candidates = preferred ? preferred : fallback_legal;
        result->policy = first_bit(&candidates);
        result->value = mid_evaluate(&board);
        result->depth = 0;
        result->is_end_search = false;
        result->probability = 0;
    }
    int main_depth = 1;
    int main_mpc_level = MPC_100_LEVEL;
    const int max_depth = HW2 - board.n_discs();
    if (show_log) {
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }
    int before_raw_value = -100;
    bool policy_changed_before = true;
#if IS_GGS_TOURNAMENT
    int defensive_alt_verify_depth = -1;
    int narrow_alt_verify_depth = -1;
#endif
    while (global_searching && (*searching) && ((tim() - strt < time_limit) || main_depth <= 1)) {
        bool main_is_end_search = false;
        if (main_depth >= max_depth) {
            main_is_end_search = true;
            main_depth = max_depth;
        }
        bool main_is_complete_search = main_is_end_search && main_mpc_level == MPC_100_LEVEL;
        if (show_log) {
            if (main_is_end_search) {
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
            std::cerr << "depth " << main_depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "% " << std::flush;
        }
        Search main_search(&board, main_mpc_level, use_multi_thread, false);
        main_search.thread_id = thread_id;
        std::pair<int, int> id_result;
        bool search_success = false;
        bool main_searching = true;
        uint64_t time_limit_this_search = get_this_search_time_limit(time_limit, tim() - strt);
#if IS_GGS_TOURNAMENT
        uint64_t reserved_for_defensive_alt = 0ULL;
        if (
            ai_tl_ggs_is_defensive_alt_verify_candidate(
                max_depth,
                main_depth,
                (uint_fast8_t)main_mpc_level,
                main_is_complete_search,
                main_is_end_search,
                time_limit,
                !ai_tl_ggs_can_retry_alt_verify(defensive_alt_verify_depth, main_depth),
                result->value,
                result->policy,
                fallback_legal
            ) &&
            time_limit_this_search > AI_TL_GGS_DEFENSIVE_ALT_VERIFY_RESERVE_TIME + AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_TIME_LEFT
        ) {
            reserved_for_defensive_alt = std::min<uint64_t>(
                AI_TL_GGS_DEFENSIVE_ALT_VERIFY_RESERVE_TIME,
                time_limit_this_search - AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_TIME_LEFT
            );
            time_limit_this_search -= reserved_for_defensive_alt;
            if (show_log) {
                std::cerr << "defensive-alt reserve " << reserved_for_defensive_alt << " ms ";
            }
        }
#endif
        std::future<std::pair<int, int>> f = std::async(std::launch::async, first_nega_scout_legal, &main_search, alpha, beta, main_depth, main_is_end_search, clogs, use_legal, strt, &main_searching);
        if (f.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
            id_result = f.get();
            search_success = true;
        } else {
            main_searching = false;
            try {
                f.get();
            } catch (const std::exception &e) {
            }
            if (show_log) {
                std::cerr << "terminated " << tim() - strt << " ms" << std::endl;
            }
        }
        result->nodes += main_search.n_nodes;
        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        if (search_success) {
            Search_result previous_result = *result;
            bool verify_timeout = false;
            bool keep_previous_on_verify_timeout = false;
            bool skip_policy_change_as_bad_stable = false;
            std::string verify_log;
#if IS_GGS_TOURNAMENT
            if (
                !main_is_complete_search &&
                !main_is_end_search &&
                AI_TL_GGS_BAD_STABLE_SWITCH_MIN_N_EMPTY <= max_depth &&
                max_depth <= AI_TL_GGS_BAD_STABLE_SWITCH_MAX_N_EMPTY &&
                main_depth >= AI_TL_GGS_BAD_STABLE_SWITCH_MIN_DEPTH &&
                main_mpc_level == MPC_74_LEVEL &&
                is_valid_policy(previous_result.policy) &&
                is_valid_policy(id_result.second) &&
                previous_result.policy != id_result.second &&
                (use_legal & (1ULL << previous_result.policy)) &&
                (use_legal & (1ULL << id_result.second)) &&
                AI_TL_GGS_BAD_STABLE_SWITCH_VALUE_MIN <= previous_result.value &&
                previous_result.value <= AI_TL_GGS_BAD_STABLE_SWITCH_VALUE_MAX &&
                id_result.first <= previous_result.value + AI_TL_GGS_BAD_STABLE_SWITCH_MAX_GAIN
            ) {
                std::ostringstream ss;
                ss << " bad-stable-skip-switch "
                   << idx_to_coord(previous_result.policy) << "=" << previous_result.value << " "
                   << idx_to_coord(id_result.second) << "=" << id_result.first;
                verify_log = ss.str();
                id_result = std::make_pair(previous_result.value, previous_result.policy);
                skip_policy_change_as_bad_stable = true;
            }
            if (
                !skip_policy_change_as_bad_stable &&
                !main_is_complete_search &&
                !main_is_end_search &&
                AI_TL_GGS_HIGH_CONF_STICK_MIN_N_EMPTY <= max_depth &&
                max_depth <= AI_TL_GGS_HIGH_CONF_STICK_MAX_N_EMPTY &&
                main_depth >= AI_TL_GGS_HIGH_CONF_STICK_MIN_DEPTH &&
                main_mpc_level == MPC_74_LEVEL &&
                previous_result.probability >= AI_TL_GGS_HIGH_CONF_STICK_MIN_PREV_PROBABILITY &&
                previous_result.depth + 1 >= main_depth &&
                is_valid_policy(previous_result.policy) &&
                is_valid_policy(id_result.second) &&
                previous_result.policy != id_result.second &&
                (use_legal & (1ULL << previous_result.policy)) &&
                (use_legal & (1ULL << id_result.second)) &&
                AI_TL_GGS_HIGH_CONF_STICK_VALUE_MIN <= previous_result.value &&
                previous_result.value != SCORE_UNDEFINED &&
                id_result.first <= previous_result.value + AI_TL_GGS_HIGH_CONF_STICK_MAX_GAIN
            ) {
                std::ostringstream ss;
                ss << " high-conf-stick "
                   << idx_to_coord(previous_result.policy) << "=" << previous_result.value
                   << "@" << previous_result.depth << "/" << previous_result.probability << "% "
                   << idx_to_coord(id_result.second) << "=" << id_result.first
                   << "@" << main_depth << "/" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%";
                verify_log = ss.str();
                id_result = std::make_pair(previous_result.value, previous_result.policy);
                skip_policy_change_as_bad_stable = true;
            }
#endif
            if (
                !skip_policy_change_as_bad_stable &&
                !main_is_complete_search &&
                main_depth >= AI_TL_POLICY_CHANGE_VERIFY_MIN_DEPTH &&
                main_mpc_level < MPC_100_LEVEL &&
                is_valid_policy(result->policy) &&
                is_valid_policy(id_result.second) &&
                result->policy != id_result.second &&
                (use_legal & (1ULL << result->policy)) &&
                (use_legal & (1ULL << id_result.second))
            ) {
                uint_fast8_t verify_mpc_level = get_ai_tl_policy_change_verify_mpc_level(main_depth, main_mpc_level);
                int previous_policy = result->policy;
                int new_policy = id_result.second;
                int previous_value = SCORE_UNDEFINED;
                int new_value = SCORE_UNDEFINED;
                Search verify_search(&board, verify_mpc_level, use_multi_thread, false);
                verify_search.thread_id = thread_id;
                bool verify_searching = true;
#if IS_GGS_TOURNAMENT
                uint64_t verify_budget = std::min<uint64_t>(
                    get_this_search_time_limit(time_limit, tim() - strt),
                    std::min<uint64_t>(
                        AI_TL_GGS_POLICY_CHANGE_VERIFY_MAX_TIME,
                        (uint64_t)((double)time_limit * AI_TL_GGS_POLICY_CHANGE_VERIFY_TIME_COE)
                    )
                );
                uint64_t verify_strt = tim();
                uint64_t time_limit_verify = get_this_search_time_limit(verify_budget, tim() - verify_strt);
#else
                uint64_t time_limit_verify = get_this_search_time_limit(time_limit, tim() - strt);
#endif
                std::future<std::pair<int, int>> previous_verify_f = std::async(std::launch::async, first_nega_scout_legal, &verify_search, alpha, beta, main_depth, main_is_end_search, clogs, 1ULL << previous_policy, strt, &verify_searching);
                if (previous_verify_f.wait_for(std::chrono::milliseconds(time_limit_verify)) == std::future_status::ready) {
                    std::pair<int, int> previous_verify_result = previous_verify_f.get();
                    previous_value = previous_verify_result.first;
                } else {
                    verify_searching = false;
                    try {
                        previous_verify_f.get();
                    } catch (const std::exception &e) {
                    }
                    verify_timeout = true;
                }
                if (!verify_timeout) {
#if IS_GGS_TOURNAMENT
                    time_limit_verify = get_this_search_time_limit(verify_budget, tim() - verify_strt);
#else
                    time_limit_verify = get_this_search_time_limit(time_limit, tim() - strt);
#endif
                    std::future<std::pair<int, int>> new_verify_f = std::async(std::launch::async, first_nega_scout_legal, &verify_search, alpha, beta, main_depth, main_is_end_search, clogs, 1ULL << new_policy, strt, &verify_searching);
                    if (new_verify_f.wait_for(std::chrono::milliseconds(time_limit_verify)) == std::future_status::ready) {
                        std::pair<int, int> new_verify_result = new_verify_f.get();
                        new_value = new_verify_result.first;
                    } else {
                        verify_searching = false;
                        try {
                            new_verify_f.get();
                        } catch (const std::exception &e) {
                        }
                        verify_timeout = true;
                    }
                }
                if (verify_timeout) {
                    if (show_log) {
                        std::cerr << "policy-change verify@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% terminated " << tim() - strt << " ms" << std::endl;
                    }
                } else {
                    std::ostringstream ss;
                    ss << " verify@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% " << idx_to_coord(previous_policy) << "=" << previous_value << " " << idx_to_coord(new_policy) << "=" << new_value;
                    if (new_value > previous_value) {
                        id_result = std::make_pair(new_value, new_policy);
                        ss << "->" << idx_to_coord(new_policy);
                    } else {
                        id_result = std::make_pair(previous_value, previous_policy);
                        ss << "->" << idx_to_coord(previous_policy);
                    }
                    verify_log = ss.str();
                }
                result->nodes += verify_search.n_nodes;
                result->time = tim() - strt;
                result->nps = calc_nps(result->nodes, result->time);
            }
            if (verify_timeout) {
#if IS_GGS_TOURNAMENT
                const int verify_timeout_keep_margin =
                    previous_result.value >= 0 ? AI_TL_GGS_VERIFY_TIMEOUT_POSITIVE_KEEP_MARGIN : AI_TL_VERIFY_TIMEOUT_KEEP_MARGIN;
                if (
                    is_valid_policy(previous_result.policy) &&
                    (use_legal & (1ULL << previous_result.policy)) &&
                    previous_result.value != SCORE_UNDEFINED &&
                    id_result.first <= previous_result.value + verify_timeout_keep_margin
                ) {
                    const uint64_t accumulated_nodes = result->nodes;
                    const uint64_t elapsed = tim() - strt;
                    const uint64_t nps = calc_nps(accumulated_nodes, elapsed);
                    *result = previous_result;
                    result->nodes = accumulated_nodes;
                    result->time = elapsed;
                    result->nps = nps;
                    keep_previous_on_verify_timeout = true;
                    verify_log = " verify-timeout-keep-previous";
                } else {
                    verify_log = " verify-timeout-use-main";
                }
#else
                break;
#endif
            }
#if IS_GGS_TOURNAMENT
            if (
                !verify_timeout &&
                !keep_previous_on_verify_timeout &&
                ai_tl_ggs_is_defensive_alt_verify_candidate(
                    max_depth,
                    main_depth,
                    (uint_fast8_t)main_mpc_level,
                    main_is_complete_search,
                    main_is_end_search,
                    time_limit,
                    !ai_tl_ggs_can_retry_alt_verify(defensive_alt_verify_depth, main_depth),
                    id_result.first,
                    id_result.second,
                    fallback_legal
                )
            ) {
                const uint64_t alternative_legal = fallback_legal & ~(1ULL << id_result.second);
                const uint64_t time_left_for_verify = get_this_search_time_limit(time_limit, tim() - strt);
                if (alternative_legal != 0ULL && time_left_for_verify >= AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_TIME_LEFT) {
                    uint64_t defensive_verify_budget = std::min<uint64_t>(
                        time_left_for_verify,
                        std::min<uint64_t>(
                            AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MAX_TIME,
                            (uint64_t)((double)time_limit * AI_TL_GGS_DEFENSIVE_ALT_VERIFY_TIME_COE)
                        )
                    );
                    defensive_verify_budget = std::max<uint64_t>(defensive_verify_budget, AI_TL_GGS_DEFENSIVE_ALT_VERIFY_MIN_TIME_LEFT);
                    const int current_policy = id_result.second;
                    const uint_fast8_t verify_mpc_level = std::max<uint_fast8_t>(
                        main_mpc_level,
                        AI_TL_GGS_DEFENSIVE_ALT_VERIFY_SEARCH_MPC_LEVEL
                    );
                    defensive_alt_verify_depth = main_depth;
                    uint64_t defensive_verify_strt = tim();
                    int current_value = id_result.first;
                    int alternative_value = SCORE_UNDEFINED;
                    int alternative_policy = MOVE_NOMOVE;
                    bool defensive_verify_timeout = false;
                    Search defensive_verify_search(&board, verify_mpc_level, use_multi_thread, false);
                    defensive_verify_search.thread_id = thread_id;
                    bool defensive_verify_searching = true;
                    uint64_t time_limit_defensive_verify = get_this_search_time_limit(defensive_verify_budget, tim() - defensive_verify_strt);
                    std::future<std::pair<int, int>> alternative_verify_f = std::async(std::launch::async, first_nega_scout_legal, &defensive_verify_search, alpha, beta, main_depth, main_is_end_search, clogs, alternative_legal, strt, &defensive_verify_searching);
                    if (alternative_verify_f.wait_for(std::chrono::milliseconds(time_limit_defensive_verify)) == std::future_status::ready) {
                        std::pair<int, int> alternative_verify_result = alternative_verify_f.get();
                        alternative_value = alternative_verify_result.first;
                        alternative_policy = alternative_verify_result.second;
                    } else {
                        defensive_verify_searching = false;
                        try {
                            alternative_verify_f.get();
                        } catch (const std::exception &e) {
                        }
                        defensive_verify_timeout = true;
                    }
                    if (
                        !defensive_verify_timeout &&
                        is_valid_policy(alternative_policy) &&
                        alternative_value > id_result.first
                    ) {
                        time_limit_defensive_verify = get_this_search_time_limit(defensive_verify_budget, tim() - defensive_verify_strt);
                        std::future<std::pair<int, int>> current_verify_f = std::async(std::launch::async, first_nega_scout_legal, &defensive_verify_search, alpha, beta, main_depth, main_is_end_search, clogs, 1ULL << current_policy, strt, &defensive_verify_searching);
                        if (current_verify_f.wait_for(std::chrono::milliseconds(time_limit_defensive_verify)) == std::future_status::ready) {
                            std::pair<int, int> current_verify_result = current_verify_f.get();
                            current_value = current_verify_result.first;
                        } else {
                            defensive_verify_searching = false;
                            try {
                                current_verify_f.get();
                            } catch (const std::exception &e) {
                            }
                            defensive_verify_timeout = true;
                        }
                    }
                    if (defensive_verify_timeout) {
                        if (show_log) {
                            std::ostringstream ss;
                            ss << " defensive-alt@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% terminated";
                            verify_log += ss.str();
                        }
                    } else {
                        std::ostringstream ss;
                        ss << " defensive-alt@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% "
                           << idx_to_coord(current_policy) << "=" << current_value << " "
                           << idx_to_coord(alternative_policy) << "=" << alternative_value;
                        if (is_valid_policy(alternative_policy) && alternative_value > current_value) {
                            id_result = std::make_pair(alternative_value, alternative_policy);
                            ss << "->" << idx_to_coord(alternative_policy);
                        } else {
                            id_result = std::make_pair(current_value, current_policy);
                            ss << "->" << idx_to_coord(current_policy);
                        }
                        verify_log += ss.str();
                    }
                    result->nodes += defensive_verify_search.n_nodes;
                    result->time = tim() - strt;
                    result->nps = calc_nps(result->nodes, result->time);
                }
            }
#endif
#if IS_GGS_TOURNAMENT
            if (
                !verify_timeout &&
                !keep_previous_on_verify_timeout &&
                ai_tl_ggs_is_narrow_alt_verify_candidate(
                    max_depth,
                    main_depth,
                    (uint_fast8_t)main_mpc_level,
                    main_is_complete_search,
                    time_limit,
                    !ai_tl_ggs_can_retry_alt_verify(narrow_alt_verify_depth, main_depth),
                    id_result.first,
                    id_result.second,
                    fallback_legal
                )
            ) {
                const uint64_t alternative_legal = fallback_legal & ~(1ULL << id_result.second);
                const uint64_t time_left_for_verify = get_this_search_time_limit(time_limit, tim() - strt);
                if (alternative_legal != 0ULL && time_left_for_verify >= AI_TL_GGS_NARROW_ALT_VERIFY_MIN_TIME_LEFT) {
                    uint64_t narrow_verify_budget = std::min<uint64_t>(
                        time_left_for_verify,
                        std::min<uint64_t>(
                            AI_TL_GGS_NARROW_ALT_VERIFY_MAX_TIME,
                            (uint64_t)((double)time_limit * AI_TL_GGS_NARROW_ALT_VERIFY_TIME_COE)
                        )
                    );
                    narrow_verify_budget = std::max<uint64_t>(narrow_verify_budget, AI_TL_GGS_NARROW_ALT_VERIFY_MIN_TIME_LEFT);
                    const int current_policy = id_result.second;
                    const uint_fast8_t verify_mpc_level = std::max<uint_fast8_t>(
                        main_mpc_level,
                        AI_TL_GGS_NARROW_ALT_VERIFY_SEARCH_MPC_LEVEL
                    );
                    narrow_alt_verify_depth = main_depth;
                    uint64_t narrow_verify_strt = tim();
                    int current_value = id_result.first;
                    int alternative_value = SCORE_UNDEFINED;
                    int alternative_policy = MOVE_NOMOVE;
                    bool narrow_verify_timeout = false;
                    Search narrow_verify_search(&board, verify_mpc_level, use_multi_thread, false);
                    narrow_verify_search.thread_id = thread_id;
                    bool narrow_verify_searching = true;
                    uint64_t time_limit_narrow_verify = get_this_search_time_limit(narrow_verify_budget, tim() - narrow_verify_strt);
                    std::future<std::pair<int, int>> alternative_verify_f = std::async(std::launch::async, first_nega_scout_legal, &narrow_verify_search, alpha, beta, main_depth, main_is_end_search, clogs, alternative_legal, strt, &narrow_verify_searching);
                    if (alternative_verify_f.wait_for(std::chrono::milliseconds(time_limit_narrow_verify)) == std::future_status::ready) {
                        std::pair<int, int> alternative_verify_result = alternative_verify_f.get();
                        alternative_value = alternative_verify_result.first;
                        alternative_policy = alternative_verify_result.second;
                    } else {
                        narrow_verify_searching = false;
                        try {
                            std::pair<int, int> discarded_result = alternative_verify_f.get();
                            (void)discarded_result;
                        } catch (const std::exception &e) {
                        }
                        narrow_verify_timeout = true;
                    }
                    if (
                        !narrow_verify_timeout &&
                        is_valid_policy(alternative_policy) &&
                        alternative_value > id_result.first
                    ) {
                        time_limit_narrow_verify = get_this_search_time_limit(narrow_verify_budget, tim() - narrow_verify_strt);
                        std::future<std::pair<int, int>> current_verify_f = std::async(std::launch::async, first_nega_scout_legal, &narrow_verify_search, alpha, beta, main_depth, main_is_end_search, clogs, 1ULL << current_policy, strt, &narrow_verify_searching);
                        if (current_verify_f.wait_for(std::chrono::milliseconds(time_limit_narrow_verify)) == std::future_status::ready) {
                            std::pair<int, int> current_verify_result = current_verify_f.get();
                            current_value = current_verify_result.first;
                        } else {
                            narrow_verify_searching = false;
                            try {
                                std::pair<int, int> discarded_result = current_verify_f.get();
                                (void)discarded_result;
                            } catch (const std::exception &e) {
                            }
                            narrow_verify_timeout = true;
                        }
                    }
                    if (narrow_verify_timeout) {
                        if (show_log) {
                            std::ostringstream ss;
                            ss << " narrow-alt@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% terminated";
                            verify_log += ss.str();
                        }
                    } else {
                        std::ostringstream ss;
                        ss << " narrow-alt@" << SELECTIVITY_PERCENTAGE[verify_mpc_level] << "% "
                           << idx_to_coord(current_policy) << "=" << current_value << " "
                           << idx_to_coord(alternative_policy) << "=" << alternative_value;
                        if (is_valid_policy(alternative_policy) && alternative_value > current_value) {
                            id_result = std::make_pair(alternative_value, alternative_policy);
                            ss << "->" << idx_to_coord(alternative_policy);
                        } else {
                            id_result = std::make_pair(current_value, current_policy);
                            ss << "->" << idx_to_coord(current_policy);
                        }
                        verify_log += ss.str();
                    }
                    result->nodes += narrow_verify_search.n_nodes;
                    result->time = tim() - strt;
                    result->nps = calc_nps(result->nodes, result->time);
                }
            }
#endif
            bool policy_changed = false;
            if (!keep_previous_on_verify_timeout) {
                if (result->value != SCORE_UNDEFINED && !main_is_end_search) {
                    double n_value = (0.9 * result->value + 1.1 * id_result.first) / 2.0;
                    result->value = round(n_value);
                } else{
                    result->value = id_result.first;
                }
                policy_changed = result->policy != id_result.second;
                result->policy = id_result.second;
                result->depth = main_depth;
                result->is_end_search = main_is_end_search;
                result->probability = SELECTIVITY_PERCENTAGE[main_mpc_level];
            }
            if (show_log) {
                std::cerr << "value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(result->policy) << verify_log << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
            }
            uint64_t legal_without_bestmove = use_legal ^ (1ULL << result->policy);
            if (
                AI_TL_USE_EARLY_BREAK &&
                (!main_is_end_search && main_depth >= 29 && main_depth <= 30) &&
                max_depth >= 44 &&
                !policy_changed &&
                !policy_changed_before &&
                main_mpc_level == MPC_74_LEVEL &&
                legal_without_bestmove != 0
            ) {
                int nws_alpha = result->value - AI_TL_EARLY_BREAK_THRESHOLD;
                if (nws_alpha >= -SCORE_MAX) {
                    uint_fast8_t early_break_mpc_level = std::max<uint_fast8_t>(main_mpc_level, (uint_fast8_t)get_ai_tl_early_break_mpc_level(main_depth));
                    Search nws_search(&board, early_break_mpc_level, use_multi_thread, false);
                    nws_search.thread_id = thread_id;
                    bool nws_searching = true;
                    if (show_log) {
                        std::cerr << "trying early break@" << SELECTIVITY_PERCENTAGE[early_break_mpc_level] << "% [" << nws_alpha << ", " << nws_alpha + 1 << "] ";
                    }
                    uint64_t time_limit_nws = get_this_search_time_limit(time_limit, tim() - strt);
                    std::future<std::pair<int, int>> nws_f = std::async(std::launch::async, first_nega_scout_legal, &nws_search, nws_alpha, nws_alpha + 1, main_depth, main_is_end_search, clogs, legal_without_bestmove, strt, &nws_searching);
                    int nws_value = SCORE_INF;
                    int nws_move = MOVE_NOMOVE;
                    bool nws_success = false;
                    if (nws_f.wait_for(std::chrono::milliseconds(time_limit_nws)) == std::future_status::ready) {
                        std::pair<int, int> nws_result = nws_f.get();
                        nws_value = nws_result.first;
                        nws_move = nws_result.second;
                        nws_success = true;
                    } else {
                        nws_searching = false;
                        try {
                            nws_f.get();
                        } catch (const std::exception &e) {
                        }
                        if (show_log) {
                            std::cerr << "terminate early cut nws by time limit " << tim() - strt << " ms" << std::endl;
                        }
                    }
                    result->nodes += nws_search.n_nodes;
                    result->time = tim() - strt;
                    result->nps = calc_nps(result->nodes, result->time);
                    if (nws_success) {
                        if (nws_value <= nws_alpha) {
                            if (show_log) {
                                std::cerr << "SUCCEEDED second best " << idx_to_coord(nws_move) << " value <= " << nws_value << " time " << tim() - strt << std::endl;
                            }
                            break;
                        } else if (nws_searching) {
                            if (show_log) {
                                std::cerr << "FAILED second best " << idx_to_coord(nws_move) << " value >= " << nws_value << " time " << tim() - strt << std::endl;
                            }
                        }
                    }
                }
            }
            // check second best move
            /*
            if (!main_is_complete_search && main_depth >= 30 && legal_without_bestmove != 0) {
                Search second_search(&board, main_mpc_level, use_multi_thread, false);
                std::pair<int, int> second_id_result;
                bool second_search_success = false;
                bool second_searching = true;
                uint64_t time_limit_second_search = get_this_search_time_limit(time_limit, tim() - strt);
                std::future<std::pair<int, int>> second_f = std::async(std::launch::async, first_nega_scout_legal, &second_search, alpha, beta, main_depth, main_is_end_search, clogs, legal_without_bestmove, strt, &second_searching);
                if (second_f.wait_for(std::chrono::milliseconds(time_limit_second_search)) == std::future_status::ready) {
                    std::pair<int, int> nws_result = second_f.get();
                    second_search_success = true;
                } else {
                    second_searching = false;
                    second_f.get();
                    if (show_log) {
                        std::cerr << "terminate second search by time limit " << tim() - strt << " ms" << std::endl;
                    }
                }
            }
            */
            before_raw_value = id_result.first;
            policy_changed_before = policy_changed;
        }
        //if (main_depth > 10 && pop_count_ull(board.get_legal()) == 1) { // not use_legal
        //    break; // there is only 1 move
        //}
        if (main_depth < max_depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) { // next: midgame search
            if (main_depth <= 15 && main_depth < max_depth - 3) {
                main_depth += 3;
                if (main_depth > 13 && main_mpc_level == MPC_100_LEVEL) {
                    main_mpc_level = MPC_74_LEVEL;
                }
            } else {
                if (main_depth + 1 < 23) {
                    ++main_depth;
                    if (main_depth > 13 && main_mpc_level == MPC_100_LEVEL) {
                        main_mpc_level = MPC_74_LEVEL;
                    }
                } else {
                    int verify_mpc_level = get_ai_tl_mid_verify_mpc_level(main_depth);
                    if (main_mpc_level < verify_mpc_level) {
                        ++main_mpc_level;
                    } else {
                        ++main_depth;
                        main_mpc_level = MPC_74_LEVEL;
                    }
                }
            }
        } else { // next: endgame search
//#if IS_GGS_TOURNAMENT
            // if (max_depth >= 43) {
            //     std::cerr << "no endgame search here with " << max_depth << " empties" << std::endl;
            //     break;
            // } else
//#endif
            if (main_depth < max_depth) {
                main_depth = max_depth;
                main_mpc_level = AI_TL_ENDGAME_INITIAL_MPC_LEVEL;
            } else {
                if (main_mpc_level < MPC_100_LEVEL) {
                    ++main_mpc_level;
                } else {
                    // if (show_log) {
                    //     std::cerr << "completely searched" << std::endl;
                    // }
                    break;
                }
            }
        }
    }
    if (show_log && result->is_end_search && result->probability == 100) {
        std::cerr << "completely searched" << std::endl;
    }
}


/*
    @brief Get a result of a search

    Firstly, if using MPC, execute clog search for finding special endgame.
    Then do some pre-search, and main search.

    @param board                board to solve
    @param depth                depth to search
    @param mpc_level            MPC level
    @param show_log             show log?
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/
inline Search_result tree_search_legal(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, thread_id_t thread_id, bool *searching) {
    //thread_pool.tell_start_using();
    Search_result res;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    bool use_time_limit = (time_limit != TIME_LIMIT_INF);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL || use_time_limit) {
        uint64_t strt = tim();
        int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
        if (use_time_limit) {
            clog_depth = CLOG_SEARCH_MAX_DEPTH;
        }
        clogs = first_clog_search(board, &clog_nodes, clog_depth, use_legal, searching);
        clog_time = tim() - strt;
        if (show_log) {
            std::cerr << "clog search depth " << clog_depth << " time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i) {
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
        res.clog_nodes = clog_nodes;
        res.clog_time = clog_time;
    }
    if (use_legal) {
        uint64_t time_limit_proc = TIME_LIMIT_INF;
        if (use_time_limit) {
            if (time_limit > clog_time) {
                time_limit_proc = time_limit - clog_time;
            } else {
                time_limit_proc = 1;
            }
        }
        if (use_time_limit) {
            /*
            if (HW2 - board.n_discs() >= 30) {
                uint64_t strt_selfplay = tim();
                uint64_t selfplay_time = time_limit_proc * 0.3;
                time_management_selfplay(board, show_log, use_legal, selfplay_time);
                time_limit_proc -= tim() - strt_selfplay;
            }
            */
            iterative_deepening_search_time_limit(board, alpha, beta, show_log, clogs, use_legal, use_multi_thread, thread_id, &res, time_limit_proc, searching);
        } else {
            iterative_deepening_search(board, alpha, beta, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread, thread_id, &res, searching);
        }
    }
    //thread_pool.tell_finish_using();
    //thread_pool.reset_unavailable();
    //delete_tt(&board, 6);
    return res;
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param alpha                search window (alpha)
    @param beta                 search window (beta)
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai_common(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool use_specified_move_book, uint64_t time_limit, thread_id_t thread_id, bool *searching) {
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL) {
        board.pass();
        if (board.get_legal() == 0ULL) {
            res.level = MAX_LEVEL;
            res.policy = 64;
            res.value = -board.score_player();
            res.depth = 0;
            res.nps = 0;
            res.is_end_search = true;
            res.probability = 100;
            return res;
        } else{
            if (show_log) {
                std::cerr << "pass found in ai_common!" << std::endl;
            }
            value_sign = -1;
        }
    }
    Book_value book_result;
    if (use_specified_move_book) {
        book_result = book.get_specified_best_move(&board, use_legal);
    } else{
        book_result = book.get_random(&board, book_acc_level, use_legal);
    }
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    if (show_log && time_limit == TIME_LIMIT_INF) {
        std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
    }
    if (is_valid_policy(book_result.policy) && use_book) {
        if (show_log) {
            std::cerr << "book found " << value_sign * book_result.value << " " << idx_to_coord(book_result.policy) << std::endl;
        }
        if (book_acc_level == 0) { // accurate book level
            std::vector<Book_value> book_moves = book.get_all_moves_with_value(&board);
            for (const Book_value &move: book_moves) {
                use_legal &= ~(1ULL << move.policy);
            }
            if (use_legal != 0) { // there is moves out of book
                if (show_log) {
                    std::cerr << "there are " << pop_count_ull(use_legal) << " moves out of book" << std::endl;
                }
                int n_alpha = book_result.value;
                Search_result additional_res = tree_search_legal(board, n_alpha, n_alpha + 1, depth, mpc_level, show_log, use_legal, use_multi_thread, time_limit, thread_id, searching);
                if (additional_res.value <= n_alpha) { // no better move found in book
                    res.level = LEVEL_TYPE_BOOK;
                    res.policy = book_result.policy;
                    res.value = value_sign * book_result.value;
                    res.depth = SEARCH_BOOK;
                    res.clog_nodes = additional_res.clog_nodes;
                    res.clog_time = additional_res.clog_time;
                    res.nodes = additional_res.nodes;
                    res.depth = additional_res.depth;
                    res.time = additional_res.time;
                    res.nps = additional_res.nps;
                    res.is_end_search = false;
                    res.probability = 100;
                } else {
                    if (show_log) {
                        std::cerr << "there are better move out of book" << std::endl;
                    }
                    res = tree_search_legal(board, n_alpha, beta, depth, mpc_level, show_log, use_legal, use_multi_thread, time_limit, thread_id, searching);
                    res.time += additional_res.time;
                    res.nodes += additional_res.nodes;
                    res.clog_nodes += additional_res.clog_nodes;
                    res.clog_time += additional_res.clog_time;
                    res.nps = calc_nps(res.nodes, res.time);
                }
            } else {
                if (show_log) {
                    std::cerr << "all moves are in book" << std::endl;
                }
                res.level = LEVEL_TYPE_BOOK;
                res.policy = book_result.policy;
                res.value = value_sign * book_result.value;
                res.depth = SEARCH_BOOK;
                res.nps = 0;
                res.is_end_search = false;
                res.probability = 100;
            }
        } else {
            // book accuracy != 0
            res.level = LEVEL_TYPE_BOOK;
            res.policy = book_result.policy;
            res.value = value_sign * book_result.value;
            res.depth = SEARCH_BOOK;
            res.nps = 0;
            res.is_end_search = false;
            res.probability = 100;
        }
    } else { // no move in book
        res = tree_search_legal(board, alpha, beta, depth, mpc_level, show_log, use_legal, use_multi_thread, time_limit, thread_id, searching);
        res.level = level;
    }
    res.value *= value_sign;
    return res;
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    // std::cerr << "ai " << (HW2 - board.n_discs()) << " empties level " << level << std::endl;
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_searching_thread_id(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, thread_id_t thread_id, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, thread_id, searching);
}

Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_legal_window(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_legal_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_legal_searching_thread_id(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, thread_id_t thread_id, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, thread_id, searching);
}

Search_result ai_window(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    bool searching = true;
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_window_legal(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_window_legal_searching(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool *searching) {
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_window_searching(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, bool *searching) {
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_specified(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

std::vector<Search_result> ai_best_n_moves_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_moves, bool *searching) {
    Search_result best = ai_searching(board, level, true, 0, true, show_log, searching);
    std::vector<Search_result> search_results;
    search_results.emplace_back(best);
    int alpha = -SCORE_MAX;
    int beta = best.value;
    uint64_t legal = board.get_legal() ^ (1ULL << best.policy);
    while (legal && search_results.size() < n_moves) {
        Search_result result_loss = ai_window_legal_searching(board, alpha, beta, level, true, 0, true, show_log, legal, searching);
        legal ^= 1ULL << result_loss.policy;
        search_results.emplace_back(result_loss);
    }
    return search_results;
}

std::vector<Search_result> ai_best_moves_loss_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max, bool *searching) {
    Search_result best = ai_searching(board, level, true, 0, true, show_log, searching);
    std::vector<Search_result> search_results;
    search_results.emplace_back(best);
    int alpha = best.value - loss_max - 1;
    int beta = best.value;
    uint64_t legal = board.get_legal() ^ (1ULL << best.policy);
    while (legal && (*searching)) {
        Search_result result_loss = ai_window_legal_searching(board, alpha, beta, level, true, 0, true, show_log, legal, searching);
        legal ^= 1ULL << result_loss.policy;
        if (*searching) {
            if (result_loss.value >= best.value - loss_max) {
                search_results.emplace_back(result_loss);
            } else {
                break;
            }
        }
    }
    return search_results;
}

Search_result ai_loss_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max, bool *searching) {
    std::vector<Search_result> search_results = ai_best_moves_loss_searching(board, level, use_book, book_acc_level, use_multi_thread, show_log, loss_max, searching);
    if (show_log){
        std::cerr << "play loss candidate " << search_results.size() << " moves: ";
        for (const Search_result &elem: search_results) {
            std::cerr << idx_to_coord(elem.policy) << "@" << elem.value << " ";
        }
        std::cerr << std::endl;
    }
    return search_results[myrandrange(0, (int)search_results.size())];
}

Search_result ai_loss(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max) {
    bool searching = true;
    return ai_loss_searching(board, level, use_book, book_acc_level, use_multi_thread, show_log, loss_max, &searching);
}




struct AI_TL_Elem {
    Board board;
    AI_TL_Elem* parent;
    AI_TL_Elem* children[40] = {nullptr}; // max 40 children
    int n_children;
    int n;
    double v;
    double search_v;
    int last_move;
    bool is_complete_search;
    bool good_moves_already_searched;
    int sgn;

    void reset() {
        parent = nullptr;
        n_children = 0;
        n = 0;
        v = SCORE_UNDEFINED;
        search_v = -SCORE_MAX;
        last_move = MOVE_UNDEFINED;
        is_complete_search = false;
        good_moves_already_searched = false;
        sgn = 1;
    }

    AI_TL_Elem() {
        reset();
    }
};



constexpr int AI_TIME_LIMIT_LEVEL = 21;
constexpr int AI_TIME_LIMIT_LEVEL_ROOT = 25;
constexpr int N_MAX_NODES_AI_TL = 1000000;
constexpr int AI_TIME_LIMIT_EXPAND_THRESHOLD = 4;
constexpr int START_NORMAL_SEARCH_EMPTIES = 38;

struct AI_TL_Array {
    std::mutex mtx;
    AI_TL_Elem ai_time_limit_elems[N_MAX_NODES_AI_TL];
    int n_expanded_nodes;

    AI_TL_Array() {
        n_expanded_nodes = 0;
    }
};

AI_TL_Array ai_tl_array;


// Search_result ai_time_limit(Board board, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t remaining_time_msec, thread_id_t thread_id, bool *searching) {
//     uint64_t time_limit = calc_time_limit_ply_MCTS(board, remaining_time_msec, show_log);
//     if (show_log) {
//         std::cerr << "ai_time_limit start! tl " << time_limit << " remaining " << remaining_time_msec << " n_empties " << HW2 - board.n_discs() << " " << board.to_str() << std::endl;
//     }
//     uint64_t strt = tim();
//     int n_empties = HW2 - board.n_discs();
//     Search_result search_result;
//     // if (n_empties <= START_NORMAL_SEARCH_EMPTIES || time_limit < 20000ULL) { // normal search
//         search_result = ai_common(board, -SCORE_MAX, SCORE_MAX, MAX_LEVEL, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, time_limit, thread_id, searching);
//     // } else {
//     //     AI_TL_Elem *root = nullptr;
//     //     for (int i = 0; i < ai_tl_array.n_expanded_nodes; ++i) {
//     //         if (ai_tl_array.ai_time_limit_elems[i].board == board) {
//     //             root = &ai_tl_array.ai_time_limit_elems[i];
//     //             break;
//     //         }
//     //     }
//     //     if (root == nullptr) {
//     //         ai_tl_array.mtx.lock();
//     //             ai_tl_array.n_expanded_nodes = 0;
//     //             root = &ai_tl_array.ai_time_limit_elems[ai_tl_array.n_expanded_nodes++];
//     //             root->reset();
//     //             root->sgn = 1;
//     //             root->board = board;
//     //         ai_tl_array.mtx.unlock();
//     //     }
//     //     search_result.nodes = 0;
//     //     // Search_result root_search_result = ai_searching_thread_id(board, AI_TIME_LIMIT_LEVEL_ROOT, true, 0, true, false, thread_id, searching);
//     //     // search_result.nodes += root_search_result.nodes;
//     //     // std::cerr << "root " << root_search_result.value << std::endl;
//     //     while (tim() - strt < time_limit && *searching && ai_tl_array.n_expanded_nodes < N_MAX_NODES_AI_TL - 3000) {
//     //         if (root->n_children == 1 && root->good_moves_already_searched && root->n >= 20) {
//     //             break;
//     //         }
//     //         AI_TL_Elem *current_node = root;
//     //         // double sum_loss = 0.0;
//     //         ai_tl_array.mtx.lock();
//     //             while (current_node->n_children) {
//     //                 // if (current_node == root && !current_node->good_moves_already_searched) {
//     //                 //     break;
//     //                 // }
//     //                 double current_node_w = -INF;
//     //                 if (!current_node->good_moves_already_searched) {
//     //                     double min_child_v = INF;
//     //                     for (int i = 0; i < current_node->n_children; ++i) {
//     //                         AI_TL_Elem *child = current_node->children[i];
//     //                         if (-HW2 <= -child->v && -child->v <= HW2) {
//     //                             double vv = -child->v;
//     //                             if (-HW2 <= -child->search_v && -child->search_v <= HW2) {
//     //                                 vv = std::max(vv, -child->search_v);
//     //                             }
//     //                             // std::cerr << -child->v << " " << -child->search_v << std::endl;
//     //                             min_child_v = std::min(min_child_v, vv);
//     //                         }
//     //                     }
//     //                     double v = min_child_v / 64.0;
//     //                     double u = 0.2 * sqrt(current_node->n) / (1 + 0);
//     //                     current_node_w = v + u - myrandom() * 0.15;
//     //                     // break;
//     //                 }
//     //                 double max_w = -INF;
//     //                 double additional_loss = 0.0;
//     //                 AI_TL_Elem *max_node = nullptr;
//     //                 // double max_v = -INF;
//     //                 // for (int i = 0; i < current_node->n_children; ++i) {
//     //                 //     AI_TL_Elem *child = current_node->children[i];
//     //                 //     max_v = std::max(max_v, -child->v);
//     //                 // }
//     //                 for (int i = 0; i < current_node->n_children; ++i) {
//     //                     AI_TL_Elem *child = current_node->children[i];
//     //                     // double loss = max_v - (-child->v);
//     //                     // double v = 1.0 - exp((max_v - (-child->v)) - 4.0);
//     //                     // double v = -std::max(0.0, (max_v - (-child->v)) - 2) / 64.0;
//     //                     double v = -child->v / 64.0;
//     //                     double u = 0.2 * sqrt(current_node->n) / (1 + child->n);
//     //                     // double l = 1.5 * loss / 64.0;
//     //                     // double w = v + u - l;
//     //                     double w = v + u;
//     //                     if (current_node == &ai_tl_array.ai_time_limit_elems[0]) {
//     //                         w += myrandom() * 0.05;
//     //                     }
//     //                     if (w > max_w && !child->is_complete_search) {
//     //                         max_w = w;
//     //                         max_node = child;
//     //                         // additional_loss = loss;
//     //                     }
//     //                 }
//     //                 if (current_node_w >= max_w) {
//     //                     // std::cerr << "current node selected" << std::endl;
//     //                     break;
//     //                 } else if (current_node_w != -INF) {
//     //                     std::cerr << "current node not selected" << std::endl;
//     //                 }
//     //                 current_node = max_node;
//     //                 // sum_loss += additional_loss;
//     //                 // if (myrandom() < 0.5 && !current_node->good_moves_already_searched) {
//     //                 //     break;
//     //                 // }
//     //             }
//     //         ai_tl_array.mtx.unlock();
//     //         if (current_node == nullptr) {
//     //             if (show_log) {
//     //                 std::cerr << "no children found, breaking" << std::endl;
//     //             }
//     //             break;
//     //         }
//     //         // std::cerr << "START current node n_discs " << current_node->board.n_discs() << " " << current_node->board.to_str() << " " << idx_to_coord(current_node->last_move) << " n " << current_node->n << " v " << -current_node->v << " complete_search " << current_node->is_complete_search << std::endl;

//     //         uint64_t legal = current_node->board.get_legal();
//     //         for (int i = 0; i < current_node->n_children; ++i) {
//     //             AI_TL_Elem *child = current_node->children[i];
//     //             legal ^= 1ULL << child->last_move;
//     //         }
//     //         if (legal == 0) {
//     //             current_node->good_moves_already_searched = true;
//     //             continue;
//     //         }
            
//     //         AI_TL_Elem *new_node;
//     //         AI_TL_Elem *selfplay_node = current_node;

//     //         double current_node_max_v = -INF;
//     //         for (int i = 0; i < current_node->n_children; ++i) {
//     //             AI_TL_Elem *child = current_node->children[i];
//     //             current_node_max_v = std::max(current_node_max_v, -child->v);
//     //         }

//     //         Flip flip;
//     //         int level = AI_TIME_LIMIT_LEVEL;
//     //         if (selfplay_node == root) {
//     //             level = AI_TIME_LIMIT_LEVEL_ROOT;
//     //         }
//     //         Search_result next_move_search_result = ai_legal_searching_thread_id(selfplay_node->board, level, true, 0, true, false, legal, thread_id, searching);
//     //         search_result.nodes += next_move_search_result.nodes;
//     //         if (next_move_search_result.value + AI_TIME_LIMIT_EXPAND_THRESHOLD < current_node_max_v && next_move_search_result.value + AI_TIME_LIMIT_EXPAND_THRESHOLD < selfplay_node->search_v) {
//     //             selfplay_node->good_moves_already_searched = true;
//     //             continue;
//     //         }
//     //         selfplay_node->search_v = std::max(selfplay_node->search_v, (double)next_move_search_result.value);
//     //         ai_tl_array.mtx.lock();
//     //             new_node = &ai_tl_array.ai_time_limit_elems[ai_tl_array.n_expanded_nodes++];
//     //             new_node->reset();
//     //             new_node->sgn = selfplay_node->sgn * -1;
//     //             new_node->board = selfplay_node->board.copy();
//     //             calc_flip(&flip, &new_node->board, next_move_search_result.policy);
//     //             new_node->board.move_board(&flip);
//     //             new_node->parent = selfplay_node;
//     //             new_node->last_move = next_move_search_result.policy;
//     //             selfplay_node->children[selfplay_node->n_children++] = new_node;
//     //             selfplay_node = new_node;
//     //         ai_tl_array.mtx.unlock();
//     //         if (show_log) {
//     //             // std::cerr << idx_to_coord(next_move_search_result.policy);
//     //         }

//     //         int score_sgn = -1;
//     //         bool search_failed = false;
//     //         while (*searching) {
//     //             if (selfplay_node->board.get_legal() == 0ULL) {
//     //                 score_sgn *= -1;
//     //                 ai_tl_array.mtx.lock();
//     //                     new_node = &ai_tl_array.ai_time_limit_elems[ai_tl_array.n_expanded_nodes++];
//     //                     new_node->reset();
//     //                     new_node->sgn = selfplay_node->sgn * -1;
//     //                     new_node->board = selfplay_node->board.copy();
//     //                     new_node->board.pass();
//     //                     new_node->parent = selfplay_node;
//     //                     new_node->last_move = MOVE_PASS;
//     //                     new_node->v = SCORE_UNDEFINED;
//     //                     new_node->search_v = SCORE_UNDEFINED;
//     //                     selfplay_node->children[selfplay_node->n_children++] = new_node;
//     //                     selfplay_node = new_node;
//     //                 ai_tl_array.mtx.unlock();
//     //                 if (board.get_legal() == 0ULL) {
//     //                     selfplay_node->v = board.score_player();
//     //                     if (show_log) {
//     //                         // std::cerr << "... result " << score_sgn * selfplay_node->v;
//     //                     }
//     //                     break;
//     //                 }
//     //             }
//     //             if (*searching) {
//     //                 Search_result selfplay_search_result = ai_searching_thread_id(selfplay_node->board, AI_TIME_LIMIT_LEVEL, false, 0, true, false, thread_id, searching);
//     //                 if (*searching && is_valid_policy(selfplay_search_result.policy)) {
//     //                     selfplay_node->search_v = selfplay_search_result.value;
//     //                     if (show_log) {
//     //                         // std::cerr << idx_to_coord(selfplay_search_result.policy);
//     //                     }
//     //                     if (selfplay_node->board.n_discs() >= HW2 - 21) { // complete search with last 21 empties in lv.15- (initial level)
//     //                         if (show_log) {
//     //                             // std::cerr << "... result " << score_sgn * selfplay_search_result.value;
//     //                         }
//     //                         selfplay_node->v = selfplay_search_result.value;
//     //                         break;
//     //                     }
//     //                     calc_flip(&flip, &selfplay_node->board, selfplay_search_result.policy);
//     //                     score_sgn *= -1;
//     //                     ai_tl_array.mtx.lock();
//     //                         new_node = &ai_tl_array.ai_time_limit_elems[ai_tl_array.n_expanded_nodes++];
//     //                         new_node->reset();
//     //                         new_node->sgn = selfplay_node->sgn * -1;
//     //                         new_node->board = selfplay_node->board.copy();
//     //                         new_node->board.move_board(&flip);
//     //                         new_node->parent = selfplay_node;
//     //                         new_node->last_move = selfplay_search_result.policy;
//     //                         selfplay_node->children[selfplay_node->n_children++] = new_node;
//     //                         selfplay_node = new_node;
//     //                     ai_tl_array.mtx.unlock();
//     //                 }
//     //             }
//     //             if (!(*searching)) {
//     //                 if (show_log) {
//     //                     // std::cerr << " terminated";
//     //                 }
//     //                 search_failed = true;
//     //                 break;
//     //             }
//     //         }
//     //         if (show_log) {
//     //             // std::cerr << std::endl;
//     //         }

//     //         // std::cerr << "FINISH current node n_discs " << current_node->board.n_discs() << " " << current_node->board.to_str() << " " << idx_to_coord(current_node->last_move) << " n " << current_node->n << " v " << -current_node->v << " complete_search " << current_node->is_complete_search << std::endl;
//     //         ++selfplay_node->n;
//     //         if (!search_failed) {
//     //             ai_tl_array.mtx.lock();
//     //                 AI_TL_Elem *node = selfplay_node->parent;
//     //                 while (node != nullptr) {
//     //                     // node->v = -INF;
//     //                     double v = -INF;
//     //                     node->is_complete_search = node->good_moves_already_searched;
//     //                     node->n = 0;
//     //                     for (int i = 0; i < node->n_children; ++i) {
//     //                         AI_TL_Elem *child = node->children[i];
//     //                         v = std::max(v, -child->v);
//     //                         node->is_complete_search &= child->is_complete_search;
//     //                         node->n += child->n;
//     //                     }
//     //                     // node->v = 0.9 * v + 0.1 * node->v;
//     //                     node->v = v;
//     //                     // if (-HW2 <= node->v && node->v <= HW2) {
//     //                     //     node->v = 0.9 * v + 0.1 * node->v;
//     //                     // } else {
//     //                     //     node->v = v;
//     //                     // }
//     //                     // std::cerr << "discs " << node->board.n_discs() << " " << node->board.to_str() << " n " << node->n << " v " << node->v << std::endl;
//     //                     node = node->parent;
//     //                 }
//     //             ai_tl_array.mtx.unlock();
//     //         }
//     //         // std::cerr << n_expanded_nodes << " nodes expanded" << std::endl;
//     //         // for (int i = 0; i < n_expanded_nodes; ++i) {
//     //         //     AI_TL_Elem elem = ai_time_limit_elems[i];
//     //         //     std::cerr << "elem n_discs " << elem.board.n_discs() << " " << idx_to_coord(elem.last_move) << " n " << elem.n << " v " << -elem.v << " complete_search " << elem.is_complete_search << std::endl;
//     //         // }
//     //         // std::cerr << std::endl;

//     //         int best_n = -1;
//     //         int best_policy = -1;
//     //         double best_value = -INF;
//     //         int second_best_n = -1;
//     //         int second_policy = -1;
//     //         double second_value = -INF;
//     //         for (int i = 0; i < root->n_children; ++i) {
//     //             AI_TL_Elem *child = root->children[i];
//     //             // if (child->n > best_n || (child->n == best_n && -child->v > best_value)) {
//     //             if (-child->v > best_value || (-child->v == best_value && child->n > best_n)) {
//     //                 second_best_n = best_n;
//     //                 second_policy = best_policy;
//     //                 second_value = best_value;
//     //                 best_n = child->n;
//     //                 best_policy = child->last_move;
//     //                 best_value = -child->v;
//     //             // } else if (child->n > second_best_n) {
//     //             } else if (-child->v > second_value) {
//     //                 second_best_n = child->n;
//     //                 second_policy = child->last_move;
//     //                 second_value = -child->v;
//     //             }
//     //         }
//     //         std::cerr << "expanded " << ai_tl_array.n_expanded_nodes << " root n " << root->n << " n_children " << root->n_children << " best " << idx_to_coord(best_policy) << " " << best_value << " / " << best_n << " second " << idx_to_coord(second_policy) << " " << second_value << " / " << second_best_n << " time " << tim() - strt << std::endl;
//     //         if (best_n > second_best_n * 2 && best_value > second_value && root->n > 40) {
//     //             if (show_log) {
//     //                 std::cerr << "enough searched best_n " << best_n << " second_best_n " << second_best_n << std::endl;
//     //             }
//     //             break;
//     //         }
//     //     }
//     //     std::cerr << "level " << AI_TIME_LIMIT_LEVEL << " " << ai_tl_array.n_expanded_nodes << " nodes expanded in " << tim() - strt << " ms" << std::endl;
//     //     int best_n = -1;
//     //     double best_value = -INF;
//     //     for (int i = 0; i < root->n_children; ++i) {
//     //         AI_TL_Elem *child = root->children[i];
//     //         std::cerr << idx_to_coord(child->last_move) << " " << -child->v << " " << child->n << std::endl;
//     //         // if (child->n > best_n || (child->n == best_n && -child->v > best_value)) {
//     //         if (-child->v > best_value || (-child->v == best_value && child->n > best_n)) {
//     //             best_n = child->n;
//     //             best_value = -child->v;
//     //             search_result.policy = child->last_move;
//     //             search_result.value = -child->v;
//     //         }
//     //     }
//     //     search_result.time = tim() - strt;
//     //     search_result.nps = calc_nps(search_result.nodes, search_result.time);
//     // }
//     if (show_log) {
//         std::cerr << "ai_time_limit selected " << idx_to_coord(search_result.policy) << " value " << search_result.value << " depth " << search_result.depth << "@" << search_result.probability << "%" << " time " << tim() - strt << " " << board.to_str() << std::endl << std::endl;
//     }
//     return search_result;
// }






constexpr uint64_t AI_TL_MAIN_SEARCH_RESERVED_TIME = 1000ULL;
constexpr int AI_TL_PRESEARCH_LEVEL = 21;
constexpr int AI_TL_PRESEARCH_MASK_NWS_VALUE_OFFSET = 2;
constexpr int AI_TL_PRESEARCH_ROOT_MASK_NWS_VALUE_OFFSET = 4;
constexpr int AI_TL_PRESEARCH_MASK_INTERVAL = 4;

struct AI_TL_Presearch_Record {
    Board board;
    int n_visits;
    int previous_policy;
    int previous_value;
};

struct AI_TL_Presearch_Path {
    Board board;
    std::vector<int> line;
};

inline uint64_t get_ai_time_limit_presearch_remaining(uint64_t time_limit, uint64_t strt) {
    if (time_limit == TIME_LIMIT_INF) {
        return TIME_LIMIT_INF;
    }
    uint64_t elapsed = tim() - strt;
    if (elapsed >= time_limit) {
        return 0;
    }
    return time_limit - elapsed;
}

inline bool move_ai_time_limit_presearch_child(Board board, int policy, Board *child, std::vector<int> *child_line) {
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            return false;
        }
        if (child_line != nullptr) {
            child_line->emplace_back(MOVE_NOMOVE);
        }
    }
    if (!is_valid_policy(policy) || (legal & (1ULL << policy)) == 0) {
        return false;
    }
    Flip flip;
    calc_flip(&flip, &board, policy);
    board.move_copy(&flip, child);
    if (child_line != nullptr) {
        child_line->emplace_back(policy);
    }
    return true;
}

inline std::string ai_time_limit_presearch_line_str(const std::vector<int> &line, bool passed) {
    std::string res;
    bool last_pass = false;
    for (int policy: line) {
        if (policy == MOVE_NOMOVE) {
            if (!res.empty()) {
                res += " ";
            }
            res += "pass";
            last_pass = true;
        } else {
            if (last_pass) {
                res += " ";
            }
            res += idx_to_coord(policy);
            last_pass = false;
        }
    }
    if (passed) {
        if (!res.empty()) {
            res += " ";
        }
        res += "pass";
    }
    if (res.empty()) {
        res = "root";
    }
    return res;
}

inline void print_ai_time_limit_presearch_result(const std::vector<int> &line, bool passed, const Search_result &result, uint64_t elapsed, bool succeeded) {
    std::cerr << "ai_time_limit presearch line " << ai_time_limit_presearch_line_str(line, passed);
    if (succeeded) {
        std::cerr << " result " << idx_to_coord(result.policy)
                  << " value " << result.value
                  << " depth " << result.depth << "@" << result.probability << "%"
                  << " time " << elapsed << " ms"
                  << " nodes " << result.nodes
                  << std::endl;
    } else {
        std::cerr << " result failed"
                  << " time " << elapsed << " ms"
                  << std::endl;
    }
}

inline void print_ai_time_limit_presearch_root_result(const Search_result &result, bool succeeded, bool used_mask_search) {
    std::cerr << "root ";
    if (succeeded) {
        std::cerr << idx_to_coord(result.policy) << " " << result.value;
        if (used_mask_search) {
            std::cerr << " (masked)";
        }
    } else {
        std::cerr << "failed";
    }
    std::cerr << std::endl;
}

inline int ai_time_limit_presearch_find_record(const Board &board, const std::vector<AI_TL_Presearch_Record> &searched_boards) {
    for (int i = 0; i < (int)searched_boards.size(); ++i) {
        if (searched_boards[i].board == board) {
            return i;
        }
    }
    return -1;
}

inline bool ai_time_limit_presearch_search(Board board, int alpha, int beta, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, uint64_t strt, thread_id_t thread_id, bool *searching, Search_result *result) {
    uint64_t remaining = get_ai_time_limit_presearch_remaining(time_limit, strt);
    if (remaining == 0 || !global_searching || !(*searching)) {
        return false;
    }
    bool search_finished = false;
    if (time_limit == TIME_LIMIT_INF) {
        *result = ai_common(board, alpha, beta, AI_TL_PRESEARCH_LEVEL, false, 0, use_multi_thread, false, use_legal, false, TIME_LIMIT_INF, thread_id, searching);
        search_finished = true;
    } else {
        bool presearch_searching = true;
        std::future<Search_result> search_future = std::async(std::launch::async, ai_common, board, alpha, beta, AI_TL_PRESEARCH_LEVEL, false, 0, use_multi_thread, false, use_legal, false, TIME_LIMIT_INF, thread_id, &presearch_searching);
        if (search_future.wait_for(std::chrono::milliseconds(remaining)) == std::future_status::ready) {
            *result = search_future.get();
            search_finished = true;
        } else {
            presearch_searching = false;
            try {
                *result = search_future.get();
            } catch (const std::exception &e) {
            }
        }
    }
    return search_finished && global_searching && *searching && is_valid_policy(result->policy) && (use_legal & (1ULL << result->policy));
}

inline Search_result ai_time_limit_presearch_raw_search_body(Board board, int alpha, int beta, uint64_t use_legal, bool use_multi_thread, thread_id_t thread_id, bool *searching) {
    uint64_t strt = tim();
    Search_result result;
    bool is_mid_search;
    int depth;
    uint_fast8_t mpc_level;
    get_level(AI_TL_PRESEARCH_LEVEL, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    bool is_end_search = (depth == max_depth);
    Search search(&board, mpc_level, use_multi_thread, false);
    search.thread_id = thread_id;
    std::pair<int, int> raw_result = first_nega_scout_legal(&search, alpha, beta, depth, is_end_search, std::vector<Clog_result>(), use_legal, strt, searching);
    result.level = AI_TL_PRESEARCH_LEVEL;
    result.policy = raw_result.second;
    result.value = raw_result.first;
    result.depth = depth;
    result.nodes = search.n_nodes;
    result.time = tim() - strt;
    result.nps = calc_nps(result.nodes, result.time);
    result.is_end_search = is_end_search;
    result.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return result;
}

inline bool ai_time_limit_presearch_raw_search(Board board, int alpha, int beta, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, uint64_t strt, thread_id_t thread_id, bool *searching, Search_result *result) {
    uint64_t remaining = get_ai_time_limit_presearch_remaining(time_limit, strt);
    if (remaining == 0 || !global_searching || !(*searching)) {
        return false;
    }
    bool search_finished = false;
    if (time_limit == TIME_LIMIT_INF) {
        *result = ai_time_limit_presearch_raw_search_body(board, alpha, beta, use_legal, use_multi_thread, thread_id, searching);
        search_finished = true;
    } else {
        bool presearch_searching = true;
        std::future<Search_result> search_future = std::async(std::launch::async, ai_time_limit_presearch_raw_search_body, board, alpha, beta, use_legal, use_multi_thread, thread_id, &presearch_searching);
        if (search_future.wait_for(std::chrono::milliseconds(remaining)) == std::future_status::ready) {
            *result = search_future.get();
            search_finished = true;
        } else {
            presearch_searching = false;
            try {
                *result = search_future.get();
            } catch (const std::exception &e) {
            }
        }
    }
    return search_finished && global_searching && *searching && is_valid_policy(result->policy) && (use_legal & (1ULL << result->policy));
}

inline bool ai_time_limit_presearch_once(Board board, const std::vector<int> &line, bool use_multi_thread, uint64_t time_limit, uint64_t strt, thread_id_t thread_id, bool *searching, Search_result *result, std::vector<AI_TL_Presearch_Record> *searched_boards, bool *already_searched, bool *passed, bool allow_mask_search, bool *used_mask_search, bool delete_tt_before_search) {
    uint64_t remaining = get_ai_time_limit_presearch_remaining(time_limit, strt);
    if (remaining == 0 || !global_searching || !(*searching)) {
        return false;
    }
    uint64_t legal = board.get_legal();
    *passed = false;
    if (legal == 0) {
        board.pass();
        *passed = true;
        legal = board.get_legal();
        if (legal == 0) {
            return false;
        }
    }
    int record_idx = ai_time_limit_presearch_find_record(board, *searched_boards);
    *already_searched = record_idx != -1;
    *used_mask_search = false;
    if (delete_tt_before_search) {
        transposition_table.del(&board, board.hash());
    }
    bool succeeded = false;
    bool should_not_fallback_to_normal_search = false;
    if (
        allow_mask_search && 
        record_idx != -1 && 
        searched_boards->at(record_idx).n_visits > 0 && 
        searched_boards->at(record_idx).n_visits % AI_TL_PRESEARCH_MASK_INTERVAL == AI_TL_PRESEARCH_MASK_INTERVAL - 1 && 
        is_valid_policy(searched_boards->at(record_idx).previous_policy) && 
        is_valid_score(searched_boards->at(record_idx).previous_value)
    ) {
        uint64_t legal_without_previous_best = legal & ~(1ULL << searched_boards->at(record_idx).previous_policy);
        if (legal_without_previous_best != 0) {
            int previous_value = searched_boards->at(record_idx).previous_value;
            Search_result nws_result;
            int nws_value_offset = line.empty() ? AI_TL_PRESEARCH_ROOT_MASK_NWS_VALUE_OFFSET : AI_TL_PRESEARCH_MASK_NWS_VALUE_OFFSET;
            int nws_beta = previous_value - nws_value_offset;
            int nws_alpha = std::max(-SCORE_MAX, nws_beta - 1);
            bool nws_succeeded = ai_time_limit_presearch_raw_search(board, nws_alpha, nws_beta, legal_without_previous_best, use_multi_thread, time_limit, strt, thread_id, searching, &nws_result);
            if (!nws_succeeded) {
                return false;
            }
            if (nws_result.value >= nws_beta) {
                Search_result masked_result;
                bool masked_succeeded = ai_time_limit_presearch_raw_search(board, previous_value, SCORE_MAX, legal_without_previous_best, use_multi_thread, time_limit, strt, thread_id, searching, &masked_result);
                if (!masked_succeeded) {
                    return false;
                }
                if (masked_result.value == previous_value) {
                    Search_result exact_masked_result;
                    bool exact_masked_succeeded = ai_time_limit_presearch_raw_search(board, -SCORE_MAX, SCORE_MAX, legal_without_previous_best, use_multi_thread, time_limit, strt, thread_id, searching, &exact_masked_result);
                    if (!exact_masked_succeeded) {
                        return false;
                    }
                    masked_result = exact_masked_result;
                }
                if (masked_result.value >= previous_value) {
                    *result = masked_result;
                    succeeded = true;
                    *used_mask_search = true;
                    should_not_fallback_to_normal_search = true;
                } else {
                    succeeded = ai_time_limit_presearch_search(board, -SCORE_MAX, SCORE_MAX, legal, use_multi_thread, time_limit, strt, thread_id, searching, result);
                }
            } else {
                succeeded = ai_time_limit_presearch_search(board, -SCORE_MAX, SCORE_MAX, legal, use_multi_thread, time_limit, strt, thread_id, searching, result);
            }
        }
    }
    if (!succeeded && should_not_fallback_to_normal_search) {
        return false;
    }
    if (!succeeded) {
        succeeded = ai_time_limit_presearch_search(board, -SCORE_MAX, SCORE_MAX, legal, use_multi_thread, time_limit, strt, thread_id, searching, result);
    }
    if (succeeded) {
        if (record_idx == -1) {
            searched_boards->push_back(AI_TL_Presearch_Record{board, 1, result->policy, result->value});
        } else {
            ++searched_boards->at(record_idx).n_visits;
            searched_boards->at(record_idx).previous_policy = result->policy;
            searched_boards->at(record_idx).previous_value = result->value;
        }
    }
    return succeeded;
}

inline bool ai_time_limit_presearch(Board board, bool use_multi_thread, bool show_log, uint64_t time_limit, thread_id_t thread_id, bool *searching, Search_result *last_unmasked_root_result) {
    uint64_t strt = tim();
    int n_loop = 0;
    if (show_log) {
        std::cerr << "ai_time_limit presearch tl " << time_limit << " level " << AI_TL_PRESEARCH_LEVEL << std::endl;
    }
    std::vector<AI_TL_Presearch_Record> searched_boards;
    std::vector<AI_TL_Presearch_Path> path;
    path.push_back(AI_TL_Presearch_Path{board, std::vector<int>()});
    bool going_down = true;
    while (get_ai_time_limit_presearch_remaining(time_limit, strt) > 0 && global_searching && *searching) {
        Search_result result;
        bool already_searched = false;
        bool passed = false;
        bool used_mask_search = false;
        uint64_t strt_search = tim();
        bool succeeded = ai_time_limit_presearch_once(path.back().board, path.back().line, use_multi_thread, time_limit, strt, thread_id, searching, &result, &searched_boards, &already_searched, &passed, going_down, &used_mask_search, !going_down);
        bool is_root_search = path.back().line.empty();
        if (show_log && is_root_search) {
            print_ai_time_limit_presearch_root_result(result, succeeded, used_mask_search);
        }
        if (!succeeded) {
            if (show_log && going_down && !is_root_search) {
                print_ai_time_limit_presearch_result(path.back().line, passed, result, tim() - strt_search, false);
            }
            break;
        }
        if (path.back().line.empty() && !passed && !used_mask_search) {
            *last_unmasked_root_result = result;
        }
        if (going_down) {
            if (already_searched) {
                AI_TL_Presearch_Path child = path.back();
                if (!move_ai_time_limit_presearch_child(path.back().board, result.policy, &child.board, &child.line)) {
                    if (show_log && !is_root_search) {
                        print_ai_time_limit_presearch_result(path.back().line, passed, result, tim() - strt_search, true);
                    }
                    break;
                }
                path.push_back(child);
            } else if (path.size() > 1) {
                if (show_log) {
                    print_ai_time_limit_presearch_result(path.back().line, passed, result, tim() - strt_search, true);
                }
                path.pop_back();
                going_down = false;
            } else if (show_log && !is_root_search) {
                print_ai_time_limit_presearch_result(path.back().line, passed, result, tim() - strt_search, true);
            }
        } else {
            if (path.size() > 1) {
                path.pop_back();
            } else {
                going_down = true;
            }
        }
        ++n_loop;
    }
    if (show_log) {
        std::cerr << "ai_time_limit presearch finished loops " << n_loop << " time " << tim() - strt << " ms" << std::endl;
    }
    return is_valid_policy(last_unmasked_root_result->policy);
}

#if IS_GGS_TOURNAMENT
constexpr int AI_TL_GGS_AMBIGUITY_MIN_N_EMPTY = 29;
constexpr int AI_TL_GGS_AMBIGUITY_MAX_N_EMPTY = 48;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_MIN_TIME_LIMIT = 3000ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_MIN_REMAINING_TIME = 30000ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_PROBE_MIN_TIME = 320ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_PROBE_MAX_TIME = 1600ULL;
constexpr double AI_TL_GGS_AMBIGUITY_PROBE_TIME_COE = 0.12;
constexpr double AI_TL_GGS_AMBIGUITY_CLOSE_VALUE = AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 2.0;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_BASE_BONUS = 2500ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_BONUS_PER_CLOSE_MOVE = 3600ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_BONUS_PER_SHALLOW_CLOSE_MOVE = 1200ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_TINY_GAP_BONUS = 2500ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_NARROW_GAP_BONUS = 5500ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_MAX_BONUS = 28000ULL;
constexpr double AI_TL_GGS_AMBIGUITY_MAX_BONUS_COE = 1.35;
constexpr double AI_TL_GGS_AMBIGUITY_NARROW_BEST_ABS_VALUE = 3.0;
constexpr double AI_TL_GGS_AMBIGUITY_NARROW_SECOND_GAP = 5.0;
constexpr double AI_TL_GGS_AMBIGUITY_NARROW_GAP_BONUS_COE = 0.65;
constexpr double AI_TL_GGS_AMBIGUITY_DEFENSIVE_BEST_MIN = -10.0;
constexpr double AI_TL_GGS_AMBIGUITY_DEFENSIVE_BEST_MAX = -3.0;
constexpr double AI_TL_GGS_AMBIGUITY_DEFENSIVE_SECOND_GAP = 6.0;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_DEFENSIVE_BASE_BONUS = 3500ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_DEFENSIVE_MAX_BONUS = 8500ULL;
constexpr double AI_TL_GGS_AMBIGUITY_DEFENSIVE_MAX_BONUS_COE = 0.80;
constexpr double AI_TL_GGS_AMBIGUITY_DEEP_BAD_BEST_MIN = -18.0;
constexpr double AI_TL_GGS_AMBIGUITY_DEEP_BAD_BEST_MAX = -10.0;
constexpr double AI_TL_GGS_AMBIGUITY_DEEP_BAD_SECOND_GAP = 5.0;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_DEEP_BAD_BASE_BONUS = 1800ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_DEEP_BAD_MAX_BONUS = 4800ULL;
constexpr double AI_TL_GGS_AMBIGUITY_DEEP_BAD_MAX_BONUS_COE = 0.35;
constexpr int AI_TL_GGS_AMBIGUITY_MIN_RELIABLE_DEPTH = 20;
constexpr int AI_TL_GGS_AMBIGUITY_DEPTH_LAG_TOLERANCE = 8;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_UNRELIABLE_BASE_BONUS = 2000ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_UNRELIABLE_BONUS_PER_CLOSE_MOVE = 1900ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_UNRELIABLE_TINY_GAP_BONUS = 3500ULL;
constexpr uint64_t AI_TL_GGS_AMBIGUITY_UNRELIABLE_MAX_BONUS = 11500ULL;
constexpr double AI_TL_GGS_AMBIGUITY_UNRELIABLE_MAX_BONUS_COE = 1.00;

inline uint64_t ai_time_limit_ggs_ambiguity_probe_time(const Board &board, uint64_t time_limit, uint64_t remaining_time_msec) {
    const int n_empties = HW2 - board.n_discs();
    if (n_empties < AI_TL_GGS_AMBIGUITY_MIN_N_EMPTY || n_empties > AI_TL_GGS_AMBIGUITY_MAX_N_EMPTY) {
        return 0ULL;
    }
    if (time_limit < AI_TL_GGS_AMBIGUITY_MIN_TIME_LIMIT || remaining_time_msec < AI_TL_GGS_AMBIGUITY_MIN_REMAINING_TIME) {
        return 0ULL;
    }
    if (pop_count_ull(board.get_legal()) < 2) {
        return 0ULL;
    }
    uint64_t probe_time = (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_PROBE_TIME_COE);
    probe_time = std::min<uint64_t>(probe_time, AI_TL_GGS_AMBIGUITY_PROBE_MAX_TIME);
    probe_time = std::min<uint64_t>(probe_time, remaining_time_msec / 40ULL);
    if (probe_time < AI_TL_GGS_AMBIGUITY_PROBE_MIN_TIME) {
        return 0ULL;
    }
    return probe_time;
}

inline uint64_t ai_time_limit_ggs_ambiguity_boost(const Board &board, const std::vector<Ponder_elem> &move_list, uint64_t time_limit, uint64_t remaining_time_msec, bool show_log) {
    double best_value = -INF;
    double second_value = -INF;
    int best_depth = 0;
    int valid_moves = 0;
    for (const Ponder_elem &elem: move_list) {
        if (elem.count <= 0) {
            continue;
        }
        ++valid_moves;
        if (elem.value > best_value) {
            second_value = best_value;
            best_value = elem.value;
            best_depth = elem.depth;
        } else if (elem.value > second_value) {
            second_value = elem.value;
        }
    }
    if (valid_moves < 2) {
        if (show_log) {
            std::cerr << "ggs ambiguity skip valid_moves " << valid_moves << std::endl;
        }
        return time_limit;
    }

    int close_moves = 0;
    int reliable_close_moves = 0;
    std::ostringstream close_move_str;
    for (const Ponder_elem &elem: move_list) {
        if (elem.count <= 0 || elem.value < best_value - AI_TL_GGS_AMBIGUITY_CLOSE_VALUE) {
            continue;
        }
        ++close_moves;
        const bool reliable = elem.depth >= AI_TL_GGS_AMBIGUITY_MIN_RELIABLE_DEPTH && elem.depth + AI_TL_GGS_AMBIGUITY_DEPTH_LAG_TOLERANCE >= best_depth;
        if (reliable) {
            ++reliable_close_moves;
        }
        if (show_log) {
            close_move_str << " " << idx_to_coord(elem.flip.pos) << "(" << std::fixed << std::setprecision(2) << elem.value << ")";
            if (!reliable) {
                close_move_str << "?d" << elem.depth;
            }
        }
    }
    if (close_moves < 2) {
        const double second_gap = best_value - second_value;
        if (
            AI_TL_GGS_AMBIGUITY_DEFENSIVE_BEST_MIN <= best_value &&
            best_value <= AI_TL_GGS_AMBIGUITY_DEFENSIVE_BEST_MAX &&
            second_gap <= AI_TL_GGS_AMBIGUITY_DEFENSIVE_SECOND_GAP
        ) {
            const double gap_ratio = std::max(0.0, std::min(1.0, (AI_TL_GGS_AMBIGUITY_DEFENSIVE_SECOND_GAP - second_gap) / AI_TL_GGS_AMBIGUITY_DEFENSIVE_SECOND_GAP));
            uint64_t bonus = AI_TL_GGS_AMBIGUITY_DEFENSIVE_BASE_BONUS +
                             (uint64_t)((double)(AI_TL_GGS_AMBIGUITY_DEFENSIVE_MAX_BONUS - AI_TL_GGS_AMBIGUITY_DEFENSIVE_BASE_BONUS) * gap_ratio);
            bonus = std::min<uint64_t>(bonus, AI_TL_GGS_AMBIGUITY_DEFENSIVE_MAX_BONUS);
            bonus = std::min<uint64_t>(bonus, (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_DEFENSIVE_MAX_BONUS_COE));
            const int n_empties = HW2 - board.n_discs();
            const double remaining_moves = (double)(n_empties + 1) / 2.0;
            const uint64_t boosted_time_limit = time_management_ggs_cap_time_limit(time_limit + bonus, remaining_time_msec, remaining_moves);
            if (show_log) {
                std::cerr << "ggs ambiguity defensive best " << best_value
                          << " second_gap " << second_gap
                          << " bonus " << bonus
                          << " tl " << time_limit << " -> " << boosted_time_limit << std::endl;
            }
            return boosted_time_limit;
        }
        if (
            AI_TL_GGS_AMBIGUITY_DEEP_BAD_BEST_MIN <= best_value &&
            best_value < AI_TL_GGS_AMBIGUITY_DEEP_BAD_BEST_MAX &&
            second_gap <= AI_TL_GGS_AMBIGUITY_DEEP_BAD_SECOND_GAP
        ) {
            const double gap_ratio = std::max(0.0, std::min(1.0, (AI_TL_GGS_AMBIGUITY_DEEP_BAD_SECOND_GAP - second_gap) / AI_TL_GGS_AMBIGUITY_DEEP_BAD_SECOND_GAP));
            uint64_t bonus = AI_TL_GGS_AMBIGUITY_DEEP_BAD_BASE_BONUS +
                             (uint64_t)((double)(AI_TL_GGS_AMBIGUITY_DEEP_BAD_MAX_BONUS - AI_TL_GGS_AMBIGUITY_DEEP_BAD_BASE_BONUS) * gap_ratio);
            bonus = std::min<uint64_t>(bonus, AI_TL_GGS_AMBIGUITY_DEEP_BAD_MAX_BONUS);
            bonus = std::min<uint64_t>(bonus, (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_DEEP_BAD_MAX_BONUS_COE));
            const int n_empties = HW2 - board.n_discs();
            const double remaining_moves = (double)(n_empties + 1) / 2.0;
            const uint64_t boosted_time_limit = time_management_ggs_cap_time_limit(time_limit + bonus, remaining_time_msec, remaining_moves);
            if (show_log) {
                std::cerr << "ggs ambiguity deep-bad best " << best_value
                          << " second_gap " << second_gap
                          << " bonus " << bonus
                          << " tl " << time_limit << " -> " << boosted_time_limit << std::endl;
            }
            return boosted_time_limit;
        }
        if (
            -AI_TL_GGS_AMBIGUITY_NARROW_BEST_ABS_VALUE <= best_value &&
            best_value <= AI_TL_GGS_AMBIGUITY_NARROW_BEST_ABS_VALUE &&
            second_gap <= AI_TL_GGS_AMBIGUITY_NARROW_SECOND_GAP
        ) {
            uint64_t bonus = std::min<uint64_t>(
                AI_TL_GGS_AMBIGUITY_NARROW_GAP_BONUS,
                (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_NARROW_GAP_BONUS_COE)
            );
            const int n_empties = HW2 - board.n_discs();
            const double remaining_moves = (double)(n_empties + 1) / 2.0;
            const uint64_t boosted_time_limit = time_management_ggs_cap_time_limit(time_limit + bonus, remaining_time_msec, remaining_moves);
            if (show_log) {
                std::cerr << "ggs ambiguity narrow best " << best_value
                          << " second_gap " << second_gap
                          << " bonus " << bonus
                          << " tl " << time_limit << " -> " << boosted_time_limit << std::endl;
            }
            return boosted_time_limit;
        }
        if (show_log) {
            std::cerr << "ggs ambiguity quiet best " << best_value << " second_gap " << second_gap << std::endl;
        }
        return time_limit;
    }
    if (reliable_close_moves < 2) {
        uint64_t bonus = AI_TL_GGS_AMBIGUITY_UNRELIABLE_BASE_BONUS +
                         AI_TL_GGS_AMBIGUITY_UNRELIABLE_BONUS_PER_CLOSE_MOVE * (uint64_t)(close_moves - 1);
        const double second_gap = best_value - second_value;
        if (
            -AI_TL_GGS_AMBIGUITY_NARROW_BEST_ABS_VALUE <= best_value &&
            best_value <= AI_TL_GGS_AMBIGUITY_NARROW_BEST_ABS_VALUE &&
            second_gap <= AI_TL_ADDITIONAL_SEARCH_THRESHOLD
        ) {
            bonus += AI_TL_GGS_AMBIGUITY_UNRELIABLE_TINY_GAP_BONUS;
        }
        bonus = std::min<uint64_t>(bonus, AI_TL_GGS_AMBIGUITY_UNRELIABLE_MAX_BONUS);
        bonus = std::min<uint64_t>(bonus, (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_UNRELIABLE_MAX_BONUS_COE));

        const int n_empties = HW2 - board.n_discs();
        const double remaining_moves = (double)(n_empties + 1) / 2.0;
        const uint64_t boosted_time_limit = time_management_ggs_cap_time_limit(time_limit + bonus, remaining_time_msec, remaining_moves);
        if (show_log) {
            std::cerr << "ggs ambiguity unreliable close_moves " << close_moves << "/" << valid_moves
                      << " reliable " << reliable_close_moves
                      << " best_depth " << best_depth
                      << " second_gap " << second_gap
                      << " bonus " << bonus
                      << " tl " << time_limit << " -> " << boosted_time_limit
                      << " moves" << close_move_str.str() << std::endl;
        }
        return boosted_time_limit;
    }

    uint64_t bonus = AI_TL_GGS_AMBIGUITY_BASE_BONUS +
                     AI_TL_GGS_AMBIGUITY_BONUS_PER_CLOSE_MOVE * (uint64_t)(reliable_close_moves - 1);
    if (close_moves > reliable_close_moves) {
        bonus += AI_TL_GGS_AMBIGUITY_BONUS_PER_SHALLOW_CLOSE_MOVE * (uint64_t)(close_moves - reliable_close_moves);
    }
    const double second_gap = best_value - second_value;
    if (second_gap <= AI_TL_ADDITIONAL_SEARCH_THRESHOLD) {
        bonus += AI_TL_GGS_AMBIGUITY_TINY_GAP_BONUS;
    }
    bonus = std::min<uint64_t>(bonus, AI_TL_GGS_AMBIGUITY_MAX_BONUS);
    bonus = std::min<uint64_t>(bonus, (uint64_t)((double)time_limit * AI_TL_GGS_AMBIGUITY_MAX_BONUS_COE));

    const int n_empties = HW2 - board.n_discs();
    const double remaining_moves = (double)(n_empties + 1) / 2.0;
    const uint64_t boosted_time_limit = time_management_ggs_cap_time_limit(time_limit + bonus, remaining_time_msec, remaining_moves);
    if (show_log) {
        std::cerr << "ggs ambiguity close_moves " << close_moves << "/" << valid_moves
                  << " reliable " << reliable_close_moves
                  << " second_gap " << second_gap
                  << " bonus " << bonus
                  << " tl " << time_limit << " -> " << boosted_time_limit
                  << " moves" << close_move_str.str() << std::endl;
    }
    return boosted_time_limit;
}
#endif

Search_result ai_time_limit(Board board, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t remaining_time_msec, thread_id_t thread_id, bool *searching) {
    uint64_t time_limit = calc_time_limit_ply(board, remaining_time_msec, show_log);
    if (show_log) {
        std::cerr << "ai_time_limit start! tl " << time_limit << " remaining " << remaining_time_msec << " n_empties " << HW2 - board.n_discs() << " " << board.to_str() << std::endl;
    }
    uint64_t strt = tim();
    int n_empties = HW2 - board.n_discs();
    // if (n_empties >= 38 /*35*/ && time_limit >= 2000ULL) {
    //     bool get_values_searching = true;
    //     uint64_t get_values_tl = 600ULL;
    //     if (show_log) {
    //         std::cerr << "getting values tl " << get_values_tl << std::endl;
    //     }
    //     uint64_t strt_get_values = tim();
    //         std::vector<Ponder_elem> get_values_move_list = ai_get_values(board, show_log, get_values_tl, thread_id);
    //     uint64_t elapsed_get_values = tim() - strt_get_values;
    //     if (time_limit > elapsed_get_values) {
    //         time_limit -= elapsed_get_values;
    //     } else {
    //         time_limit = 10;
    //     }
    //     if (remaining_time_msec > elapsed_get_values) {
    //         remaining_time_msec -= elapsed_get_values;
    //     } else {
    //         remaining_time_msec = 0;
    //     }

    //     if (time_limit > 5000ULL && get_values_move_list.size() >= 2) {
    //         double best_value = get_values_move_list[0].value;
    //         int n_good_moves = 0;
    //         if (show_log) {
    //             std::cerr << "good moves (1):";
    //         }
    //         for (const Ponder_elem &elem: get_values_move_list) {
    //             if (elem.value >= best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 2.0) {
    //                 ++n_good_moves;
    //                 if (show_log) {
    //                     std::cerr << " " << idx_to_coord(elem.flip.pos);
    //                 }
    //             }
    //         }
    //         if (show_log) {
    //             std::cerr << std::endl;
    //         }
    //         if (n_good_moves >= 2) {
    //             uint64_t align_moves_tl = 0;
    //             if (remaining_time_msec > 40000) {
    //                 align_moves_tl = std::min<uint64_t>(10000ULL, time_limit * 0.8);
    //             }
    //             uint64_t strt_align_move_levels = tim();
    //                 std::vector<Ponder_elem> after_move_list = ai_align_move_levels(board, show_log, get_values_move_list, n_good_moves, align_moves_tl, thread_id, 29);
    //             uint64_t elapsed_align_move_levels = tim() - strt_align_move_levels;
    //             if (time_limit > elapsed_align_move_levels) {
    //                 time_limit -= elapsed_align_move_levels;
    //             } else {
    //                 time_limit = 10;
    //             }
    //             if (remaining_time_msec > elapsed_align_move_levels) {
    //                 remaining_time_msec -= elapsed_align_move_levels;
    //             } else {
    //                 remaining_time_msec = 0;
    //             }

    //             double new_best_value = after_move_list[0].value;
    //             int new_n_good_moves = 0;
    //             if (show_log) {
    //                 std::cerr << "good moves (2):";
    //             }
    //             for (const Ponder_elem &elem: after_move_list) {
    //                 if (elem.value >= new_best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD) {
    //                     ++new_n_good_moves;
    //                     if (show_log) {
    //                         std::cerr << " " << idx_to_coord(elem.flip.pos);
    //                     }
    //                 }
    //             }
    //             if (show_log) {
    //                 std::cerr << std::endl;
    //             }
    //             if (new_n_good_moves >= 2) {
    //                 time_limit = request_more_time(board, remaining_time_msec, time_limit, show_log);
    //             }
    //         }
    //     }
    // }
    // if (n_empties >= 33 && time_limit >= 1ULL) {
    //     Search_result presearch_result;
    //     ai_time_limit_presearch(board, use_multi_thread, show_log, time_limit, thread_id, searching, &presearch_result);
    //     if (is_valid_policy(presearch_result.policy)) {
    //         presearch_result.time = tim() - strt;
    //         presearch_result.nps = calc_nps(presearch_result.nodes, presearch_result.time);
    //         if (show_log) {
    //             std::cerr << "ai_time_limit selected by presearch " << idx_to_coord(presearch_result.policy) << " value " << presearch_result.value << " depth " << presearch_result.depth << "@" << presearch_result.probability << "%" << " time " << presearch_result.time << " " << board.to_str() << std::endl << std::endl;
    //         }
    //         return presearch_result;
    //     }
    //     time_limit = 1;
    // } else if (n_empties >= 31 && time_limit >= 2000ULL) {
    //     uint64_t presearch_time_limit = time_limit - AI_TL_MAIN_SEARCH_RESERVED_TIME;
    //     uint64_t strt_presearch = tim();
    //     Search_result presearch_result;
    //     ai_time_limit_presearch(board, use_multi_thread, show_log, presearch_time_limit, thread_id, searching, &presearch_result);
    //     uint64_t elapsed_presearch = tim() - strt_presearch;
    //     if (time_limit > elapsed_presearch) {
    //         time_limit -= elapsed_presearch;
    //     } else {
    //         time_limit = 1;
    //     }
    // }
    // if (n_empties >= 31 && time_limit >= 2000ULL) {
    //     uint64_t presearch_time_limit = time_limit - AI_TL_MAIN_SEARCH_RESERVED_TIME;
    //     uint64_t strt_presearch = tim();
    //     Search_result presearch_result;
    //     ai_time_limit_presearch(board, use_multi_thread, show_log, presearch_time_limit, thread_id, searching, &presearch_result);
    //     uint64_t elapsed_presearch = tim() - strt_presearch;
    //     if (time_limit > elapsed_presearch) {
    //         time_limit -= elapsed_presearch;
    //     } else {
    //         time_limit = 1;
    //     }
    // }
#if IS_GGS_TOURNAMENT
    uint64_t ambiguity_probe_time = ai_time_limit_ggs_ambiguity_probe_time(board, time_limit, remaining_time_msec);
    if (ambiguity_probe_time > 0ULL && global_searching && *searching) {
        if (show_log) {
            std::cerr << "ggs ambiguity probe tl " << ambiguity_probe_time << std::endl;
        }
        uint64_t strt_ambiguity_probe = tim();
        std::vector<Ponder_elem> ambiguity_move_list = ai_get_values(board, show_log, ambiguity_probe_time, thread_id);
        uint64_t elapsed_ambiguity_probe = tim() - strt_ambiguity_probe;
        time_limit = std::max<uint64_t>(1ULL, get_this_search_time_limit(time_limit, elapsed_ambiguity_probe));
        if (remaining_time_msec > elapsed_ambiguity_probe) {
            remaining_time_msec -= elapsed_ambiguity_probe;
        } else {
            remaining_time_msec = 1ULL;
        }
        time_limit = ai_time_limit_ggs_ambiguity_boost(board, ambiguity_move_list, time_limit, remaining_time_msec, show_log);
    }
#endif
    if (show_log) {
        std::cerr << "ai_common main search tl " << time_limit << std::endl;
    }
    Search_result search_result = ai_common(board, -SCORE_MAX, SCORE_MAX, MAX_LEVEL, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, time_limit, thread_id, searching);
    if (show_log) {
        std::cerr << "ai_time_limit selected " << idx_to_coord(search_result.policy) << " value " << search_result.value << " depth " << search_result.depth << "@" << search_result.probability << "%" << " time " << tim() - strt << " " << board.to_str() << std::endl << std::endl;
    }
    return search_result;
}

/*
    @brief Search for analyze command

    @param board                board to solve
    @param level                level of AI
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/

Analyze_result ai_analyze(Board board, int level, bool use_multi_thread, uint_fast8_t played_move) {
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    bool searching = true;
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
    if (mpc_level != MPC_100_LEVEL) {
        uint64_t clog_strt = tim();
        clogs = first_clog_search(board, &clog_nodes, clog_depth, board.get_legal(), &searching);
        clog_time = tim() - clog_strt;
    }
    Search search(&board, mpc_level, use_multi_thread, false);
    uint64_t strt = tim();
    //thread_pool.tell_start_using();
    Analyze_result res = first_nega_scout_analyze(&search, -SCORE_MAX, SCORE_MAX, depth, is_end_search, clogs, clog_depth, played_move, strt, &searching);
    //thread_pool.tell_finish_using();
    return res;
}



Search_result ai_accept_loss(Board board, int level, int acceptable_loss) {
    uint64_t strt = tim();
    Flip flip;
    int v = SCORE_UNDEFINED;
    uint64_t legal = board.get_legal();
    std::vector<std::pair<int, int>> moves;
    //thread_pool.tell_start_using();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            int g = -ai(board, level, true, 0, true, false).value;
        board.undo_board(&flip);
        v = std::max(v, g);
        moves.emplace_back(std::make_pair(g, cell));
    }
    //thread_pool.tell_finish_using();
    std::vector<std::pair<int, int>> acceptable_moves;
    for (std::pair<int, int> move: moves) {
        if (move.first >= v - acceptable_loss)
            acceptable_moves.emplace_back(move);
    }
    int rnd_idx = myrandrange(0, (int)acceptable_moves.size());
    int use_policy = acceptable_moves[rnd_idx].second;
    int use_value = acceptable_moves[rnd_idx].first;
    Search_result res;
    res.depth = 1;
    res.nodes = 0;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.policy = use_policy;
    res.value = use_value;
    res.is_end_search = board.n_discs() == HW2 - 1;
    res.probability = SELECTIVITY_PERCENTAGE[MPC_100_LEVEL];
    return res;
}

std::vector<std::pair<int, Board>> get_legal_moves_and_representative_board(Board board) {
    std::vector<std::pair<int, Board>> legal_moves_and_representative_board;
    uint64_t legal = board.get_legal();
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            Board unique_board = representative_board(board);
            legal_moves_and_representative_board.emplace_back(std::make_pair((int)cell, unique_board));
        board.undo_board(&flip);
    }
    return legal_moves_and_representative_board;
}

std::vector<int> get_symmetry_moves_including_self(std::vector<std::pair<int, Board>> legal_moves_and_representative_board, Board board, int policy) {
    Flip flip;
    calc_flip(&flip, &board, policy);
    Board moved_unique_board = representative_board(board.move_copy(&flip));
    std::vector<int> res;
    for (std::pair<int, Board> &elem: legal_moves_and_representative_board) {
        if (elem.second == moved_unique_board) {
            res.emplace_back(elem.first);
        }
    }
    return res;
}

void ai_hint(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_display, double values[], int hint_types[]) {
    std::vector<std::pair<int, Board>> legal_moves_and_representative_board = get_legal_moves_and_representative_board(board);
    uint64_t legal = board.get_legal();
    if (use_book) {
        std::vector<Book_value> links = book.get_all_moves_with_value(&board);
        for (Book_value &link: links) {
            values[link.policy] = link.value;
            hint_types[link.policy] = AI_TYPE_BOOK;
            legal ^= 1ULL << link.policy;
            --n_display;
        }
    }
    //thread_pool.tell_start_using();
    for (int search_level = 1; search_level <= level && global_searching; ++search_level) {
        //if (show_log) {
        //    std::cerr << "hint level " << search_level << " calculating" << std::endl;
        //}
        uint64_t search_legal = legal;
        for (int i = 0; i < n_display && search_legal && global_searching; ++i) {
            Search_result elem = ai_legal(board, search_level, use_book, book_acc_level, use_multi_thread, false, search_legal);
            if (global_searching && (search_legal & (1ULL << elem.policy))) {
                search_legal ^= 1ULL << elem.policy;
                values[elem.policy] = elem.value;
                if (elem.is_end_search) {
                    hint_types[elem.policy] = elem.probability;
                } else{
                    hint_types[elem.policy] = search_level;
                }
            }
        }
        for (std::pair<int, Board> &elem: legal_moves_and_representative_board) {
            int cell = elem.first;
            std::vector<int> symmetry_moves = get_symmetry_moves_including_self(legal_moves_and_representative_board, board, cell);
            if (symmetry_moves.size() > 1) {
                double avg = 0.0;
                std::vector<int> symmetry_same_level_moves;
                for (int symmetry_move: symmetry_moves) {
                    if (hint_types[symmetry_move] == hint_types[cell]) {
                        symmetry_same_level_moves.emplace_back(symmetry_move);
                        avg += values[symmetry_move];
                    }
                }
                avg /= symmetry_same_level_moves.size();
                if (symmetry_same_level_moves.size() > 1) {
                    for (int symmetry_move: symmetry_same_level_moves) {
                        values[symmetry_move] = avg;
                    }
                }
            }
        }
    }
    //thread_pool.tell_finish_using();
}

double selfplay_and_analyze(Board board, int level, bool show_log, thread_id_t thread_id, double before_val, bool *searching) {
    uint64_t strt = tim();
    // selfplay
    std::vector<Board> boards;
    Flip flip;
    int score_sgn = -1;
    while (*searching) {
        if (board.get_legal() == 0ULL) {
            board.pass();
            score_sgn *= -1;
            if (board.get_legal() == 0ULL) {
                if (show_log) {
                    std::cerr << " result " << score_sgn * board.score_player();
                }
                break;
            }
        }
        boards.emplace_back(board);
        if (*searching) {
            Search_result search_result = ai_searching_thread_id(board, level, false, 0, true, false, thread_id, searching);
            if (*searching && is_valid_policy(search_result.policy)) {
                if (show_log) {
                    std::cerr << idx_to_coord(search_result.policy);
                }
                if (board.n_discs() >= HW2 - 21 && boards.size()) { // complete search with last 21 empties in lv.17- (initial level)
                    if (show_log) {
                        std::cerr << "... result " << score_sgn * search_result.value;
                    }
                    break;
                }
                calc_flip(&flip, &board, search_result.policy);
                board.move_board(&flip);
                score_sgn *= -1;
            }
        }
        if (!(*searching)) {
            if (show_log) {
                std::cerr << " terminated" << std::endl;
            }
            break;
        }
    }
    if (!(*searching)) {
        return SCORE_UNDEFINED;
    }
    if (show_log) {
        std::cerr << " selfplay " << tim() - strt << " ms";
    }
    // analyze
    double res = SCORE_UNDEFINED;
    for (int i = boards.size() - 1; i >= 0; --i) {
        if (*searching) {
            transposition_table.del(&boards[i], boards[i].hash());
            Search_result search_result = ai_searching_thread_id(boards[i], level, false, 0, true, false, thread_id, searching);
            if (*searching) {
                int v = search_result.value;
                if (i == 0) {
                    if (show_log) {
                        std::cerr << " analyzed raw " << -v;
                    }
                    if (search_result.is_end_search) {
                        res = -v;
                    } else {
                        res = (0.9 * before_val + 1.1 * -v) / 2.0;
                    }
                }
            }
        }
        if (!(*searching)) {
            if (show_log) {
                std::cerr << " terminated" << std::endl;
            }
            break;
        }
    }
    return res;
}




Search_result selfplay_and_analyze_search_result(Board board, int level, bool show_log, thread_id_t thread_id, bool *searching) {
    uint64_t strt = tim();
    Search_result res;
    // selfplay
    std::vector<Board> boards;
    Flip flip;
    int score_sgn = -1;
    while (*searching) {
        if (board.get_legal() == 0ULL) {
            board.pass();
            score_sgn *= -1;
            if (board.get_legal() == 0ULL) {
                if (show_log) {
                    std::cerr << " result " << score_sgn * board.score_player();
                }
                break;
            }
        }
        boards.emplace_back(board);
        if (*searching) {
            Search_result search_result = ai_searching_thread_id(board, level, false, 0, true, false, thread_id, searching);
            if (*searching && is_valid_policy(search_result.policy)) {
                if (show_log) {
                    std::cerr << idx_to_coord(search_result.policy);
                }
                if (board.n_discs() >= HW2 - 21 && boards.size()) { // complete search with last 21 empties in lv.17- (initial level)
                    if (show_log) {
                        std::cerr << "... result " << score_sgn * search_result.value;
                    }
                    break;
                }
                calc_flip(&flip, &board, search_result.policy);
                board.move_board(&flip);
                score_sgn *= -1;
            }
        }
        if (!(*searching)) {
            if (show_log) {
                std::cerr << " terminated" << std::endl;
            }
            break;
        }
    }
    if (!(*searching)) {
        return res;
    }
    if (show_log) {
        std::cerr << " selfplay " << tim() - strt << " ms";
    }
    // analyze
    for (int i = boards.size() - 1; i >= 0; --i) {
        if (*searching) {
            transposition_table.del(&boards[i], boards[i].hash());
            Search_result search_result = ai_searching_thread_id(boards[i], level, false, 0, true, false, thread_id, searching);
            if (*searching) {
                int v = search_result.value;
                if (i == 0) {
                    if (show_log) {
                        std::cerr << " analyzed raw " << -v;
                    }
                    res.value = -v;
                    res.policy = search_result.policy;
                }
            }
        }
        if (!(*searching)) {
            if (show_log) {
                std::cerr << " terminated" << std::endl;
            }
            break;
        }
    }
    std::cerr << std::endl;
    return res;
}


bool comp_ponder_elem(Ponder_elem &a, Ponder_elem &b) {
    if (a.count == b.count) {
        return a.value > b.value;
    }
    return a.count > b.count;
}

std::vector<Ponder_elem> ai_ponder(Board board, bool show_log, thread_id_t thread_id, bool *searching) {
    uint64_t strt = tim();
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            if (show_log) {
                std::cerr << "no ponder needed because of game over" << std::endl;
            }
            std::vector<Ponder_elem> empty_list;
            return empty_list;
        } else {
            std::cerr << "ponder pass found" << std::endl;
        }
    }
    const int canput = pop_count_ull(legal);
    std::vector<Ponder_elem> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &board, cell);
        move_list[idx].value = INF;
        move_list[idx].count = 0;
        move_list[idx].level = 0;
        move_list[idx].depth = 0;
        move_list[idx].mpc_level = MPC_74_LEVEL;
        move_list[idx].is_endgame_search = false;
        move_list[idx].is_complete_search = false;
        ++idx;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    int n_searched_all = 0;
    while (*searching) {
        bool all_complete = true;
        for (int i = 0; i < canput; ++i) {
            all_complete &= move_list[i].is_complete_search;
        }
        if (all_complete) {
            if (show_log) {
                std::cerr << "ponder completely searched" << std::endl;
            }
            break;
        }

        int selected_idx = -1;
        double max_ucb = -INF - 1;
        for (int i = 0; i < canput; ++i) {
            double ucb = -INF;
            if (n_searched_all == 0) { // not searched at all
                ucb = move_list[i].value;
            } else if (move_list[i].count == 0) { // this node is not searched
                ucb = INF;
            } else if (move_list[i].is_complete_search) { // fully searched
                ucb = -INF;
            } else {
                //double depth_weight = (double)std::min(10, move_list[i].depth) / (double)std::min(10, max_depth);
                ucb = move_list[i].value / (double)HW2 + PONDER_UCB_COE * sqrt(log(2.0 * (double)n_searched_all) / (double)move_list[i].count);
            }
            if (ucb > max_ucb) {
                selected_idx = i;
                max_ucb = ucb;
            }
        }
        //std::cerr << std::endl;
        if (selected_idx == -1) {
            if (show_log) {
                std::cerr << "ponder: no move selected n_moves " << canput << std::endl;
            }
            break;
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int max_depth = HW2 - n_board.n_discs();
        int level = move_list[selected_idx].level + 1;
        Search_result search_result = ai_searching_thread_id(n_board, level, false, 0, true, false, thread_id, searching);
        double v = -search_result.value;
        if (PONDER_USE_SELFPLAY && move_list[selected_idx].depth >= PONDER_START_SELFPLAY_DEPTH && !move_list[selected_idx].is_endgame_search) { // selfplay
            double max_value = -INF;
            for (int i = 0; i < canput; ++i) {
                max_value = std::max(max_value, move_list[i].value);
            }
            if (v >= max_value - 3.25 && level >= 17) {
                // std::cerr << "ponder selfplay " << idx_to_coord(move_list[selected_idx].flip.pos) << " depth " << new_depth << std::endl;
                double selfplay_val = selfplay_and_analyze(n_board, level, false, thread_id, v, searching);
                if (selfplay_val != SCORE_UNDEFINED) {
                    v = selfplay_val;
                }
            }
        }
        if (*searching) {
            move_list[selected_idx].level = level;
            bool is_mid_search;
            get_level(move_list[selected_idx].level, board.n_discs() - 4, &is_mid_search, &move_list[selected_idx].depth, &move_list[selected_idx].mpc_level);
            move_list[selected_idx].is_endgame_search = !is_mid_search;
            move_list[selected_idx].is_complete_search = !is_mid_search && move_list[selected_idx].mpc_level == MPC_100_LEVEL;
            move_list[selected_idx].response_board_key = n_board.to_str();
            move_list[selected_idx].response = search_result;
            ++move_list[selected_idx].count;
            if (move_list[selected_idx].value == INF || !is_mid_search) {
                move_list[selected_idx].value = v;
            } else {
                move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
            }
            ++n_searched_all;
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_ponder_elem);
    // if (show_log && n_searched_all) {
    //     std::cerr << "ponder loop " << n_searched_all << " in " << tim() - strt << " ms" << std::endl;
    //     std::cerr << "ponder board " << board.to_str() << std::endl;
    //     for (int i = 0; i < canput; ++i) {
    //         std::cerr << "pd " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
    //         std::cerr << " count " << move_list[i].count << " level " << move_list[i].level << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
    //         if (move_list[i].is_complete_search) {
    //             std::cerr << " complete";
    //         } else if (move_list[i].is_endgame_search) {
    //             std::cerr << " endgame";
    //         }
    //         std::cerr << std::endl;
    //     }
    //     std::cerr << std::endl;
    // }
    return move_list;
}

void print_ponder_result(std::vector<Ponder_elem> move_list) {
    std::cerr << "ponder result" << std::endl;
    for (int i = 0; i < move_list.size(); ++i) {
        std::cerr << "pd " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
        std::cerr << " count " << move_list[i].count << " level " << move_list[i].level << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
        if (move_list[i].is_complete_search) {
            std::cerr << " complete";
        } else if (move_list[i].is_endgame_search) {
            std::cerr << " endgame";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

bool comp_get_values_elem(Ponder_elem &a, Ponder_elem &b) {
    if (a.value == b.value) {
        if (a.depth == b.depth) {
            return a.mpc_level > b.mpc_level;
        }
        return a.depth > b.depth;
    }
    return a.value > b.value;
}

std::vector<Ponder_elem> ai_get_values(Board board, bool show_log, uint64_t time_limit, thread_id_t thread_id) {
    uint64_t strt = tim();
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            if (show_log) {
                std::cerr << "get values game overgame over" << std::endl;
            }
            std::vector<Ponder_elem> empty_list;
            return empty_list;
        } else {
            std::cerr << "get values pass found" << std::endl;
        }
    }
    const int canput = pop_count_ull(legal);
    std::vector<Ponder_elem> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &board, cell);
        move_list[idx].value = INF;
        move_list[idx].count = 0;
        move_list[idx].depth = 0;
        move_list[idx].mpc_level = MPC_74_LEVEL;
        move_list[idx].is_endgame_search = false;
        move_list[idx].is_complete_search = false;
        ++idx;
    }
    uint64_t tl_per_move = time_limit / canput;
    if (show_log) {
        std::cerr << "get values tl per move " << tl_per_move << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    for (Ponder_elem &elem: move_list) {
        uint64_t elem_strt = tim();
        while (tim() - elem_strt < tl_per_move && !elem.is_complete_search && global_searching) {
            Board n_board = board.copy();
            n_board.move_board(&elem.flip);
            int max_depth = HW2 - n_board.n_discs();
            int new_depth = elem.depth + 1;
            uint_fast8_t new_mpc_level = elem.mpc_level;
            if (new_depth > max_depth) {
                new_depth = max_depth;
                ++new_mpc_level;
            } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
                new_depth = max_depth;
            }
            bool new_is_end_search = (new_depth == max_depth);
            bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
            Search search(&n_board, new_mpc_level, true, false);
            search.thread_id = thread_id;
            bool n_searching = true;
            uint64_t time_limit_this_search = get_this_search_time_limit(tl_per_move, tim() - elem_strt);
            std::future<int> v_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &n_searching);
            if (v_future.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
                int v = -v_future.get();
                if (global_searching) {
                    if (elem.value == INF || new_is_end_search) {
                        elem.value = v;
                    } else {
                        elem.value = (0.9 * elem.value + 1.1 * v) / 2.0;
                    }
                    elem.depth = new_depth;
                    elem.mpc_level = new_mpc_level;
                    elem.is_endgame_search = new_is_end_search;
                    elem.is_complete_search = new_is_complete_search;
                    ++elem.count;
                }
            } else {
                n_searching = false;
                try {
                    v_future.get();
                } catch (const std::exception &e) {
                }
            }
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_get_values searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_get_values board " << board.to_str() << std::endl;
        for (int i = 0; i < canput; ++i) {
            std::cerr << "gb " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

std::vector<Ponder_elem> ai_align_move_levels(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id, int aligned_min_level) {
    uint64_t strt = tim();
    if (show_log) {
        std::cerr << "align levels tl " << time_limit << " n_good_moves " << n_good_moves << " out of " << move_list.size() << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    while (tim() - strt < time_limit) {
        int min_depth = 100;
        for (int i = 0; i < n_good_moves; ++i) {
            min_depth = std::min(min_depth, move_list[i].depth);
        }
        if (min_depth >= aligned_min_level) {
            std::cerr << "min depth >= " << aligned_min_level << std::endl;
            break;
        }
        bool level_aligned = true;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].depth != move_list[0].depth || move_list[i].mpc_level != move_list[0].mpc_level) {
                level_aligned = false;
                break;
            }
        }
        if (level_aligned && min_depth >= aligned_min_level) {
            if (show_log) {
                std::cerr << "level aligned & min depth >= " << aligned_min_level << std::endl;
            }
            break;
        }
        int min_depth2 = INF;
        uint_fast8_t min_mpc_level = 100;
        int selected_idx = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].depth < min_depth2) {
                min_depth2 = move_list[i].depth;
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            } else if (move_list[i].depth == min_depth2 && move_list[i].mpc_level < min_mpc_level) {
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            }
        }
        if (move_list[selected_idx].is_complete_search) {
            if (show_log) {
                std::cerr << "completely searched" << std::endl;
            }
            break;
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int max_depth = HW2 - n_board.n_discs();
        int new_depth = move_list[selected_idx].depth + 1;
        uint_fast8_t new_mpc_level = move_list[selected_idx].mpc_level;
        if (new_depth > max_depth) {
            new_depth = max_depth;
            ++new_mpc_level;
        } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
            new_depth = max_depth;
        }
        bool new_is_end_search = (new_depth == max_depth);
        bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
        Search search(&n_board, new_mpc_level, true, false);
        search.thread_id = thread_id;
        bool n_searching = true;
        uint64_t time_limit_this_search = get_this_search_time_limit(time_limit, tim() - strt);
        std::future<int> v_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &n_searching);
        if (v_future.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
            int v = -v_future.get();
            if (global_searching) {
                if (move_list[selected_idx].value == INF || new_is_end_search) {
                    move_list[selected_idx].value = v;
                } else {
                    move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
                }
                // double max_value = -INF;
                // for (int i = 0; i < n_good_moves; ++i) {
                //     max_value = std::max(max_value, move_list[i].value);
                // }
                // // additional selfplay & analyze
                // if (move_list[selected_idx].value >= max_value - 1.0) {
                //     int level = std::min(21, get_level_from_depth_mpc_level(n_board.n_discs(), new_depth, new_mpc_level));
                //     bool n_searching2 = true;
                //     if (show_log) {
                //         std::cerr << "try selfplay " << idx_to_coord(move_list[selected_idx].flip.pos) << " level " << level << " val " << move_list[selected_idx].value << " max " << max_value << " ";
                //     }
                //     std::future<double> selfplay_future = std::async(std::launch::async, selfplay_and_analyze, n_board, level, show_log, thread_id, move_list[selected_idx].value, &n_searching2);
                //     uint64_t time_limit_selfplay = get_this_search_time_limit(time_limit, tim() - strt);
                //     if (selfplay_future.wait_for(std::chrono::milliseconds(time_limit_selfplay)) == std::future_status::ready) {
                //         double selfplay_val = selfplay_future.get();
                //         if (selfplay_val != SCORE_UNDEFINED) {
                //             if (show_log) {
                //                 std::cerr << " selfplay success " << move_list[selected_idx].value << " & " << selfplay_val;
                //             }
                //             move_list[selected_idx].value = (1.2 * move_list[selected_idx].value + 0.8 * selfplay_val) / 2.0;
                //             if (show_log) {
                //                 std::cerr << " -> " << move_list[selected_idx].value << std::endl;
                //             }
                //         }
                //     } else {
                //         n_searching2 = false;
                //         try {
                //             selfplay_future.get();
                //         } catch (const std::exception &e) {
                //         }
                //     }
                // }
                move_list[selected_idx].depth = new_depth;
                move_list[selected_idx].mpc_level = new_mpc_level;
                move_list[selected_idx].is_endgame_search = new_is_end_search;
                move_list[selected_idx].is_complete_search = new_is_complete_search;
                ++move_list[selected_idx].count;
            }
        } else {
            n_searching = false;
            try {
                v_future.get();
            } catch (const std::exception &e) {
            }
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_align_move_levels searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_align_move_levels board " << board.to_str() << std::endl;
        for (int i = 0; i < n_good_moves; ++i) {
            std::cerr << "ag " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

std::vector<Ponder_elem> ai_additional_selfplay(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, double threshold, uint64_t time_limit, thread_id_t thread_id) {
    uint64_t strt = tim();
    if (show_log) {
        std::cerr << "additional selfplay tl " << time_limit << " n_good_moves " << n_good_moves << " out of " << move_list.size() << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    const int initial_level = 21;
    constexpr int n_same_level = 1;
    std::vector<int> levels;
    for (int i = 0; i < n_good_moves; ++i) {
        levels.emplace_back(initial_level * n_same_level);
    }
    std::vector<bool> is_first_searches;
    for (int i = 0; i < n_good_moves; ++i) {
        is_first_searches.emplace_back(true);
    }
    // selfplay & analyze
    std::vector<Clog_result> clogs;
    std::vector<Board> n_boards;
    Flip flip;
    while (tim() - strt < time_limit) {
        double first_val = -INF, second_val = -INF;
        int first_level = -1, second_level = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].value > first_val) {
                second_val = first_val;
                first_val = move_list[i].value;
                first_level = (levels[i] - 1) / n_same_level;
            } else if (move_list[i].value > second_val) {
                second_val = move_list[i].value;
                second_level = (levels[i] - 1) / n_same_level;
            }
        }
        if (
            (first_val - second_val > threshold * 1.114 && first_level >= 25 && second_level >= 25) || 
            first_val - second_val > threshold * 1.686
        ) {
            if (show_log) {
                std::cerr << "enough differences found first " << first_val << "@lv." << first_level << " second " << second_val << "@lv." << second_level << std::endl;
            }
            break;
        }
        double max_val = -INF;
        int selected_idx = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            //if (!move_list[i].is_complete_search) {
            if (!(move_list[i].is_endgame_search && move_list[i].mpc_level >= MPC_99_LEVEL)) {
                if (levels[i] == initial_level * n_same_level) {
                    selected_idx = i;
                    break;
                } else {
                    double val = move_list[i].value + myrandom() * threshold * 2.0 + (double)(60 - initial_level - levels[i] / n_same_level) * 0.5; // 2 level for 1 score
                    if (val > max_val) {
                        max_val = val;
                        selected_idx = i;
                    }
                }
            }
        }
        if (selected_idx == -1) {
            if (show_log) {
                std::cerr << "enough searched" << std::endl;
            }
            break;
        }
        int level = levels[selected_idx] / n_same_level;
        std::cerr << "move " << idx_to_coord(move_list[selected_idx].flip.pos) << " selfplay lv." << level << " ";
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        uint64_t tl_selfplay = 1;
        uint64_t elapsed_now = tim() - strt;
        if (elapsed_now < time_limit) {
            tl_selfplay = time_limit - elapsed_now;
        }
        bool searching = true;
        std::future<double> selfplay_future = std::async(std::launch::async, selfplay_and_analyze, n_board, level, true, thread_id, move_list[selected_idx].value, &searching);
        if (selfplay_future.wait_for(std::chrono::milliseconds(tl_selfplay)) == std::future_status::ready) {
            double selfplay_val = selfplay_future.get();
            if (selfplay_val != SCORE_UNDEFINED) {
                move_list[selected_idx].value = selfplay_val;
                int depth;
                uint_fast8_t mpc_level;
                bool is_mid_search;
                get_level(level, n_board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
                move_list[selected_idx].depth = depth;
                move_list[selected_idx].mpc_level = mpc_level;
                move_list[selected_idx].is_endgame_search = (HW2 - n_board.n_discs()) <= depth;
                move_list[selected_idx].is_complete_search = move_list[selected_idx].is_endgame_search && mpc_level == MPC_100_LEVEL;
                ++move_list[selected_idx].count;
                std::cerr << " value " << move_list[selected_idx].value << " depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "% " << tim() - strt << " ms" << std::endl;
                is_first_searches[selected_idx] = false;
                ++levels[selected_idx];
            }
        } else {
            searching = false;
            selfplay_future.get();
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_additional_selfplay searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_additional_selfplay board " << board.to_str() << std::endl;
        for (int i = 0; i < n_good_moves; ++i) {
            std::cerr << "sm " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

Search_result ai_range(Board board, int level, int score_min, int score_max, bool *searching) {
    std::vector<Search_result> moves_in_range;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal && *searching; cell = next_bit(&legal)) {
        Flip flip;
        bool passed = false;
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            if (board.is_end()) {
                board.undo_board(&flip);
                continue;
            }
            if (board.get_legal() == 0) {
                board.pass();
                passed = true;
            }
            int alpha = passed ? score_min : -score_max;
            int beta = passed ? score_max : -score_min;
            alpha = std::min(-SCORE_MAX, alpha - 1);
            beta = std::max(SCORE_MAX, beta + 1);
            Search_result res = ai_window_legal_searching(board, alpha, beta, level, true, 0, true, false, board.get_legal(), searching);
            if (res.value >= score_min && res.value <= score_max) {
                Search_result move_res = res;
                move_res.policy = cell;
                moves_in_range.push_back(move_res);
            }
            if (passed) {
                board.pass();
            }
        board.undo_board(&flip);
    }
    if (!moves_in_range.empty()) {
        return moves_in_range[myrandrange(0, (int)moves_in_range.size())];
    } else {
        Search_result res;
        res.value = SCORE_UNDEFINED;
        res.policy = MOVE_UNDEFINED;
        return res;
    }
}
