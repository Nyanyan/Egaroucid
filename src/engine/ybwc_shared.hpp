/*
    Egaroucid Project
    Persistent YBWC workers sharing the remaining siblings and search window.
    SPDX-License-Identifier: GPL-3.0-or-later
*/
#pragma once

#include "ybwc_split_point.hpp"

// Everything a helper or a joiner needs to search the moves of a split. It
// lives in the owner's frame, which outlives every worker of the split.
struct Ybwc_join_context {
    const Search *seed;
    Flip_value *moves;
    int depth;
    bool is_end_search;
    bool pv;
    Search_node_type node_type;
    const Search_cancellation_context *cancellation;
};

// Each worker owns its Search and one claimed move at a time. The move list
// stays immutable until every worker has retired.
inline void ybwc_shared_worker(
    Search *search, Ybwc_split_point &split, Flip_value *moves, int depth,
    bool is_end_search, bool pv, Search_node_type node_type,
    const Search_cancellation_context &cancellation, bool helper,
    Ybwc_split_point::Work first = {}
) {
    auto work = first ? first : split.take();
    while (work && global_searching && is_searching(cancellation)) {
        const Flip_value &move = moves[work.move_index];
        const bool end_move = ybwc_use_endsearch_move(search, depth - 1, is_end_search);
        const Nws_node_hint hint = child_nws_hint(move, is_end_search);
        ybwc_move_child(search, &move.flip, end_move);
        int g;
        const int reduction = !pv && !helper
            ? mid_nws_lmr_reduction(search, depth, work.ordinal, is_end_search) : 0;
        g = -nega_alpha_ordering_nws(search, -work.alpha - 1, depth - 1 - reduction,
            hint, move.n_legal, is_end_search, cancellation);
        if (reduction && g > work.alpha && is_searching(cancellation)) {
            g = -nega_alpha_ordering_nws(search, -work.alpha - 1, depth - 1,
                hint, move.n_legal, is_end_search, cancellation);
        }

        auto bound = g <= work.alpha ? Ybwc_split_point::Bound::upper : Ybwc_split_point::Bound::lower;
        if (pv) {
            while (g > work.alpha && g < split.beta() && global_searching && is_searching(cancellation)) {
                const int current_alpha = split.alpha();
                if (g <= current_alpha) {
                    // A different sibling improved alpha while this one was
                    // running. Its old lower bound cannot eliminate this move.
                    work.alpha = current_alpha;
                    g = -nega_alpha_ordering_nws(search, -work.alpha - 1, depth - 1,
                        hint, move.n_legal, is_end_search, cancellation);
                    bound = g <= work.alpha ? Ybwc_split_point::Bound::upper : Ybwc_split_point::Bound::lower;
                } else {
                    // The completed probe proved g as this move's lower bound.
                    // Endsearch-only move updates did not update eval features,
                    // whereas full-window search needs those features again.
                    if (end_move) {
                        ybwc_undo_child(search, &move.flip, true);
                        search->move(&move.flip);
                    }
                    // Pass the whole chain: the split's own flag is not
                    // cleared when an ancestor search is stopped.
                    g = -nega_scout_node(search, -split.beta(), -g, depth - 1,
                        false, move.n_legal, is_end_search, search_child_node_type(node_type),
                        cancellation);
                    if (end_move) {
                        search->undo(&move.flip);
                        ybwc_move_child(search, &move.flip, true);
                    }
                    bound = g >= split.beta() ? Ybwc_split_point::Bound::lower : Ybwc_split_point::Bound::exact;
                    break;
                }
            }
        }
        ybwc_undo_child(search, &move.flip, end_move);
        if (!global_searching || !is_searching(cancellation)) break;
        // Undefined values can be negated while propagating out of a search;
        // they are never a bound even if the immediate local flag is still set.
        if (g < -SCORE_MAX || g > SCORE_MAX) {
            split.cancel();
            break;
        }
        const bool published = split.publish(work, g, move.flip.pos, bound);
        if (!published && split.searching()) {
            // Every fail-high below beta must have been fully re-searched.
            std::terminate();
        }
        work = split.take();
    }
}

// Searches moves of split on a copy of its seed until none are left, then
// retires from the split. Used for helpers and joiners alike.
inline void ybwc_run_shared_worker(Ybwc_split_point &split, const Ybwc_join_context &context) {
    uint64_t nodes;
    {
        const Ybwc_current_split_scope scope(&split);
        Search local = *context.seed;
        local.n_nodes = 0;
        ybwc_shared_worker(&local, split, context.moves, context.depth, context.is_end_search,
            context.pv, context.node_type, *context.cancellation, true);
        nodes = local.n_nodes;
    }
    split.worker_done(nodes);
}

// Workers a split may have at once, the owner included.
inline int ybwc_join_max_workers(const Ybwc_split_point &split) {
    const auto *context = static_cast<const Ybwc_join_context *>(split.join_context());
    return context != nullptr && !context->is_end_search ? YBWC_MID_JOIN_MAX_WORKERS : YBWC_JOIN_MAX_WORKERS;
}

// Requires split's mutex.
template <typename Claim>
inline bool ybwc_try_admit_joiner_locked(Ybwc_split_point &split, Claim &claim) {
    return split.try_admit_joiner_locked(ybwc_join_max_workers(split), claim);
}

// Requires node's mutex. Children come before grandchildren, so a joiner
// prefers the largest subtrees still being searched below node.
template <typename Claim>
inline Ybwc_split_point *ybwc_find_joinable_locked(Ybwc_split_point &node, int levels, Claim &claim) {
    for (Ybwc_split_point *child = node.first_child_locked(); child; child = child->next_sibling_locked()) {
        std::lock_guard lock(child->mutex());
        if (ybwc_try_admit_joiner_locked(*child, claim)) return child;
    }
    if (levels <= 1) return nullptr;
    for (Ybwc_split_point *child = node.first_child_locked(); child; child = child->next_sibling_locked()) {
        std::lock_guard lock(child->mutex());
        if (Ybwc_split_point *found = ybwc_find_joinable_locked(*child, levels - 1, claim)) return found;
    }
    return nullptr;
}

// Lets the owner of split, which is only waiting for its helpers, search
// moves of a split below them. It returns once that split has no moves
// left for it. Every result the owner waits for depends on that subtree,
// so helping never delays the owner beyond its own completion.
inline bool ybwc_help_descendant(Ybwc_split_point &split) {
    Ybwc_split_point *target;
    auto no_claim = [](const void *) { return true; };
    {
        std::lock_guard lock(split.mutex());
        target = ybwc_find_joinable_locked(split, YBWC_JOIN_MAX_LEVELS, no_claim);
    }
    if (target == nullptr) return false;
    YBWC_IDLE_COUNT(ybwc_shared_joins);
    ybwc_run_shared_worker(*target, *static_cast<const Ybwc_join_context *>(target->join_context()));
    return true;
}

#if USE_YBWC_IDLE_JOIN
inline int ybwc_split_task_limit(const Ybwc_join_context &context) {
    return context.is_end_search ? YBWC_END_MAX_SPLIT_TASKS : context.seed->mid_split_task_limit;
}

// Run by an idle pool worker before it sleeps: joins a split anywhere in the
// registered trees, like a helper pushed there late. Without it, a split
// created while every thread was busy keeps its unclaimed moves for its own
// few workers even after other threads become free.
inline bool ybwc_idle_join() noexcept {
    const Ybwc_join_context *context = nullptr;
    auto claim = [&context](const void *join_context) {
        const auto *c = static_cast<const Ybwc_join_context *>(join_context);
        if (!thread_pool.claim_idle_worker(c->seed->thread_id, ybwc_split_task_limit(*c))) return false;
        context = c;
        return true;
    };
    Ybwc_split_point *target = nullptr;
    {
        std::lock_guard registry_lock(ybwc_root_registry.mutex());
        for (int i = 0; i < ybwc_root_registry.size_locked() && target == nullptr; ++i) {
            Ybwc_split_point *root = ybwc_root_registry.at_locked(i);
            std::lock_guard lock(root->mutex());
            if (ybwc_try_admit_joiner_locked(*root, claim)) {
                target = root;
            } else {
                target = ybwc_find_joinable_locked(*root, YBWC_JOIN_MAX_LEVELS, claim);
            }
        }
    }
    if (target == nullptr) return false;
    YBWC_IDLE_COUNT(ybwc_idle_joins);
    // A notification wakes one idle thread. Wake the next one in case more
    // work is left for it.
    thread_pool.notify_idle_work();
    // target and context may be gone once this worker has retired.
    const thread_id_t thread_id = context->seed->thread_id;
    const int task_limit = ybwc_split_task_limit(*context);
    ybwc_run_shared_worker(*target, *context);
    thread_pool.release_idle_worker(thread_id, task_limit);
    return true;
}

inline void ybwc_notify_idle_join() {
    thread_pool.notify_idle_work();
}

// Not inline: an unused inline variable need not be initialized at all.
static const bool ybwc_idle_join_installed = [] {
    thread_pool.set_idle_work(&ybwc_idle_join);
    ybwc_joinable_notifier.store(&ybwc_notify_idle_join, std::memory_order_relaxed);
    ybwc_root_registry_enabled.store(true, std::memory_order_relaxed);
    return true;
}();
#endif

inline bool ybwc_search_shared(
    Search *search, int *alpha, int beta, int *value, int *best_move,
    int depth, bool is_end_search, Flip_value *moves, int count,
    Search_node_type node_type, bool pv, const Search_cancellation_context &parent
) {
    if (!global_searching || !is_searching(parent)) return false;
#if USE_YBWC_HELPFUL_MASTER
    // Without free helpers the moves would be searched serially, invisible to
    // an owner above that has nothing to do but wait for this subtree. Share
    // them with that owner instead.
    const bool waiting_owner_above = ybwc_current_split != nullptr &&
        ybwc_current_split->has_waiting_owner(YBWC_JOIN_MAX_LEVELS);
#else
    constexpr bool waiting_owner_above = false;
#endif
    if (thread_pool.get_n_idle() <= 0 && !waiting_owner_above) return false;
    Ybwc_split_point split(*alpha, beta, *value, *best_move, pv);
    for (int i = 0; i < count; ++i) if (moves[i].flip.flip) split.add_move(i);
    const int child_depth = depth - 1;
    const int min_younger = is_end_search
        ? (YBWC_END_MIN_REMAINING_MOVES > 0 ? YBWC_END_MIN_REMAINING_MOVES
            : (child_depth <= YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH
                ? YBWC_END_LOW_DEPTH_N_YOUNGER_CHILD : YBWC_END_N_YOUNGER_CHILD))
        : (child_depth <= YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD_MAX_DEPTH
            ? YBWC_MID_LOW_DEPTH_N_YOUNGER_CHILD : YBWC_MID_N_YOUNGER_CHILD);
    if (split.move_count() <= min_younger) return false;

    // Reserve work for the parent before helpers can claim the remaining moves.
    const auto parent_work = split.take();
    const Search seed = *search;
    const Search_cancellation_context cancellation{split.cancellation_flag(), &parent};
    const Ybwc_join_context join_context{&seed, moves, depth, is_end_search, pv, node_type, &cancellation};
    split.set_join_context(&join_context, YBWC_JOIN_MIN_FAIL_LOW);
    const int task_limit = is_end_search ? YBWC_END_MAX_SPLIT_TASKS : search->mid_split_task_limit;
    const int helper_limit = std::min(3, split.move_count() - 1);
    int submitted = 0;
    for (int i = 0; i < helper_limit && split.searching(); ++i) {
        bool pushed = false;
        auto task = [&]() noexcept {
            try {
                ybwc_run_shared_worker(split, join_context);
            } catch (...) { std::terminate(); }
        };
        split.register_worker();
        if (task_limit == THREAD_SIZE_INF) {
            auto ignored = thread_pool.push(search->thread_id, &pushed, task);
        } else {
            auto ignored = thread_pool.push(search->thread_id, task_limit, &pushed, task);
        }
        if (!pushed) {
            split.unregister_worker();
            break;
        }
        ++submitted;
    }
    if (!submitted && !waiting_owner_above) return false;
    YBWC_IDLE_COUNT(ybwc_shared_splits);
#if YBWC_IDLE_PROFILE
    ybwc_shared_helpers.fetch_add(submitted, std::memory_order_relaxed);
#endif
    split.link_to(ybwc_current_split);
    {
        const Ybwc_current_split_scope scope(&split);
        ybwc_shared_worker(search, split, moves, depth, is_end_search, pv, node_type, cancellation, false, parent_work);
        // Every move has been handed out; only helpers and joiners remain.
#if USE_YBWC_HELPFUL_MASTER
        // Set before the first look for work: a descendant that becomes
        // joinable afterwards wakes this owner, so it never sleeps past it.
        split.set_owner_waiting(true);
#endif
        for (;;) {
            const uint32_t seen = split.wake_sequence();
            if (!split.has_running_workers()) break;
#if USE_YBWC_HELPFUL_MASTER
            {
                YBWC_IDLE_SCOPE(YBWC_IDLE_OWNER_HELP);
                if (ybwc_help_descendant(split)) continue;
            }
#endif
            YBWC_IDLE_SCOPE(YBWC_IDLE_OWNER_SLEEP);
            split.wait_wake(seen);
        }
#if USE_YBWC_HELPFUL_MASTER
        split.set_owner_waiting(false);
#endif
    }
    split.unlink();
    // Helpers may still be returning from their final access to split.
    {
        YBWC_IDLE_SCOPE(YBWC_IDLE_RETIRE);
        split.wait_retired();
    }
    search->n_nodes += split.worker_nodes();
    const auto result = split.result();
    *value = result.value;
    *best_move = result.best_move;
    if (pv) *alpha = result.alpha;
    for (int i = 0; i < count; ++i) {
        if (result.completed & (uint64_t{1} << i)) moves[i].flip.flip = 0;
    }
    return true;
}
