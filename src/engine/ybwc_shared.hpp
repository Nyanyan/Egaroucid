/*
    Egaroucid Project
    Persistent YBWC workers sharing the remaining siblings and search window.
    SPDX-License-Identifier: GPL-3.0-or-later
*/
#pragma once

#include "ybwc_split_point.hpp"

struct Ybwc_shared_worker_result {
    uint64_t nodes;
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
                    g = -nega_scout_node(search, -split.beta(), -g, depth - 1,
                        false, move.n_legal, is_end_search, search_child_node_type(node_type),
                        cancellation.current);
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

inline bool ybwc_search_shared(
    Search *search, int *alpha, int beta, int *value, int *best_move,
    int depth, bool is_end_search, Flip_value *moves, int count,
    Search_node_type node_type, bool pv, const Search_cancellation_context &parent
) {
    if (thread_pool.get_n_idle() <= 0 || !global_searching || !is_searching(parent)) return false;
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
    Ybwc_completion_group<Ybwc_shared_worker_result, 3> completions;
    const int task_limit = is_end_search ? YBWC_END_MAX_SPLIT_TASKS : search->mid_split_task_limit;
    const int helper_limit = std::min(3, split.move_count() - 1);
    int submitted = 0;
    for (int i = 0; i < helper_limit && split.searching(); ++i) {
        const auto slot = completions.reserve_slot();
        bool pushed = false;
        auto task = [&, slot]() noexcept {
            try {
                Search local = seed;
                local.n_nodes = 0;
                ybwc_shared_worker(&local, split, moves, depth, is_end_search, pv, node_type, cancellation, true);
                completions.publish(slot, {local.n_nodes});
            } catch (...) { std::terminate(); }
        };
        if (task_limit == THREAD_SIZE_INF) {
            auto ignored = thread_pool.push(search->thread_id, &pushed, task);
        } else {
            auto ignored = thread_pool.push(search->thread_id, task_limit, &pushed, task);
        }
        if (!pushed) break;
        completions.mark_submitted(slot);
        ++submitted;
    }
    if (!submitted) return false;
    ybwc_shared_worker(search, split, moves, depth, is_end_search, pv, node_type, cancellation, false, parent_work);
    for (int remaining = submitted; remaining;) {
        Ybwc_shared_worker_result done;
        if (completions.try_pop(&done)) {
            search->n_nodes += done.nodes;
            --remaining;
        } else if (!(!is_end_search || YBWC_END_WAIT_HELP) || !thread_pool.try_execute_one(search->thread_id)) {
            completions.wait_for_ready();
        }
    }
    const auto result = split.result();
    *value = result.value;
    *best_move = result.best_move;
    if (pv) *alpha = result.alpha;
    for (int i = 0; i < count; ++i) {
        if (result.completed & (uint64_t{1} << i)) moves[i].flip.flip = 0;
    }
    // The completion-group destructor waits for each helper's final access,
    // including after the ready bit was published, before releasing seed/split.
    return true;
}
