/*
    Egaroucid Project
    Shared work and bounds for a persistent YBWC split point.
    SPDX-License-Identifier: GPL-3.0-or-later
*/
#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <thread>

// Synchronization is needed only when taking or completing a whole subtree.
// No lock is held while searching it (including any nested split points).
//
// Split points also form a tree: a split created while searching a move of
// another split is linked below it. An owner that has handed out all of its
// moves can then join a descendant split instead of sleeping. Each split's
// mutex guards its own moves, results and child list; the sibling links are
// guarded by the parent's mutex. Locks are only ever taken parent-first.
class Ybwc_split_point {
public:
    enum class Bound { upper, exact, lower };
    struct Work {
        int move_index = -1;
        int ordinal = 0;
        int alpha = 0;
        explicit operator bool() const { return move_index >= 0; }
    };
    struct Result {
        int alpha;
        int value;
        int best_move;
        uint64_t completed;
    };

private:
    std::mutex mutex_;
    std::array<int, 64> moves_{};
    int count_ = 0;
    int next_ = 0;
    std::atomic<int> alpha_;
    const int beta_;
    const bool pv_;
    int value_;
    int best_move_;
    uint64_t completed_ = 0;
    int n_fail_low_ = 0;
    alignas(std::atomic_ref<bool>::required_alignment) bool searching_ = true;

    // Tree links, see the class comment. parent_ is atomic because workers
    // of descendants read the ancestor chain without locks; every ancestor of
    // a linked split outlives it.
    std::atomic<Ybwc_split_point *> parent_{nullptr};
    Ybwc_split_point *first_child_ = nullptr;
    Ybwc_split_point *prev_sibling_ = nullptr;
    Ybwc_split_point *next_sibling_ = nullptr;
    const void *join_context_ = nullptr;
    int join_min_fail_low_ = 0;

    // Helpers and joiners, excluding the owner. A worker's final access to
    // this object is its increment of retired_.
    std::atomic<int> running_{0};
    std::atomic<int> admitted_{0};
    std::atomic<int> retired_{0};
    std::atomic<uint64_t> worker_nodes_{0};
    std::atomic<uint32_t> wake_{0};
    std::atomic<bool> owner_waiting_{false};

    void wake_owner() {
        wake_.fetch_add(1, std::memory_order_release);
        wake_.notify_one();
    }
    // Owners sleeping above this split may now join it.
    void wake_waiting_ancestors() {
        for (Ybwc_split_point *p = parent_.load(std::memory_order_acquire); p; p = p->parent_.load(std::memory_order_acquire)) {
            if (p->owner_waiting_.load(std::memory_order_seq_cst)) p->wake_owner();
        }
    }

public:
    Ybwc_split_point(int alpha, int beta, int value, int best_move, bool pv)
        : alpha_(alpha), beta_(beta), pv_(pv), value_(value), best_move_(best_move) {
        assert(alpha < beta);
    }
    Ybwc_split_point(const Ybwc_split_point&) = delete;
    Ybwc_split_point& operator=(const Ybwc_split_point&) = delete;

    // Populate before publishing this object to helper threads.
    void add_move(int index) {
        assert(count_ < 64 && index >= 0 && index < 64);
        moves_[count_++] = index;
    }
    int move_count() const { return count_; }
    bool *cancellation_flag() { return &searching_; }
    bool searching() const {
        return std::atomic_ref<bool>(const_cast<bool&>(searching_)).load(std::memory_order_relaxed);
    }
    int alpha() const { return alpha_.load(std::memory_order_relaxed); }
    int beta() const { return beta_; }
    void cancel() { std::atomic_ref<bool>(searching_).store(false, std::memory_order_relaxed); }

    Work take() {
        std::lock_guard lock(mutex_);
        if (!searching() || next_ == count_) return {};
        const int ordinal = next_++;
        return {moves_[ordinal], ordinal + 1, alpha()};
    }

    // An obsolete fail-high is still only a lower bound. In particular, a
    // value <= the new alpha is NOT evidence that the move can be discarded.
    // The caller must re-probe or re-search it before marking it complete.
    bool publish(Work work, int value, int policy, Bound bound) {
        bool became_joinable = false;
        {
            std::lock_guard lock(mutex_);
            if (!searching()) return false;
            if (bound == Bound::lower && value < beta_) return false;
            if (bound == Bound::upper && value > work.alpha) return false;
            assert(work.move_index >= 0 && work.move_index < 64);
            const uint64_t bit = uint64_t{1} << work.move_index;
            assert((completed_ & bit) == 0);
            completed_ |= bit;
            if (bound == Bound::upper && ++n_fail_low_ == join_min_fail_low_) {
                became_joinable = join_context_ != nullptr && next_ < count_;
            }
            if (value > value_) {
                value_ = value;
                best_move_ = policy;
            }
            if (pv_ && bound != Bound::upper && value > alpha()) {
                alpha_.store(value, std::memory_order_relaxed);
            }
            if (bound != Bound::upper && value >= beta_) cancel();
        }
        // The caller still works for this split, so it is linked if its
        // owner linked it, and every ancestor is alive.
        if (became_joinable) wake_waiting_ancestors();
        return true;
    }

    Result result() {
        std::lock_guard lock(mutex_);
        return {alpha(), value_, best_move_, completed_};
    }

    // ---- Workers (helpers and joiners) ----

    // Count a helper before it can start; undo if it could not be scheduled.
    void register_worker() {
        running_.fetch_add(1, std::memory_order_relaxed);
        admitted_.fetch_add(1, std::memory_order_relaxed);
    }
    void unregister_worker() {
        running_.fetch_sub(1, std::memory_order_relaxed);
        admitted_.fetch_sub(1, std::memory_order_relaxed);
    }
    // The worker must not touch this object after this call.
    void worker_done(uint64_t nodes) {
        worker_nodes_.fetch_add(nodes, std::memory_order_relaxed);
        running_.fetch_sub(1, std::memory_order_release);
        wake_owner();
        retired_.fetch_add(1, std::memory_order_release);
    }
    bool has_running_workers() const { return running_.load(std::memory_order_acquire) > 0; }
    uint32_t wake_sequence() const { return wake_.load(std::memory_order_acquire); }
    // While set, a descendant that becomes joinable wakes the owner.
    void set_owner_waiting(bool waiting) { owner_waiting_.store(waiting, std::memory_order_seq_cst); }
    // Whether the owner of this split or of one of its ancestors within
    // levels would join a split created below. Call only from a thread that
    // works for this split, which keeps the chain alive.
    bool has_waiting_owner(int levels) const {
        const Ybwc_split_point *p = this;
        for (int i = 0; p != nullptr && i < levels; ++i, p = p->parent_.load(std::memory_order_acquire)) {
            if (p->owner_waiting_.load(std::memory_order_relaxed)) return true;
        }
        return false;
    }
    void wait_wake(uint32_t seen) const { wake_.wait(seen, std::memory_order_acquire); }
    // Call after the last worker stopped running and no joiner can be admitted.
    void wait_retired() const {
        const int admitted = admitted_.load(std::memory_order_acquire);
        while (retired_.load(std::memory_order_acquire) != admitted) std::this_thread::yield();
    }
    uint64_t worker_nodes() const { return worker_nodes_.load(std::memory_order_acquire); }

    // ---- Split tree ----

    // Joiners may search this split once min_fail_low handed-out moves have
    // failed low. Call before helpers can publish results.
    void set_join_context(const void *context, int min_fail_low) {
        join_context_ = context;
        join_min_fail_low_ = min_fail_low;
    }
    const void *join_context() const { return join_context_; }
    std::mutex &mutex() { return mutex_; }

    void link_to(Ybwc_split_point *parent) {
        if (parent == nullptr) return;
        {
            std::lock_guard lock(parent->mutex_);
            parent_.store(parent, std::memory_order_release);
            next_sibling_ = parent->first_child_;
            if (next_sibling_) next_sibling_->prev_sibling_ = this;
            parent->first_child_ = this;
        }
        if (join_context_ != nullptr && join_min_fail_low_ <= 0) wake_waiting_ancestors();
    }
    // Call only after every worker of this split has stopped running.
    void unlink() {
        Ybwc_split_point *parent = parent_.load(std::memory_order_relaxed);
        if (parent == nullptr) return;
        std::lock_guard lock(parent->mutex_);
        if (prev_sibling_) prev_sibling_->next_sibling_ = next_sibling_;
        else parent->first_child_ = next_sibling_;
        if (next_sibling_) next_sibling_->prev_sibling_ = prev_sibling_;
        prev_sibling_ = next_sibling_ = nullptr;
        parent_.store(nullptr, std::memory_order_relaxed);
    }
    // Requires this split's mutex.
    Ybwc_split_point *first_child_locked() const { return first_child_; }
    // Requires the parent's mutex.
    Ybwc_split_point *next_sibling_locked() const { return next_sibling_; }

    // Requires this split's mutex. A joiner is admitted only while moves are
    // left, so the owner, which waits only after its own take() came back
    // empty, never sees a new worker after it started waiting.
    bool try_admit_joiner_locked(int max_workers) {
        if (join_context_ == nullptr || !searching() || next_ == count_) return false;
        if (n_fail_low_ < join_min_fail_low_) return false;
        if (1 + running_.load(std::memory_order_relaxed) >= max_workers) return false;
        register_worker();
        return true;
    }
};

// The split whose move the current thread is searching. A split created on
// this thread is linked below it.
inline thread_local Ybwc_split_point *ybwc_current_split = nullptr;

class Ybwc_current_split_scope {
    Ybwc_split_point *const saved_;
public:
    explicit Ybwc_current_split_scope(Ybwc_split_point *split) : saved_(ybwc_current_split) {
        ybwc_current_split = split;
    }
    ~Ybwc_current_split_scope() { ybwc_current_split = saved_; }
    Ybwc_current_split_scope(const Ybwc_current_split_scope &) = delete;
    Ybwc_current_split_scope &operator=(const Ybwc_current_split_scope &) = delete;
};
