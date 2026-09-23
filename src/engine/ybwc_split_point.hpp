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

// Synchronization is needed only when taking or completing a whole subtree.
// No lock is held while searching it (including any nested split points).
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
    alignas(std::atomic_ref<bool>::required_alignment) bool searching_ = true;

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
        std::lock_guard lock(mutex_);
        if (!searching()) return false;
        if (bound == Bound::lower && value < beta_) return false;
        if (bound == Bound::upper && value > work.alpha) return false;
        assert(work.move_index >= 0 && work.move_index < 64);
        const uint64_t bit = uint64_t{1} << work.move_index;
        assert((completed_ & bit) == 0);
        completed_ |= bit;
        if (value > value_) {
            value_ = value;
            best_move_ = policy;
        }
        if (pv_ && bound != Bound::upper && value > alpha()) {
            alpha_.store(value, std::memory_order_relaxed);
        }
        if (bound != Bound::upper && value >= beta_) cancel();
        return true;
    }

    Result result() {
        std::lock_guard lock(mutex_);
        return {alpha(), value_, best_move_, completed_};
    }
};
