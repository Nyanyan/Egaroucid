/*
    Egaroucid Project

    @file ybwc_completion_group.hpp
        Fixed-slot completion notification for YBWC tasks
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <utility>

template <typename Result, std::size_t Capacity>
class Ybwc_completion_group {
    static_assert(Capacity > 0);
    static_assert(Capacity <= 64);
    static_assert(std::is_trivially_default_constructible_v<Result>);
    static_assert(std::is_trivially_destructible_v<Result>);
    static_assert(std::is_trivially_copyable_v<Result>);
    static_assert(std::is_nothrow_move_assignable_v<Result>);

    std::array<Result, Capacity> results_;
    std::atomic<uint64_t> ready_mask_{0};
    std::atomic<uint64_t> retired_mask_{0};
    uint64_t submitted_mask_ = 0;
    std::size_t next_slot_ = 0;

public:
    Ybwc_completion_group() = default;
    ~Ybwc_completion_group() {
        unsigned int spins = 0;
        while ((retired_mask_.load(std::memory_order_acquire) & submitted_mask_) != submitted_mask_) {
            if (++spins == 64) {
                spins = 0;
                std::this_thread::yield();
            }
        }
    }
    Ybwc_completion_group(const Ybwc_completion_group&) = delete;
    Ybwc_completion_group& operator=(const Ybwc_completion_group&) = delete;

    std::size_t reserve_slot() noexcept {
        assert(next_slot_ < Capacity);
        return next_slot_++;
    }

    void mark_submitted(const std::size_t slot) noexcept {
        assert(slot < Capacity);
        submitted_mask_ |= uint64_t{1} << slot;
    }

    void publish(const std::size_t slot, Result result) noexcept {
        assert(slot < Capacity);
        const uint64_t slot_bit = uint64_t{1} << slot;
        results_[slot] = std::move(result);
        ready_mask_.fetch_or(slot_bit, std::memory_order_release);
        ready_mask_.notify_one();
        // This must remain the worker's final access to the group. The parent
        // destructor acquires retired_mask_ before releasing stack storage.
        retired_mask_.fetch_or(slot_bit, std::memory_order_release);
    }

    bool try_pop(Result *result) {
        const uint64_t ready = ready_mask_.load(std::memory_order_acquire);
        if (ready == 0) {
            return false;
        }
        const unsigned int slot = std::countr_zero(ready);
        const uint64_t slot_bit = uint64_t{1} << slot;
        *result = std::move(results_[slot]);
        ready_mask_.fetch_and(~slot_bit, std::memory_order_acq_rel);
        return true;
    }

    void wait_for_ready() const noexcept {
        uint64_t ready = ready_mask_.load(std::memory_order_acquire);
        while (ready == 0) {
            ready_mask_.wait(0, std::memory_order_acquire);
            ready = ready_mask_.load(std::memory_order_acquire);
        }
    }
};
