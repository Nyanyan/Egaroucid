/*
    Egaroucid Project

    @file spinlock.hpp
        Spinlock
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include <atomic>
#include <thread>
#include <mutex>

// original: https://rigtorp.se/spinlock/
// modified by Nyanyan
struct Spinlock {
    std::atomic<bool> lock_ = {0};

    void lock(){
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            //while (lock_.load(std::memory_order_relaxed)) {
            //    __builtin_ia32_pause();
            //}
        }
    }

    bool try_lock() noexcept {
        return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        lock_.store(false, std::memory_order_release);
    }
};
// end of modification