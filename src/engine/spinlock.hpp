/*
    Egaroucid Project

    @file spinlock.hpp
        Spinlock
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
    @notice I referred to codes written by others
*/

#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
#endif

// original: https://rigtorp.se/spinlock/
// modified by Nyanyan
struct Spinlock {
    std::atomic<bool> lock_ = {0};

    void lock() {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            while (lock_.load(std::memory_order_relaxed)) {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
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
