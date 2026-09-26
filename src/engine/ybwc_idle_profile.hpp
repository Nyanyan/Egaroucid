/*
    Egaroucid Project

    @file ybwc_idle_profile.hpp
        Where parallel search threads spend time without searching
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

// Build with -DYBWC_IDLE_PROFILE=true; -solve then prints, per kind of wait,
// the thread time spent in it as a share of (wall time * threads).
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>

#ifndef YBWC_IDLE_PROFILE
#define YBWC_IDLE_PROFILE false
#endif

#if YBWC_IDLE_PROFILE
enum Ybwc_idle_kind {
    YBWC_IDLE_POOL,          // pool worker has no task
    YBWC_IDLE_OWNER_SLEEP,   // shared split owner sleeping for helpers
    YBWC_IDLE_OWNER_HELP,    // shared split owner searching in a descendant
    YBWC_IDLE_RETIRE,        // waiting for the last helper to retire
    YBWC_IDLE_LEGACY_WAIT,   // one-task-per-move split waiting for tasks
    YBWC_IDLE_LEGACY_HELP,   // one-task-per-move split running a queued task
    YBWC_IDLE_N
};
inline std::atomic<uint64_t> ybwc_idle_ns[YBWC_IDLE_N];
inline std::atomic<uint64_t> ybwc_idle_count[YBWC_IDLE_N];
inline std::atomic<uint64_t> ybwc_push_cap_fail{0};
inline std::atomic<uint64_t> ybwc_shared_splits{0};
inline std::atomic<uint64_t> ybwc_shared_helpers{0};
inline std::atomic<uint64_t> ybwc_shared_joins{0};
inline std::atomic<uint64_t> ybwc_idle_joins{0};

struct Ybwc_idle_timer {
    Ybwc_idle_kind kind;
    std::chrono::steady_clock::time_point start;
    explicit Ybwc_idle_timer(Ybwc_idle_kind k) : kind(k), start(std::chrono::steady_clock::now()) {}
    ~Ybwc_idle_timer() {
        ybwc_idle_ns[kind].fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count(),
            std::memory_order_relaxed);
        ybwc_idle_count[kind].fetch_add(1, std::memory_order_relaxed);
    }
};
#define YBWC_IDLE_SCOPE(kind) const Ybwc_idle_timer ybwc_idle_timer(kind)
#define YBWC_IDLE_COUNT(counter) (counter).fetch_add(1, std::memory_order_relaxed)

inline void ybwc_idle_profile_reset() {
    for (int i = 0; i < YBWC_IDLE_N; ++i) {
        ybwc_idle_ns[i] = 0;
        ybwc_idle_count[i] = 0;
    }
    ybwc_push_cap_fail = 0;
    ybwc_shared_splits = 0;
    ybwc_shared_helpers = 0;
    ybwc_shared_joins = 0;
    ybwc_idle_joins = 0;
}

inline void ybwc_idle_profile_print(double wall_sec, int n_threads) {
    static const char *names[YBWC_IDLE_N] = {"pool_idle", "owner_sleep", "owner_help", "retire_wait", "legacy_wait", "legacy_help"};
    const double thread_sec = wall_sec * n_threads;
    std::cerr << "idle profile wall " << wall_sec << "s threads " << n_threads << " thread_sec " << thread_sec << std::endl;
    for (int i = 0; i < YBWC_IDLE_N; ++i) {
        const double s = ybwc_idle_ns[i].load() / 1e9;
        std::cerr << "  " << names[i] << " " << s << "s (" << 100.0 * s / thread_sec << "%) count " << ybwc_idle_count[i].load() << std::endl;
    }
    std::cerr << "  push_cap_fail " << ybwc_push_cap_fail.load()
              << " shared_splits " << ybwc_shared_splits.load() << " shared_helpers " << ybwc_shared_helpers.load()
              << " owner_joins " << ybwc_shared_joins.load() << " idle_joins " << ybwc_idle_joins.load() << std::endl;
}
#else
#define YBWC_IDLE_SCOPE(kind) ((void)0)
#define YBWC_IDLE_COUNT(counter) ((void)0)
#endif
