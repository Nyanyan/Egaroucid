/*
    Egaroucid Project

    @file search_controller.hpp
        Fast search termination control using bitmask
    @date 2025
    @author GitHub Copilot
    @license GPL-3.0-or-later
*/

#pragma once
#include <atomic>
#include <cstdint>

/*
    @brief Fast search controller using bitmask
    
    Uses a single 64-bit atomic variable to track up to 64 threads.
    Provides O(1) search termination checking instead of O(n).
*/
class FastSearchController {
private:
    static constexpr size_t MAX_THREADS = 64;
    std::atomic<uint64_t> search_mask{UINT64_MAX};  // All bits set initially
    static std::atomic<int> next_thread_index;
    
public:
    /*
        @brief Register a new thread and get its bit index
        @return thread index (0-63) or -1 if max threads exceeded
    */
    int register_thread() {
        int index = next_thread_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= MAX_THREADS) {
            return -1; // Error: too many threads
        }
        return index;
    }
    
    /*
        @brief Stop search for a specific thread
        @param index thread index to stop
    */
    void stop_thread(int index) {
        if (index >= 0 && index < MAX_THREADS) {
            uint64_t mask = ~(1ULL << index);
            search_mask.fetch_and(mask, std::memory_order_relaxed);
        }
    }
    
    /*
        @brief Check if any thread is still searching
        @return true if at least one thread is searching
    */
    bool is_searching() const {
        return search_mask.load(std::memory_order_relaxed) != 0;
    }
    
    /*
        @brief Check if a specific thread is searching
        @param index thread index to check
        @return true if the thread is searching
    */
    bool is_thread_searching(int index) const {
        if (index < 0 || index >= MAX_THREADS) return false;
        uint64_t mask = search_mask.load(std::memory_order_relaxed);
        return (mask & (1ULL << index)) != 0;
    }
    
    /*
        @brief Reset all threads to searching state
    */
    void reset() {
        search_mask.store(UINT64_MAX, std::memory_order_relaxed);
        next_thread_index.store(0, std::memory_order_relaxed);
    }
    
    /*
        @brief Stop all threads
    */
    void stop_all() {
        search_mask.store(0, std::memory_order_relaxed);
    }
};

// Global instance
extern FastSearchController g_search_controller;

// Static member definition
std::atomic<int> FastSearchController::next_thread_index{0};

// Global instance definition
FastSearchController g_search_controller;
