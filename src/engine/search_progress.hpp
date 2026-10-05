/*
    Egaroucid Project

    @file search_progress.hpp
        Scoped progress reporting for foreground searches
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <utility>

struct Search_progress {
    int depth;
    double probability;
    int score;
    uint64_t elapsed_msec;
};

using Search_progress_callback = std::function<void(const Search_progress &)>;

class Search_progress_scope;
inline thread_local const Search_progress_scope *active_search_progress = nullptr;

// A callback belongs to the calling search only. Helper and ponder threads do
// not inherit it, and simultaneous GGS searches can use separate callbacks.
class Search_progress_scope {
    Search_progress_callback callback;
    const Search_progress_scope *previous;
    std::chrono::steady_clock::time_point start;

public:
    explicit Search_progress_scope(Search_progress_callback callback_)
        : callback(std::move(callback_)), previous(active_search_progress), start(std::chrono::steady_clock::now()) {
        active_search_progress = this;
    }

    Search_progress_scope(const Search_progress_scope &) = delete;
    Search_progress_scope &operator=(const Search_progress_scope &) = delete;

    ~Search_progress_scope() {
        active_search_progress = previous;
    }

    void report(int depth, double probability, int score) const {
        if (callback) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed >= 1000) {
                callback(Search_progress{depth, probability, score, static_cast<uint64_t>(elapsed)});
            }
        }
    }
};

inline void report_search_progress(int depth, double probability, int score) {
    if (active_search_progress != nullptr) {
        active_search_progress->report(depth, probability, score);
    }
}
