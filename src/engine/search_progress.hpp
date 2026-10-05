/*
    Egaroucid Project

    @file search_progress.hpp
        Scoped progress reporting for foreground searches
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <utility>

struct Search_progress {
    int depth;
    double probability;
    uint64_t elapsed_msec;
    const char *state;
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

    bool enabled() const {
        return static_cast<bool>(callback);
    }

    void report(int depth, double probability, const char *state) const {
        if (callback) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            callback(Search_progress{depth, probability, static_cast<uint64_t>(elapsed), state});
        }
    }
};

inline void report_search_progress(int depth, double probability, const char *state) {
    if (active_search_progress != nullptr) {
        active_search_progress->report(depth, probability, state);
    }
}

template <class T>
std::future_status wait_for_search_with_progress(std::future<T> &future, uint64_t time_limit_msec, int depth, double probability) {
    if (active_search_progress == nullptr || !active_search_progress->enabled()) {
        return future.wait_for(std::chrono::milliseconds(time_limit_msec));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_limit_msec);
    while (true) {
        const auto next_update = std::min(deadline, std::chrono::steady_clock::now() + std::chrono::seconds(1));
        const auto status = future.wait_until(next_update);
        if (status != std::future_status::timeout || std::chrono::steady_clock::now() >= deadline) {
            return status;
        }
        report_search_progress(depth, probability, "running");
    }
}
