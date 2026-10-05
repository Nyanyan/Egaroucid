/*
    Regression checks for scoped output, decimal MPC probabilities, and deadlines.

    clang++ -O2 -pthread -std=c++20 src/tools/search/test_search_progress.cpp -o bin/test_search_progress.exe
*/
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../../console/search_progress.hpp"

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        require(search_progress_summary({28, 99.9, 1234, "running"}) ==
                "search depth 28 mpc 99.9% time 1234 ms running", "decimal MPC probability");
        require(!console_search_progress_callback(false), "quiet/protocol output disabled");

        std::vector<Search_progress> events;
        {
            Search_progress_scope scope([&](const Search_progress &progress) { events.push_back(progress); });
            report_search_progress(12, 99.9, "started");
            {
                Search_progress_scope disabled({});
                report_search_progress(8, 74, "started");
            }
            auto helper = std::async(std::launch::async, [] {
                report_search_progress(20, 100, "started");
                return active_search_progress == nullptr;
            });
            require(helper.get(), "helper does not inherit foreground callback");
            auto other_game = std::async(std::launch::async, [] {
                int count = 0;
                Search_progress_scope other([&](const Search_progress &progress) {
                    require(progress.depth == 16 && progress.probability == 88, "other game parameters");
                    ++count;
                });
                report_search_progress(16, 88, "completed");
                return count;
            });
            require(other_game.get() == 1, "other game receives its own callback");
            report_search_progress(12, 99.9, "completed");
            require(events.size() == 2, "nested scope restored and threads isolated");

            std::promise<int> ready;
            auto ready_future = ready.get_future();
            ready.set_value(7);
            require(wait_for_search_with_progress(ready_future, 0, 12, 99.9) == std::future_status::ready,
                    "already ready future at zero budget");
            require(ready_future.get() == 7, "future result preserved");

            std::promise<int> pending;
            auto pending_future = pending.get_future();
            const auto start = std::chrono::steady_clock::now();
            require(wait_for_search_with_progress(pending_future, 1250, 28, 99.9) == std::future_status::timeout,
                    "unfinished search times out");
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            require(elapsed >= 1250 && elapsed < 2000, "progress does not restart deadline");
            require(events.size() == 3 && std::string(events.back().state) == "running", "periodic update while pending");
            require(events.back().depth == 28 && events.back().probability == 99.9 && events.back().elapsed_msec >= 1000,
                    "periodic update reports active depth, probability, and elapsed time");
        }
        require(active_search_progress == nullptr, "foreground scope cleaned up");
        std::promise<int> pending;
        auto future = pending.get_future();
        require(wait_for_search_with_progress(future, 0, 1, 100) == std::future_status::timeout, "disabled reporting preserves timeout");
        std::cout << "Search progress tests passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
