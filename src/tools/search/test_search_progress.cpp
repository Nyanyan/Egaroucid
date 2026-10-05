/*
    Regression checks for completed summaries, the one-second gate, and thread isolation.

    clang++ -O2 -pthread -std=c++20 src/tools/search/test_search_progress.cpp -o bin/test_search_progress.exe
*/
#include <iostream>
#include <future>
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
        require(search_progress_summary({28, 99.9, -4, 1234}) ==
                "search depth 28 mpc 99.9% score -4 time 1234 ms", "decimal MPC probability and negative score");
        require(!console_search_progress_callback(false), "quiet/protocol output disabled");

        std::vector<Search_progress> events;
        {
            Search_progress_scope scope([&](const Search_progress &progress) { events.push_back(progress); });
            report_search_progress(12, 99.9, -4);
            require(events.empty(), "completed searches before one second are suppressed");
            auto helper = std::async(std::launch::async, [] {
                report_search_progress(20, 100, 5);
                return active_search_progress == nullptr;
            });
            require(helper.get(), "helper does not inherit foreground callback");
            std::promise<void> other_ready;
            std::promise<void> release_other;
            auto release_future = release_other.get_future();
            auto other_game = std::async(std::launch::async, [&] {
                int count = 0;
                Search_progress_scope other([&](const Search_progress &progress) {
                    require(progress.depth == 16 && progress.probability == 88 && progress.score == 7, "other game parameters");
                    ++count;
                });
                report_search_progress(16, 88, 7);
                require(count == 0, "other game has its own one-second gate");
                other_ready.set_value();
                release_future.wait();
                report_search_progress(16, 88, 7);
                return count;
            });
            other_ready.get_future().wait();
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            require(events.empty(), "elapsed time alone does not emit a running update");
            release_other.set_value();
            require(other_game.get() == 1, "other game receives its own callback");
            require(events.empty(), "other game does not emit to the foreground callback");
            report_search_progress(12, 99.9, -4);
            require(events.size() == 1, "completed searches after one second are reported");
            {
                Search_progress_scope disabled({});
                report_search_progress(8, 74, 10);
            }
            report_search_progress(13, 100, 0);
            require(events.size() == 2, "nested disabled scope restored");
            require(events.front().depth == 12 && events.front().probability == 99.9 && events.front().score == -4 && events.front().elapsed_msec >= 1000,
                    "completed summary reports depth, probability, score, and move elapsed time");
            require(events.back().score == 0 && events.back().elapsed_msec >= events.front().elapsed_msec,
                    "score zero and cumulative elapsed time preserved");
        }
        require(active_search_progress == nullptr, "foreground scope cleaned up");
        report_search_progress(1, 100, 1);
        std::cout << "Search progress tests passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
