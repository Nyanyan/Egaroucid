// clang++ -O2 -pthread -std=c++20 test_ybwc_split_point.cpp -o <test executable>
#include <array>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "../../engine/ybwc_split_point.hpp"

using Bound = Ybwc_split_point::Bound;
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void stale_lower_bound() {
    Ybwc_split_point split(10, 50, 10, 0, true);
    split.add_move(1); split.add_move(2);
    auto slow = split.take();
    auto fast = split.take();
    require(split.publish(fast, 30, 2, Bound::exact), "publish improved PV");
    require(!split.publish(slow, 20, 1, Bound::lower), "obsolete lower bound needs a new search");
    require(split.result().completed == (uint64_t{1} << 2), "unproved move must remain incomplete");
    slow.alpha = split.alpha();
    require(split.publish(slow, 42, 1, Bound::exact), "re-search discovers better move");
    require(split.result().value == 42 && split.result().best_move == 1, "retain re-search result");
}

void old_upper_bound_and_cutoff() {
    Ybwc_split_point split(10, 50, 10, 0, true);
    for (int i = 1; i <= 4; ++i) split.add_move(i);
    auto old = split.take();
    auto improving = split.take();
    require(split.publish(improving, 30, 2, Bound::exact), "PV improvement");
    require(split.publish(old, 8, 1, Bound::upper), "old fail-low remains safe at higher alpha");
    require(split.result().best_move == 2, "fail-low cannot replace best move");
    auto cutoff = split.take();
    require(split.publish(cutoff, 52, 3, Bound::lower), "beta cutoff");
    require(!split.searching() && !split.take(), "cutoff stops new work");
    require(split.result().value == 52, "completed cutoff must survive cancellation");
}

void parallel_claims() {
    for (int repetition = 0; repetition < 100; ++repetition) {
        Ybwc_split_point split(-64, 65, -64, -1, true);
        for (int i = 0; i < 64; ++i) split.add_move(i);
        std::array<std::atomic<int>, 64> visits{};
        std::vector<std::thread> workers;
        for (int w = 0; w < 8; ++w) workers.emplace_back([&] {
            while (auto work = split.take()) {
                ++visits[work.move_index];
                std::this_thread::yield();
                if (!split.publish(work, work.move_index, work.move_index, Bound::exact)) std::terminate();
            }
        });
        for (auto &worker : workers) worker.join();
        for (auto &n : visits) require(n.load() == 1, "each sibling must be claimed once");
        const auto result = split.result();
        require(result.value == 63 && result.alpha == 63 && result.best_move == 63, "parallel max");
        require(result.completed == UINT64_MAX, "all moves accounted for");
    }
}

int main() {
    try {
        stale_lower_bound(); old_upper_bound_and_cutoff(); parallel_claims();
        Ybwc_split_point cancelled(0, 1, -64, -1, false);
        cancelled.add_move(0); cancelled.cancel();
        require(!cancelled.take(), "external stop prevents claiming work");
        std::cout << "PASS shared split-point bounds, cancellation, and concurrent claims\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
