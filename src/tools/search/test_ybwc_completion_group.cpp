/*
    Focused stress tests for the fixed-slot YBWC completion group.

    Example (run the executable from bin/):
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            ../src/tools/search/test_ybwc_completion_group.cpp \
            -o test_ybwc_completion_group.exe
*/

#include <array>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../engine/ybwc_completion_group.hpp"

namespace {

struct Test_result {
    int slot;
    int value;
};

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_reverse_completion_and_full_mask() {
    Ybwc_completion_group<Test_result, 64> group;
    std::array<std::size_t, 64> slots{};
    for (std::size_t i = 0; i < slots.size(); ++i) {
        slots[i] = group.reserve_slot();
        group.mark_submitted(slots[i]);
    }
    for (std::size_t i = slots.size(); i-- > 0;) {
        group.publish(slots[i], Test_result{static_cast<int>(i), static_cast<int>(i * 3)});
    }
    for (std::size_t expected = 0; expected < slots.size(); ++expected) {
        Test_result result;
        require(group.try_pop(&result), "reverse completion lost a result");
        require(result.slot == static_cast<int>(expected), "reverse completion order mismatch");
        require(result.value == static_cast<int>(expected * 3), "reverse completion payload mismatch");
    }
    Test_result result;
    require(!group.try_pop(&result), "full mask did not drain completely");
}

void test_failed_push_holes() {
    Ybwc_completion_group<Test_result, 8> group;
    const std::size_t hole0 = group.reserve_slot();
    const std::size_t slot1 = group.reserve_slot();
    const std::size_t hole2 = group.reserve_slot();
    const std::size_t slot3 = group.reserve_slot();
    (void)hole0;
    (void)hole2;
    group.mark_submitted(slot1);
    group.mark_submitted(slot3);
    group.publish(slot3, Test_result{3, 30});
    group.publish(slot1, Test_result{1, 10});

    Test_result result;
    require(group.try_pop(&result) && result.slot == 1, "a failed-push hole blocked slot 1");
    require(group.try_pop(&result) && result.slot == 3, "a failed-push hole blocked slot 3");
    require(!group.try_pop(&result), "failed-push holes appeared ready");
}

void test_concurrent_publish_and_consume() {
    constexpr std::size_t n_slots = 64;
    constexpr std::size_t n_publishers = 8;
    Ybwc_completion_group<Test_result, n_slots> group;
    std::array<std::size_t, n_slots> slots{};
    for (std::size_t i = 0; i < n_slots; ++i) {
        slots[i] = group.reserve_slot();
        group.mark_submitted(slots[i]);
    }

    std::atomic<bool> start{false};
    std::vector<std::thread> publishers;
    for (std::size_t publisher = 0; publisher < n_publishers; ++publisher) {
        publishers.emplace_back([&, publisher]() {
            start.wait(false, std::memory_order_acquire);
            for (std::size_t i = n_slots - n_publishers + publisher;; i -= n_publishers) {
                group.publish(slots[i], Test_result{static_cast<int>(i), static_cast<int>(1000 + i)});
                if (i < n_publishers) {
                    break;
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    start.notify_all();

    std::array<bool, n_slots> seen{};
    for (std::size_t count = 0; count < n_slots; ++count) {
        group.wait_for_ready();
        Test_result result;
        require(group.try_pop(&result), "ready notification had no result");
        require(result.slot >= 0 && result.slot < static_cast<int>(n_slots), "invalid concurrent slot");
        require(!seen[result.slot], "concurrent result was consumed twice");
        require(result.value == 1000 + result.slot, "concurrent payload mismatch");
        seen[result.slot] = true;
    }
    for (std::thread &publisher: publishers) {
        publisher.join();
    }
    for (bool was_seen: seen) {
        require(was_seen, "concurrent publication lost a result");
    }
}

void test_notification_races() {
    constexpr int n_repetitions = 256;
    for (int repetition = 0; repetition < n_repetitions; ++repetition) {
        Ybwc_completion_group<Test_result, 2> group;
        const std::size_t slot = group.reserve_slot();
        group.mark_submitted(slot);
        std::atomic<bool> start{false};
        std::thread publisher([&]() {
            start.wait(false, std::memory_order_acquire);
            if ((repetition & 1) != 0) {
                std::this_thread::yield();
            }
            group.publish(slot, Test_result{repetition, repetition * 7});
        });
        start.store(true, std::memory_order_release);
        start.notify_one();
        if ((repetition & 1) == 0) {
            std::this_thread::yield();
        }
        group.wait_for_ready();
        Test_result result;
        require(group.try_pop(&result), "notification race lost readiness");
        require(result.slot == repetition && result.value == repetition * 7, "notification race payload mismatch");
        publisher.join();
    }
}

void test_nested_groups() {
    Ybwc_completion_group<Test_result, 4> outer;
    const std::size_t outer0 = outer.reserve_slot();
    const std::size_t outer1 = outer.reserve_slot();
    outer.mark_submitted(outer0);
    outer.mark_submitted(outer1);

    std::thread nested([&]() {
        Ybwc_completion_group<Test_result, 4> inner;
        const std::size_t inner0 = inner.reserve_slot();
        const std::size_t inner1 = inner.reserve_slot();
        inner.mark_submitted(inner0);
        inner.mark_submitted(inner1);
        std::thread child([&]() {
            inner.publish(inner1, Test_result{1, 20});
            inner.publish(inner0, Test_result{0, 10});
        });
        int sum = 0;
        for (int i = 0; i < 2; ++i) {
            inner.wait_for_ready();
            Test_result result;
            require(inner.try_pop(&result), "nested inner notification lost a result");
            sum += result.value;
        }
        child.join();
        outer.publish(outer0, Test_result{0, sum});
    });
    std::thread sibling([&]() {
        outer.publish(outer1, Test_result{1, 40});
    });

    std::array<bool, 2> seen{};
    int sum = 0;
    for (int i = 0; i < 2; ++i) {
        outer.wait_for_ready();
        Test_result result;
        require(outer.try_pop(&result), "nested outer notification lost a result");
        require(result.slot >= 0 && result.slot < 2 && !seen[result.slot], "nested outer slot mismatch");
        seen[result.slot] = true;
        sum += result.value;
    }
    nested.join();
    sibling.join();
    require(sum == 70, "nested completion groups crossed results");
}

void test_retirement_before_stack_release() {
    constexpr int n_repetitions = 1024;
    for (int repetition = 0; repetition < n_repetitions; ++repetition) {
        auto group = std::make_unique<Ybwc_completion_group<Test_result, 2>>();
        const std::size_t slot = group->reserve_slot();
        group->mark_submitted(slot);
        std::thread publisher([completion_group = group.get(), slot, repetition]() {
            completion_group->publish(slot, Test_result{repetition, repetition * 11});
        });
        group->wait_for_ready();
        Test_result result;
        require(group->try_pop(&result), "retirement stress lost a result");
        require(result.slot == repetition && result.value == repetition * 11, "retirement stress payload mismatch");
        group.reset();
        publisher.join();
    }
}

} // namespace

int main() {
    try {
        test_reverse_completion_and_full_mask();
        test_failed_push_holes();
        test_concurrent_publish_and_consume();
        test_notification_races();
        test_nested_groups();
        test_retirement_before_stack_release();
        std::cout << "YBWC completion group tests passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "YBWC completion group test failed: " << error.what() << std::endl;
        return 1;
    }
}
