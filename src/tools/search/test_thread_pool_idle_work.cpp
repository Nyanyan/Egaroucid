// clang++ -O2 -pthread -std=c++20 test_thread_pool_idle_work.cpp -o <test executable>
// Idle workers take work offered outside the task queue, without starving
// pushed tasks or taking more threads than are idle.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include "../../engine/thread_pool.hpp"

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

template <typename Condition>
bool wait_until(Condition condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!condition()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

std::atomic<int> offered{0};
std::atomic<int> done{0};
std::atomic<int> running{0};
std::atomic<int> max_running{0};
std::atomic<bool> hold{false};
constexpr thread_id_t TEST_ID = 3;

bool take_offered_work() {
    int n = offered.load();
    do {
        if (n <= 0) return false;
    } while (!offered.compare_exchange_weak(n, n - 1));
    if (!thread_pool.claim_idle_worker(TEST_ID, THREAD_SIZE_INF)) {
        ++offered;
        return false;
    }
    // Like the search's idle work, wake the next idle worker for the rest.
    thread_pool.notify_idle_work();
    const int r = ++running;
    for (int m = max_running.load(); m < r && !max_running.compare_exchange_weak(m, r);) {}
    while (hold.load()) std::this_thread::yield();
    --running;
    ++done;
    thread_pool.release_idle_worker(TEST_ID, THREAD_SIZE_INF);
    return true;
}

int main() {
    try {
        thread_pool.resize(3);
        thread_pool.set_max_thread_size(TEST_ID, 2);
        require(wait_until([] { return thread_pool.get_n_idle() == 3; }), "workers start idle");
        thread_pool.set_idle_work(&take_offered_work);

        // Offered after every worker went to sleep: the notification wakes them.
        offered = 50;
        thread_pool.notify_idle_work();
        require(wait_until([] { return done.load() == 50; }), "sleeping workers take offered work");
        require(max_running.load() <= 2, "claims respect the per-id thread limit");
        require(wait_until([] { return thread_pool.get_n_idle() == 3; }), "workers are idle again");
        require(thread_pool.get_n_using_thread(TEST_ID) == 0, "claims are released");

        // Two workers are held in offered work. The third cannot be claimed
        // for TEST_ID and remains available for a pushed task.
        hold = true;
        offered = 3;
        thread_pool.notify_idle_work();
        require(wait_until([] { return running.load() == 2; }), "two workers hold offered work");
        bool pushed = false;
        std::atomic<bool> task_ran{false};
        auto future = thread_pool.push(&pushed, [&task_ran] { task_ran = true; });
        require(pushed, "an idle worker is left for a pushed task");
        future.get();
        require(task_ran.load(), "the pushed task ran");
        require(wait_until([] { return thread_pool.get_n_idle() == 1; }), "only the task's worker is idle again");
        require(!thread_pool.claim_idle_worker(TEST_ID, THREAD_SIZE_INF), "per-id limit reached");
        hold = false;
        require(wait_until([] { return done.load() == 53; }), "the third offer is taken after a worker is free");
        require(wait_until([] { return thread_pool.get_n_idle() == 3; }), "all workers idle at the end");

        // A claim takes one idle thread out of the count, like a push.
        require(thread_pool.claim_idle_worker(THREAD_ID_NONE, THREAD_SIZE_INF), "claim");
        require(thread_pool.get_n_idle() == 2, "claim reserves an idle thread");
        thread_pool.release_idle_worker(THREAD_ID_NONE, THREAD_SIZE_INF);

        thread_pool.set_idle_work(nullptr);
        thread_pool.exit_thread();
        std::cout << "PASS thread pool idle work, claims and notifications\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        hold = false;
        offered = 0;
        std::_Exit(1);
    }
}
