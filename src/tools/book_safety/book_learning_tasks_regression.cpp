#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
#include "thread_pool.hpp"

thread_local int task_number = -1;
struct Test_pool {
    int next_task = 0;
    int throw_on_push = -1;
    int submitted = 0;
    std::atomic<int> completed{0};
    std::atomic<bool> *callback_started = nullptr;
    int size() const { return thread_pool.size(); }
    template<class F>
    std::future<void> push(bool *pushed, F function) {
        const int number = next_task++;
        if (number == throw_on_push) throw std::bad_alloc();
        auto task = thread_pool.push(pushed, [this, function, number] {
            struct Complete {
                std::atomic<int> &count;
                ~Complete() { ++count; }
            } complete{completed};
            task_number = number;
            function();
        });
        if (*pushed) {
            ++submitted;
            if (callback_started) while (!callback_started->load()) std::this_thread::yield();
        }
        return task;
    }
} test_pool;

#define thread_pool test_pool
struct Test_book {
#include "book_parallel_method.hpp"
} test_book;
#include "book_learning_method.hpp"
#undef thread_pool

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void set_threads(int n) {
    thread_pool.resize(n - 1);
    while (thread_pool.get_n_idle() != n - 1) std::this_thread::yield();
    test_pool.next_task = 0;
    test_pool.throw_on_push = -1;
    test_pool.submitted = 0;
    test_pool.completed.store(0);
    test_pool.callback_started = nullptr;
}

template<class F>
void run_queue(bool learning, int n, F function) {
    if (learning) book_learning_for_each(n, function);
    else test_book.parallel_for(n, [&](int i) { function(i, false); });
}

// On the broken implementation exit before gated tasks can touch dead stack state.
void require_joined(bool finished, const char *message) {
    if (!finished) {
        std::cerr << "FAIL: " << message << std::endl;
        std::_Exit(1);
    }
}

void check_exception(bool learning, int source) {
    set_threads(3);
    const auto caller = std::this_thread::get_id();
    std::atomic<bool> started(false), release(false), finished(false), throwing(false);
    std::thread gate([&] {
        while (!started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.store(true);
    });
    if (source == 2) {
        test_pool.throw_on_push = 1;
        test_pool.callback_started = &started;
    }
    bool caught = false;
    try {
        run_queue(learning, 200, [&](int, bool) {
            if (std::this_thread::get_id() == caller) {
                if (source == 0) {
                    while (!started.load()) std::this_thread::yield();
                    throw std::bad_alloc();
                }
                if (source == 1) while (!throwing.load()) std::this_thread::yield();
                return;
            }
            if (source == 1 && task_number == 0) {
                while (!started.load()) std::this_thread::yield();
                throwing.store(true);
                throw std::bad_alloc();
            }
            started.store(true);
            while (!release.load()) std::this_thread::yield();
            finished.store(true);
        });
    } catch (const std::bad_alloc &) {
        require_joined(finished.load() && test_pool.completed.load() == test_pool.submitted,
                       "exception escaped before every worker finished");
        caught = true;
    }
    gate.join();
    require(caught, "exception was swallowed");
    std::cout << "PASS: " << (learning ? "learning" : "parallel_for")
              << " exception source=" << source << " joins workers\n";
}

void check_normal(bool learning) {
    for (int threads: {1, 2, 8}) for (int n: {0, 1, 2, 3, 30, 200}) {
        set_threads(threads);
        std::vector<std::atomic<int>> calls(n);
        for (auto &value: calls) value.store(0);
        std::atomic<int> multi(0);
        run_queue(learning, n, [&](int i, bool use_multi_thread) {
            ++calls[i];
            if (use_multi_thread) ++multi;
        });
        for (auto &value: calls) require(value.load() == 1, "element was skipped or repeated");
        if (learning) {
            int expected = threads == 1 ? n : std::min(n, (threads - 1) * 2 - 1);
            require(multi.load() == expected, "multi-thread search tail changed");
        }
    }
    std::cout << "PASS: normal queue runs each element once, including the search tail\n";
}

template<class F, class Continue>
void run_cancellable(int n, F function, Continue should_continue) {
#ifdef BOOK_LEARNING_HAS_STOP_PREDICATE
    book_learning_for_each(n, function, should_continue);
#else
    book_learning_for_each(n, function);
#endif
}

void check_cancellation() {
    for (int threads: {1, 2, 8, 32}) for (bool initially_running: {false, true}) {
        set_threads(threads);
        std::atomic<bool> running(initially_running);
        std::atomic<int> calls(0);
        run_cancellable(1000000, [&](int, bool) {
            ++calls;
            if (!running.load()) return; // the old callback checked the stop flag here
            running.store(false);
        }, [&] { return running.load(); });
        require(calls.load() <= (initially_running ? threads : 0), "stopped queue drained pending elements");
    }
    // With fewer elements than the multi-thread tail, cancellation must still stop the caller's loop.
    set_threads(8);
    bool running = true;
    int calls = 0;
    run_cancellable(3, [&](int, bool) { ++calls; running = false; }, [&] { return running; });
    require(calls == 1, "multi-thread tail ignored cancellation");

    set_threads(3);
    std::atomic<bool> continue_work(true), started(false), release(false);
    const auto caller = std::this_thread::get_id();
    std::thread gate([&] {
        while (!started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.store(true);
    });
    run_cancellable(1000000, [&](int, bool) {
        if (std::this_thread::get_id() == caller) {
            while (!started.load()) std::this_thread::yield();
            continue_work.store(false);
        } else {
            started.store(true);
            while (!release.load()) std::this_thread::yield();
        }
    }, [&] { return continue_work.load(); });
    require_joined(test_pool.completed.load() == test_pool.submitted, "cancellation did not join workers");
    gate.join();
    std::cout << "PASS: cancellation stops new work and joins active workers (1/2/8/32 threads)\n";
}

int main(int argc, char **argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "all";
        if (mode != "cancellation") {
            for (bool learning: {false, true}) {
                for (int source: {0, 1, 2}) check_exception(learning, source);
                check_normal(learning);
            }
        }
        if (mode != "exceptions") check_cancellation();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
