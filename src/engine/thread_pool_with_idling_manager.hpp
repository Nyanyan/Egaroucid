/*
    Egaroucid Project

    @file thread_pool_with_idling_manager.hpp
        Thread pool for Egaroucid (with idling manager)
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
    @notice This code is based on https://github.com/ContentsViewer/nodec/blob/main/nodec/include/nodec/concurrent/thread_pool_executor.hpp , which is published under Apache License 2.0
*/

#pragma once
#include <iostream>
#include <future>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_map>

#define THREAD_ID_SIZE 100
#define THREAD_ID_NONE 99 // reserved
#define THREAD_SIZE_INF 999999999
#define THREAD_SIZE_DEFAULT -1

using thread_id_t = int;

// Original code based on
//  * <https://github.com/bshoshany/thread-pool>
//  * <https://github.com/progschj/ThreadPool>
//  * <https://github.com/SandSnip3r/thread-pool>
// Thank you! :)

void reset_unavailable_task(bool *start_flag) {
    while (!*start_flag);
}

class Thread_pool {
    private:
        mutable std::mutex mtx;
        bool running;
        int n_thread;
        int n_idle;
        std::atomic<int> n_canbe_used;
        int n_set_threads;
        std::queue<std::pair<thread_id_t, std::function<void()>>> tasks{};
        std::unique_ptr<std::thread[]> threads;
        std::condition_variable condition;

        int max_thread_size[THREAD_ID_SIZE];
        std::atomic<int> n_using_thread[THREAD_ID_SIZE];

    public:
        void set_thread(int new_n_thread) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                //n_using_tasks.store(0);
                if (new_n_thread < 0) {
                    new_n_thread = 0;
                }
                n_thread = new_n_thread * 3;
                threads.reset(new std::thread[n_thread]);
                for (int i = 0; i < n_thread; ++i) {
                    threads[i] = std::thread(&Thread_pool::worker, this);
                }
                running = true;
                n_idle = 0;
                n_canbe_used = new_n_thread;
                n_set_threads = new_n_thread;
                for (int i = 0; i < THREAD_ID_SIZE; ++i) {
                    max_thread_size[i] = THREAD_SIZE_DEFAULT;
                    n_using_thread[i] = THREAD_SIZE_DEFAULT;
                }
                max_thread_size[THREAD_ID_NONE] = THREAD_SIZE_INF;
                n_using_thread[THREAD_ID_NONE] = 0;
            }
        }

        void exit_thread() {
            {
                std::lock_guard<std::mutex> lock(mtx);
                running = false;
            }
            condition.notify_all();
            for (int i = 0; i < n_thread; ++i) {
                if (threads[i].joinable()) {
                    threads[i].join();
                }
            }
            n_thread = 0;
            n_idle = 0;
        }

        void set_max_thread_size(uint64_t id, int new_max_thread_size) {
            std::lock_guard<std::mutex> lock(mtx);
            max_thread_size[id] = new_max_thread_size;
            if (n_using_thread[id] == THREAD_SIZE_DEFAULT) {
                n_using_thread[id] = 0; // first enable this id
            }
        }

        Thread_pool() {
            set_max_thread_size(THREAD_ID_NONE, THREAD_SIZE_INF);
            set_thread(0);
        }

        Thread_pool(int new_n_thread) {
            set_max_thread_size(THREAD_ID_NONE, THREAD_SIZE_INF);
            set_thread(new_n_thread);
        }

        ~Thread_pool() {
            exit_thread();
        }

        void resize(int new_n_thread) {
            exit_thread();
            set_thread(new_n_thread);
        }

        int size() const {
            return n_set_threads;
        }

        int get_n_idle() const {
            int n_running = n_thread - n_idle;
            return n_canbe_used - n_running;
        }

        int get_max_thread_size(thread_id_t id) {
            return max_thread_size[id];
        }

        int get_n_using_thread(thread_id_t id) {
            return n_using_thread[id];
        }

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
        template<typename F, typename... Args, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
#else
        template<typename F, typename... Args, typename R = typename std::result_of<std::decay_t<F>(std::decay_t<Args>...)>::type>
#endif
        std::future<R> push(bool *pushed, F &&func, const Args &&...args) {
            auto task = std::make_shared<std::packaged_task<R()>>([func, args...]() {
                return func(args...);
            });
            auto future = task->get_future();
            *pushed = push_task(THREAD_ID_NONE, [task]() {(*task)();});
            return future;
        }

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
        template<typename F, typename... Args, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
#else
        template<typename F, typename... Args, typename R = typename std::result_of<std::decay_t<F>(std::decay_t<Args>...)>::type>
#endif
    std::future<R> push(thread_id_t id, bool *pushed, F &&func, const Args &&...args) {
        //if (id != THREAD_ID_NONE) {
        if (n_using_thread[id] >= max_thread_size[id]) {
            *pushed = false;
            return std::future<R>();
        }
        //}
        auto task = std::make_shared<std::packaged_task<R()>>([func, args...]() {
            return func(args...);
        });
        auto future = task->get_future();
        *pushed = push_task(id, [task]() {(*task)();});
        return future;
    }

        void start_idling() {
            std::unique_lock<std::mutex> lock(mtx);
            if (n_canbe_used < n_thread) {
                n_canbe_used.fetch_add(1);
            }
            // int expected = n_canbe_used.load();
            // while (expected < n_thread) {
            //     if (n_canbe_used.compare_exchange_weak(expected, expected + 1)) {
            //         break;
            //     }
            // }
        }

        void finish_idling() {
            n_canbe_used.fetch_sub(1);
        }

    private:

        template<typename F>
        inline bool push_task(thread_id_t id, const F &task) {
            if (!running) {
                throw std::runtime_error("Cannot schedule new task after shutdown.");
            }
            bool pushed = false;
            if (n_idle > 0 && n_using_thread[id] < max_thread_size[id]) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    int n_running = n_thread - n_idle;
                    if (n_idle > 0 && n_using_thread[id] < max_thread_size[id] && n_running < n_canbe_used) {
                        pushed = true;
                        tasks.push(std::make_pair(id, std::function<void()>(task)));
                        --n_idle;
                        condition.notify_one();
                        if (id != THREAD_ID_NONE) {
                            n_using_thread[id].fetch_add(1);
                        }
                    }
                }
            }
            return pushed;
        }

        void worker() {
            thread_id_t id;
            std::function<void()> task;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    ++n_idle;
                    condition.wait(lock, [&] { return !tasks.empty() || !running; });
                    if (!running && tasks.empty()) {
                        return;
                    }
                    if (!tasks.empty()) {
                        id = tasks.front().first;
                        task = std::move(tasks.front().second);
                        tasks.pop();
                    }
                }
                if (task) {
                    task();
                    if (id != THREAD_ID_NONE) {
                        n_using_thread[id].fetch_sub(1);
                    }
                }
            }
        }
};

Thread_pool thread_pool(0);