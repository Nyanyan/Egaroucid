/*
    Egaroucid Project

    @file thread_pool.hpp
        Thread pool for Egaroucid
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
    @notice This code is based on https://github.com/ContentsViewer/nodec/blob/main/nodec/include/nodec/concurrent/thread_pool_executor.hpp , which is published under Apache License 2.0
*/

#pragma once
#include <iostream>
#include <future>
#include <queue>
#include <thread>
#include <atomic>
#include <functional>

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
        //std::atomic<int> n_idle;
        int n_idle;
        std::queue<std::function<void()>> tasks{};
        std::unique_ptr<std::thread[]> threads;
        std::condition_variable condition;
        //std::atomic<int> n_using_tasks;

    public:
        void set_thread(int new_n_thread) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                //n_using_tasks.store(0);
                if (new_n_thread < 0) {
                    new_n_thread = 0;
                }
                n_thread = new_n_thread;
                threads.reset(new std::thread[n_thread]);
                for (int i = 0; i < n_thread; ++i) {
                    threads[i] = std::thread(&Thread_pool::worker, this);
                }
                running = true;
                n_idle = 0;
            }
        }

        void exit_thread() {
            {
                std::lock_guard<std::mutex> lock(mtx);
                running = false;
            }
            condition.notify_all();
            for (int i = 0; i < n_thread; ++i) {
                threads[i].join();
            }
            n_thread = 0;
            n_idle = 0;
        }

        Thread_pool() {
            set_thread(0);
        }

        Thread_pool(int new_n_thread) {
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
            return n_thread;
        }

        int get_n_idle() const {
            return n_idle;
        }

        /*
        void reset_unavailable() {
            if (n_idle == n_thread && n_using_tasks.load() == 0) {
                bool start_flag = false;
                std::vector<std::future<void>> futures;
                bool need_to_reset = false;
                for (int i = 0; i < n_thread; ++i) {
                    bool pushed;
                    futures.emplace_back(push(&pushed, std::bind(reset_unavailable_task, &start_flag)));
                    if (!pushed) {
                        futures.pop_back();
                        need_to_reset = true;
                    }
                }
                start_flag = true;
                for (std::future<void> &f: futures) {
                    f.get();
                }
                if (need_to_reset) {
                    std::cerr << "reset unavailable threads" << std::endl;
                    resize(n_thread);
                }
            }
        }
        */

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
            *pushed = push_task([task]() {(*task)();});
            return future;
        }

        /*
        void tell_start_using() {
            n_using_tasks.fetch_add(1);
        }

        void tell_finish_using() {
            n_using_tasks.fetch_sub(1);
        }
        */

    private:

        template<typename F>
        inline bool push_task(const F &task) {
            if (!running) {
                throw std::runtime_error("Cannot schedule new task after shutdown.");
            }
            bool pushed = false;
            if (n_idle > 0) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    if (n_idle > 0) {
                        pushed = true;
                        tasks.push(std::function<void()>(task));
                        --n_idle;
                        condition.notify_one();
                    }
                }
            }
            return pushed;
        }

        void worker() {
            std::function<void()> task;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    ++n_idle;
                    condition.wait(lock, [&] {return !tasks.empty() || !running;});
                    if (!running) {
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        }
};

Thread_pool thread_pool(0);