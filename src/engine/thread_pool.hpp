/*
    Egaroucid Project

    @file thread_pool.hpp
        Thread pool for Egaroucid
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
    @notice This code is based on https://github.com/ContentsViewer/nodec/blob/main/nodec/include/nodec/concurrent/thread_pool_executor.hpp , which is published under Apache License 2.0
*/

#pragma once
#include <iostream>
#include <future>
#include <thread>
#include <unordered_set>

// Original code based on
//  * <https://github.com/bshoshany/thread-pool>
//  * <https://github.com/progschj/ThreadPool>
//  * <https://github.com/SandSnip3r/thread-pool>
// Thank you! :)

class Thread_pool {
    private:
        mutable std::mutex mtx;
        bool running;
        int n_thread;
        std::vector<std::function<void()>> tasks{};
        std::vector<bool> tasks_pushed;
        std::unique_ptr<std::thread[]> threads;
        std::condition_variable condition;
        std::vector<int> idle_threads;

    public:
        void set_thread(int new_n_thread){
            if (new_n_thread < 0)
                new_n_thread = 0;
            n_thread = new_n_thread;
            threads.reset(new std::thread[n_thread]);
            idle_threads.clear();
            tasks_pushed.clear();
            for (int i = 0; i < n_thread; ++i){
                threads[i] = std::thread(&Thread_pool::worker, this, i);
                idle_threads.emplace_back(i);
                tasks_pushed.emplace_back(false);
            }
            tasks.resize(n_thread);
            running = true;
        }

        void exit_thread(){
            {
                std::lock_guard<std::mutex> lock(mtx);
                running = false;
            }
            condition.notify_all();
            for (int i = 0; i < n_thread; ++i)
                threads[i].join();
            idle_threads.clear();
            tasks_pushed.clear();
            tasks.resize(0);
            n_thread = 0;
        }

        Thread_pool(){
            set_thread(0);
        }

        Thread_pool(int new_n_thread){
            set_thread(new_n_thread);
        }

        ~Thread_pool(){
            exit_thread();
        }

        void resize(int new_n_thread){
            exit_thread();
            set_thread(new_n_thread);
        }

        int size() const {
            return n_thread;
        }

        int get_n_idle() const {
            return idle_threads.size();
        }

        #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
            template<typename F, typename... Args, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
        #else
            template<typename F, typename... Args, typename R = typename std::result_of<std::decay_t<F>(std::decay_t<Args>...)>::type>
        #endif
        std::future<R> push(bool *pushed, F &&func, const Args &&...args){
            auto task = std::make_shared<std::packaged_task<R()>>([func, args...](){
                return func(args...);
            });
            auto future = task->get_future();
            *pushed = push_task([task](){(*task)();});
            return future;
        }
        /*
        #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
            template<typename F, typename... Args, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
        #else
            template<typename F, typename... Args, typename R = typename std::result_of<std::decay_t<F>(std::decay_t<Args>...)>::type>
        #endif
        std::future<R> push_forced(F &&func, const Args &&...args){
            auto task = std::make_shared<std::packaged_task<R()>>([func, args...](){
                return func(args...);
            });
            auto future = task->get_future();
            push_task_forced([task](){(*task)();});
            return future;
        }
        */

    private:
        template<typename F>
        bool push_task(const F &task){
            if (!running)
                throw std::runtime_error("Cannot schedule new task after shutdown.");
            bool pushed = false;
            mtx.lock();
                if (idle_threads.size()){
                    int use_thread = idle_threads.back();
                    idle_threads.pop_back();
                    tasks[use_thread] = std::function<void()>(task);
                    tasks_pushed[use_thread] = true;
                    pushed = true;
                }
            mtx.unlock();
            if (pushed)
                condition.notify_all();
            return pushed;
        }

        /*
        template<typename F>
        void push_task_forced(const F &task){
            if (!running)
                throw std::runtime_error("Cannot schedule new task after shutdown.");
            mtx.lock();
                tasks.push(std::function<void()>(task));
            mtx.unlock();
            condition.notify_one();
        }
        */

        void worker(const int worker_id){
            for (;;){
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    condition.wait(lock, [&] {return tasks_pushed[worker_id] || !running;});
                    if (!running)
                        return;
                    tasks_pushed[worker_id] = false;
                    task = std::move(tasks[worker_id]);
                }
                task();
                mtx.lock();
                    idle_threads.emplace_back(worker_id);
                mtx.unlock();
            }
        }
};

Thread_pool thread_pool(0);
