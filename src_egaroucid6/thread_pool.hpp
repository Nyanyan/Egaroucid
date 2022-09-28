#pragma once
#include <iostream>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include "setting.hpp"
#include "board.hpp"
#include "parallel.hpp"
#include "search.hpp"

using namespace std;

#define THREAD_POOL_TASK_ID_SIZE 0x20000000 //1048576
#define THREAD_POOL_ERASE_STEP 4096

// refer to https://zenn.dev/rita0222/articles/13953a5dfb9698

class Thread_pool{
    private:
        class Worker{
            private:
                Thread_pool* parent_{nullptr};
                int index_{-1};
                function<Parallel_task()> current_task_{};
                int32_t current_task_id{-1};
                bool has_task{false};
                bool is_requested_termination{false};
                thread thread_;
                mutex mutex_;
                condition_variable cond_;
            
            public:
                Worker() : thread_([this]() { proc_worker(); }) {}

                Worker(const Worker &worker) : thread_([this]() { proc_worker(); }) {}

                ~Worker() {
                    wait_until_idle();
                    request_termination();
                    if (thread_.joinable()) {
                        thread_.join();
                    }
                };

                void initialize(Thread_pool* parent, int index) {
                    unique_lock<mutex> lock(mutex_);
                    parent_ = parent;
                    index_ = index;
                    cond_.notify_all();
                }
                
                void wait_until_idle() {
                    unique_lock<mutex> lock(mutex_);
                    cond_.wait(lock, [this]() { return !current_task_ || is_requested_termination; });
                }
                
                void request_termination() {
                    unique_lock<mutex> lock(mutex_);
                    is_requested_termination = true;
                    cond_.notify_all();
                }

                bool run(function<Parallel_task()> &task, const int32_t task_id){
                    unique_lock<mutex> lock(mutex_);
                    if (has_task)
                        return false;
                    //cerr << "CHILD receive " << index_ << " " << task_id << endl;
                    current_task_ = {};
                    current_task_ = task;
                    current_task_id = task_id;
                    has_task = true;
                    cond_.notify_all();
                    return true;
                }
            
            private:
                void proc_worker() {
                    {
                        unique_lock<mutex> lock(mutex_);
                        cond_.wait(lock, [this]() { return parent_ != nullptr && index_ >= 0; });
                    }
                    while (!is_requested_termination){
                        {
                            unique_lock<mutex> lock(mutex_);
                            cond_.wait(lock, [this]() { return is_requested_termination || has_task;});
                        }
                        if (has_task){
                            //cerr << "CHILD do " << index_ << " " << current_task_id << endl;
                            Parallel_task res = current_task_();
                            //cerr << "CHILD fin " << index_ << " " << current_task_id << endl;
                            parent_->notify_finish(index_, current_task_id, res);
                            has_task = false;
                        }
                    }
                }
        };

    private:
        uint32_t n_threads;
        vector<Worker> workers;
        vector<bool> worker_busy;
        unordered_map<int32_t, Parallel_task> answers;
        bool is_requested_termination{false};
        mutex mutex_;
        condition_variable cond_;
        int32_t task_id{0};
        int n_empty_thread{0};
    
    public:
        Thread_pool(const int n){
            n_threads = n;
            workers.resize(n_threads);
            worker_busy.resize(n_threads);
            for (int i = 0; i < n_threads; ++i){
                workers[i].initialize(this, i);
                worker_busy[i] = false;
            }
            n_empty_thread = n_threads;
        }

        void terminate() {
            is_requested_termination = true;
            //wait_until_idle();
            request_termination();
        };

        void resize(int n){
            cerr << "resizing" << endl;
            is_requested_termination = true;
            request_termination();
            cerr << "terminated" << endl;
            unique_lock<mutex> lock(mutex_);
            workers.clear();
            n_threads = n;
            workers.resize(n_threads);
            worker_busy.resize(n_threads);
            for (int i = 0; i < n_threads; ++i){
                workers[i].initialize(this, i);
                worker_busy[i] = false;
            }
            n_empty_thread = n_threads;
            is_requested_termination = false;
        }

        int size(){
            return n_threads;
        }

        int32_t try_run(function<Parallel_task()> &&task) {
            unique_lock<mutex> lock(mutex_);
            if (is_requested_termination || n_empty_thread == 0)
                return -1;
            for (int i = 0; i < n_threads; ++i){
                if (!worker_busy[i]){
                    ++task_id;
                    if (task_id >= THREAD_POOL_TASK_ID_SIZE)
                        task_id %= THREAD_POOL_TASK_ID_SIZE;
                    if (workers[i].run(task, task_id)){
                        --n_empty_thread;
                        worker_busy[i] = true;
                        //cerr << "PARENT send " << i << " " << task_id << endl;
                        return task_id;
                    }
                    if (task_id == 0)
                        task_id = THREAD_POOL_TASK_ID_SIZE - 1;
                    else
                        --task_id;
                }
            }
            return -1;
        }

        void notify_finish(const int worker_idx, const int task_id, Parallel_task res){
            unique_lock<mutex> lock(mutex_);
            //cerr << "PARENT receive " << worker_idx << " " << task_id << endl;
            answers[task_id] = res;
            //answer_available[task_id] = true;
            worker_busy[worker_idx] = false;
            ++n_empty_thread;
        }

        bool get(int32_t id, Parallel_task *res){
            unique_lock<mutex> lock(mutex_);
            if (answers.find(id) == answers.end())
                return false;
            *res = answers[id].copy();
            answers.erase(id);
            return true;
        }

        int get_n_empty_thread() const{
            return n_empty_thread;
        }

    private:
        void wait_until_idle() {
            {
                unique_lock<mutex> lock(mutex_);
                cond_.wait(lock, [this]() { return is_requested_termination; });
            }
            for (auto& worker : workers)
                worker.wait_until_idle();
        }

        void request_termination() {
            {
                unique_lock<mutex> lock(mutex_);
                is_requested_termination = true;
                cond_.notify_all();
            }
            for (auto& worker : workers)
                worker.request_termination();
        }
};

Thread_pool thread_pool(0);
