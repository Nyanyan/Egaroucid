#pragma once
#include <iostream>
#include <functional>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>
#include "setting.hpp"
#include "board.hpp"
#include "CTPL/ctpl_stl.h"

using namespace std;

#define n_threads 6

class thread_pool {
    private:
        ctpl::thread_pool p(n_thread);
    
    public:
        inline void init(){
            this->worker_size = 0;
            cerr << "thread pool initialized" << endl;
        }

        inline int get_worker_id(){
            this->mtx.lock();
            int res = -1;
            if (!not_busy.empty()){
                res = not_busy.back();
                not_busy.pop_back();
            } else
                res = this->worker_size;
            return res;
        }

        inline void push_id(function<int()> task, int worker_id){
            if (worker_id < this->worker_size)
                execute_task(worker_id, task);
            else
                execute_task_expand(task);
            this->mtx.unlock();
        }
        
        inline int push(function<int()> task){
            this->mtx.lock();
            int res = -1;
            if (!this->not_busy.empty()){
                res = this->not_busy.back();
                execute_task(res, task);
                this->not_busy.pop_back();
            } else
                res = execute_task_expand(task);
            this->mtx.unlock();
            return res;
        }

        inline int get(int worker_id){
            this->mtx.lock();
            int res = this->workers[worker_id].get();
            this->not_busy.push_back(worker_id);
            this->mtx.unlock();
            return res;
        }

        inline bool get_check(int worker_id, int *val){
            this->mtx.lock();
            bool res = false;
            if (this->workers[worker_id].wait_for(seconds0) == future_status::ready){
                res = true;
                *val = this->workers[worker_id].get();
                this->not_busy.push_back(worker_id);
            }
            this->mtx.unlock();
            return res;
        }

        inline int get_worker_size(){
            return this->worker_size;
        }
    
    private:
        inline void execute_task(int i, function<int()> task){
            this->stop[i] = false;
            this->workers[i] = async(launch::async, task);
        }

        inline int execute_task_expand(function<int()> task){
            int i = this->worker_size;
            this->stop.push_back(false);
            this->workers.push_back(async(launch::async, task));
            ++this->worker_size;
            return i;
        }
};

thread_pool thread_pool;

inline void thread_pool_init(){
    thread_pool.init();
}