#pragma once
#include <iostream>
#include <functional>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>
#include "setting.hpp"
#include "board.hpp"

using namespace std;

class thread_pool {
    public:
        vector<bool> stop;
    private:
        vector<future<int>> workers;
        vector<int> not_busy;
        mutex mtx;
        int worker_size;
    
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
            if (!not_busy.empty()){
                res = not_busy.back();
                execute_task(res, task);
                not_busy.pop_back();
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
            if (this->workers[worker_id].wait_for(chrono::seconds(0)) == future_status::ready){
                res = true;
                *val = this->workers[worker_id].get();
                this->not_busy.push_back(worker_id);
            }
            this->mtx.unlock();
            return res;
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