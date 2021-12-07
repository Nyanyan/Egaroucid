#pragma once
#include <iostream>
#include <functional>
#include <thread>
#include <future>
#include <queue>
#include <chrono>
#include <mutex>
#include "setting.hpp"
#include "board.hpp"

using namespace std;

class thread_pool {
    private:
        vector<future<int>> workers;
        vector<bool> busy;
        mutex mtx;
        int worker_size;
    
    public:
        inline void init(){
            this->worker_size = 0;
            cerr << "thread pool initialized" << endl;
        }
        
        inline int push(function<int()> task){
            bool flag = true;
            int res = -1;
            for (int i = 0; i < this->worker_size; ++i){
                if (!this->busy[i] && flag){
                    execute_task(i, task);
                    res = i;
                    flag = false;
                    break;
                }
            }
            if (flag)
                res = execute_task_expand(task);
            return res;
        }

        inline int get(int worker_id){
            this->mtx.lock();
            int res = this->workers[worker_id].get();
            this->busy[worker_id] = false;
            this->mtx.unlock();
            return res;
        }
    
    private:
        inline void execute_task(int i, function<int()> task){
            this->mtx.lock();
            this->workers[i] = async(launch::async, task);
            this->busy[i] = true;
            this->mtx.unlock();
        }

        inline int execute_task_expand(function<int()> task){
            this->mtx.lock();
            int i = this->worker_size;
            this->busy.push_back(true);
            this->workers.push_back(async(launch::async, task));
            ++this->worker_size;
            this->mtx.unlock();
            return i;
        }
};

thread_pool thread_pool;

inline void multi_thread_init(){
    thread_pool.init();
}