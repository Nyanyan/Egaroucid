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

#define max_id 1048575

class thread_pool {
    private:
        vector<future<int>> workers;
        vector<bool> busy;
        queue<pair<function<int()>, int>> tasks;
        int result[max_id];
        int worker_ids[max_id];
        unsigned long long id;
        mutex mtx;
    
    public:
        inline void init(){
            this->id = 0;
            cerr << "thread pool initialized" << endl;
        }
        
        inline int push(function<int()> task){
            this->mtx.lock();
            int task_id = this->id++;
            this->id &= max_id;
            //cerr << "push " << task_id << endl;
            this->tasks.push(make_pair(task, task_id));
            this->mtx.unlock();
            for (int i = 0; i < (int)this->workers.size(); ++i){
                if (!this->busy[i] && !this->tasks.empty())
                    execute_task(i);
            }
            while (!this->tasks.empty())
                execute_task_expand();
            return task_id;
        }

        inline int get(int task_id){
            this->mtx.lock();
            int worker_id = this->worker_ids[task_id];
            int res = this->workers[worker_id].get();
            this->busy[worker_id] = false;
            this->mtx.unlock();
            return res;
        }
    
    private:
        inline void execute_task(int i){
            this->mtx.lock();
            this->workers[i] = async(launch::async, this->tasks.front().first);
            this->busy[i] = true;
            this->worker_ids[this->tasks.front().second] = i;
            this->tasks.pop();
            this->mtx.unlock();
        }

        inline void execute_task_expand(){
            this->mtx.lock();
            int i = (int)this->busy.size();
            this->busy.push_back(true);
            this->workers.push_back(async(launch::async, this->tasks.front().first));
            this->worker_ids[this->tasks.front().second] = i;
            this->tasks.pop();
            this->mtx.unlock();
        }
};

thread_pool thread_pool;

inline void multi_thread_init(){
    thread_pool.init();
}