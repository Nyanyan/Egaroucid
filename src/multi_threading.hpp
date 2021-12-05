#pragma once
#include <iostream>
#include <functional>
#include <thread>
#include <future>
#include <queue>
#include <chrono>
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
    
    public:
        inline void init(){
            this->id = 0;
            cerr << "thread pool initialized" << endl;
        }
        
        inline int push(function<int()> task){
            int task_id = this->id++;
            this->id &= max_id;
            //cerr << "push " << task_id << endl;
            this->tasks.push(make_pair(task, task_id));
            for (int i = 0; i < (int)this->workers.size(); ++i){
                if (!this->busy[i] && !this->tasks.empty())
                    execute_task(i);
            }
            while (!this->tasks.empty())
                execute_task_expand();
            return task_id;
        }

        inline bool get(int task_id, int *res){
            int worker_id = worker_ids[task_id];
            //if (workers[worker_id].wait_for(chrono::milliseconds(0)) == future_status::ready){
            *res = this->workers[worker_id].get();
            this->busy[worker_id] = false;
            cerr << "done " << worker_id << endl;
            return true;
            //}
            //return false;
        }
    
    private:
        inline void execute_task(int i){
            this->workers[i] = async(launch::async, this->tasks.front().first);
            this->busy[i] = true;
            this->worker_ids[this->tasks.front().second] = i;
            this->tasks.pop();
            cerr << "execute " << i << endl;
        }

        inline void execute_task_expand(){
            int i = (int)this->busy.size();
            this->busy.push_back(true);
            this->workers.push_back(async(launch::async, this->tasks.front().first));
            this->worker_ids[this->tasks.front().second] = i;
            this->tasks.pop();
            cerr << "NEW execute " << i << endl;
        }
};

thread_pool thread_pool;

inline void multi_thread_init(){
    thread_pool.init();
}