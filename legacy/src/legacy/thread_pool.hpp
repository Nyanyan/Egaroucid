#pragma once
#include <iostream>
#include <thread>
#include <future>
#include <functional>
#include <queue>
#include <mutex>
#include "setting.hpp"
#include "board.hpp"

#define default_workers 2

#define task_doing 0
#define task_end -1
#define mid_search 1
#define end_search 2

using namespace std;

int nega_alpha_ordering(board *b, bool skipped, const int depth, int alpha, int beta, int use_multi_thread, int worker_id);
int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, int use_multi_thread, int worker_id);
void thread_execution(int worker_id);

struct task_info{
    int executable;
    board *b;
    bool skipped;
    int depth;
    int alpha;
    int beta;
    int use_multi_thread;
    int worker_id;
    int val;
};

class thread_pool {
    public:
        vector<bool> stop;
        vector<task_info> task_communicator;
        mutex mtx;

    private:
        vector<future<void>> workers;
        vector<int> not_busy;
        int worker_size;
        int n_pushed;

    public:
        inline void init(){
            this->worker_size = default_workers;
            for (int i = 0; i < default_workers; ++i){
                task_info tmp_communicator;
                tmp_communicator.executable = -1;
                task_communicator.push_back(tmp_communicator);
                this->stop.push_back(false);
                this->workers.push_back(async(launch::async, bind(thread_execution, i)));
                this->not_busy.push_back(i);
            }
            this->n_pushed = 0;
            cerr << "thread pool initialized" << endl;
        }

        inline int get_worker_id(){
            this->mtx.lock();
            int res = -1;
            //if (not_busy.empty())
            //    return -1;
            if (!not_busy.empty()){
                res = not_busy.back();
                not_busy.pop_back();
            } else
                res = this->worker_size;
            return res;
        }

        inline void push_id(int task_type, board *b, bool skipped, int depth, int alpha, int beta, int use_multi_thread, int worker_id){
            if (worker_id < this->worker_size)
                execute_task(task_type, b, skipped, depth, alpha, beta, use_multi_thread, worker_id);
            else
                execute_task_expand(task_type, b, skipped, depth, alpha, beta, use_multi_thread, worker_id);
            this->mtx.unlock();
        }

        /*
        inline int get(int worker_id){
            this->mtx.lock();
            int res;
            while (task_communicator[worker_id].executable != task_end);
            res = task_communicator[worker_id].val;
            this->not_busy.push_back(worker_id);
            this->mtx.unlock();
            return res;
        }
        */

        inline bool get_check(int worker_id, int *val){
            this->mtx.lock();
            bool res = false;
            if (task_communicator[worker_id].executable == task_end){
                res = true;
                *val = task_communicator[worker_id].val;
                this->not_busy.push_back(worker_id);
            }
            this->mtx.unlock();
            return res;
        }

        inline int get_worker_size(){
            return this->worker_size;
        }

        inline int get_n_pushed(){
            return this->n_pushed;
        }
    
    private:
        inline void execute_task(int task_type, board *b, bool skipped, int depth, int alpha, int beta, int use_multi_thread, int worker_id){
            this->stop[worker_id] = false;
            task_communicator[worker_id].b = b;
            task_communicator[worker_id].skipped = skipped;
            task_communicator[worker_id].depth = depth;
            task_communicator[worker_id].alpha = alpha;
            task_communicator[worker_id].beta = beta;
            task_communicator[worker_id].use_multi_thread = use_multi_thread;
            task_communicator[worker_id].worker_id = worker_id;
            task_communicator[worker_id].executable = task_type;
            //cerr << worker_size << " " << worker_id << endl;
            ++this->n_pushed;
        }

        inline void execute_task_expand(int task_type, board *b, bool skipped, int depth, int alpha, int beta, int use_multi_thread, int worker_id){
            task_info tmp_communicator;
            tmp_communicator.executable = -1;
            task_communicator.push_back(tmp_communicator);
            this->stop.push_back(false);
            this->workers.push_back(async(launch::async, bind(thread_execution, worker_id)));
            ++this->worker_size;
            execute_task(task_type, b, skipped, depth, alpha, beta, use_multi_thread, worker_id);
            ++this->n_pushed;
        }
};

thread_pool thread_pool;

void thread_execution(int worker_id){
    int v;
    for (;;){
        thread_pool.mtx.lock();
        if (thread_pool.task_communicator[worker_id].executable == mid_search){
            //cerr << "midsearch" << endl;
            thread_pool.task_communicator[worker_id].executable = task_doing;
            thread_pool.mtx.unlock();
            v = nega_alpha_ordering(thread_pool.task_communicator[worker_id].b, thread_pool.task_communicator[worker_id].skipped, thread_pool.task_communicator[worker_id].depth, thread_pool.task_communicator[worker_id].alpha, thread_pool.task_communicator[worker_id].beta, thread_pool.task_communicator[worker_id].use_multi_thread, thread_pool.task_communicator[worker_id].worker_id);
            thread_pool.mtx.lock();
            thread_pool.task_communicator[worker_id].val = v;
            thread_pool.task_communicator[worker_id].executable = task_end;
            thread_pool.mtx.unlock();
        } else if (thread_pool.task_communicator[worker_id].executable == end_search){
            thread_pool.task_communicator[worker_id].executable = task_doing;
            thread_pool.mtx.unlock();
            v = nega_alpha_ordering_final(thread_pool.task_communicator[worker_id].b, thread_pool.task_communicator[worker_id].skipped, thread_pool.task_communicator[worker_id].depth, thread_pool.task_communicator[worker_id].alpha, thread_pool.task_communicator[worker_id].beta, thread_pool.task_communicator[worker_id].use_multi_thread, thread_pool.task_communicator[worker_id].worker_id);
            thread_pool.mtx.lock();
            thread_pool.task_communicator[worker_id].val = v;
            thread_pool.task_communicator[worker_id].executable = task_end;
            thread_pool.mtx.unlock();
        } else
            thread_pool.mtx.unlock();
    }
}

inline void thread_pool_init(){
    thread_pool.init();
}