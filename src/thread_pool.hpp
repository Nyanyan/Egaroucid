#pragma once
#include <iostream>
//#include <functional>
//#include <future>
#include "setting.hpp"
#include "board.hpp"
#include "CTPL/ctpl_stl.h"

using namespace std;

#define n_threads 6

/*
class egaroucid_thread_pool {
    private:
        ctpl::thread_pool p(int nThreads = n_threads);
    
    public:
        inline future<int> push(function<int()> task){
            return p.push(task);
        }
};

egaroucid_thread_pool thread_pool;
*/
ctpl::thread_pool thread_pool(n_threads);