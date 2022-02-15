#pragma once
#include <iostream>
//#include <functional>
//#include <future>
#include "setting.hpp"
#include "board.hpp"

// from https://github.com/vit-vit/CTPL
#if USE_BOOST
    #include "CTPL/ctpl.h"
#else
    #include "CTPL/ctpl_stl.h" // Please use this if you don't have boost
#endif

using namespace std;

//unsigned int n_threads = thread::hardware_concurrency();

//ctpl::thread_pool thread_pool(n_threads);
ctpl::thread_pool thread_pool(1);