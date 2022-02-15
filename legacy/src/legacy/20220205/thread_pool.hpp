#pragma once
#include <iostream>
//#include <functional>
//#include <future>
#include "setting.hpp"
#include "board.hpp"
#include "CTPL/ctpl_stl.h" // from https://github.com/vit-vit/CTPL

using namespace std;

//unsigned int n_threads = thread::hardware_concurrency();

//ctpl::thread_pool thread_pool(n_threads);
ctpl::thread_pool thread_pool(1);