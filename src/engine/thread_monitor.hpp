/*
    Egaroucid Project

    @file thread_monitor.hpp
        Monitor thread
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "thread_pool.hpp"

#define THREAD_MONITOR_INTERVAL 10 // ms
#define THREAD_MONITOR_TIME 2000 // ms

void thread_monitor(){
    std::chrono::system_clock::time_point strt = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(tp - strt).count() < THREAD_MONITOR_TIME){
        std::chrono::milliseconds(THREAD_MONITOR_INTERVAL);
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tp - strt).count() << " " << thread_pool.get_n_idle() << std::endl;
        tp = std::chrono::system_clock::now();
    }
}

void start_thread_monitor(){
    int n_thread = thread_pool.size();
    thread_pool.resize(n_thread + 1);
    std::cerr << "thread pool resized for monitor from " << n_thread << " to " << n_thread + 1 << std::endl;
    thread_pool.push_forced(&thread_monitor);
}
