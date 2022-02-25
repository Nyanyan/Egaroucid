#pragma once
#include <iostream>
#include <chrono>

using namespace std;

#define HW 8
#define HW_M1 7
#define HW2 64
#define HW2_M1 63

#define N_8BIT 256
#define N_DIAG_LINE 11
#define N_DIAG_LINE_M1 10

inline uint64_t tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}
