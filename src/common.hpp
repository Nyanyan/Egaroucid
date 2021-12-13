#pragma once
#include <chrono>
#include <random>
#include "setting.hpp"
#include "board.hpp"

using namespace std;

#define inf 100000000
#define n_phases 15
constexpr int phase_n_stones = 60 / n_phases;

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

mt19937 raw_myrandom(tim());

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int myrandrange(int s, int e){
    return s +(int)((e - s) * myrandom());
}

inline int calc_phase_idx(const board *b){
    return (b->n - 4) / phase_n_stones;
}