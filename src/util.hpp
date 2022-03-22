#pragma once
#include <iostream>
#include "board.hpp"

Board input_board(){
    Board res;
    char elem;
    int player;
    cin >> player;
    res.player = 0;
    res.opponent = 0;
    for (int i = 0; i < HW2; ++i){
        cin >> elem;
        if (elem == '0'){
            if (player == BLACK)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        } else if (elem == '1'){
            if (player == WHITE)
                res.player |= 1ULL << (HW2_M1 - i);
            else
                res.opponent |= 1ULL << (HW2_M1 - i);
        }
    }
    res.p = player;
    res.parity = 0;
    res.n = HW2;
    uint64_t empties = ~(res.player | res.opponent);
    for (int i = 0; i < HW2; ++i){
        if (1 & (empties >> i)){
            res.parity ^= cell_div4[i];
            --res.n;
        }
    }
    res.print();
    return res;
}

string idx_to_coord(int idx){
    int y = HW_M1 - idx / HW;
    int x = HW_M1 - idx % HW;
    const string x_coord = "abcdefgh";
    return x_coord[x] + to_string(y + 1);
}

inline int value_to_score_int(int v){
    #if EVALUATION_STEP_WIDTH_MODE == 0
        return v;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        return v * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        if (v > 0)
            ++v;
        else if (v < 0)
            --v;
        return v / 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        if (v > 0)
            v += 2;
        else if (v < 0)
            v -= 2;
        return v / 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        if (v > 0)
            v += 4;
        else if (v < 0)
            v -= 4;
        return v / 8;
    #endif
}

inline double value_to_score_double(int v){
    double vd = (double)v;
    #if EVALUATION_STEP_WIDTH_MODE == 0
        return vd;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        return vd * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        return vd / 2.0;
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        return vd / 4.0;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        return vd / 8.0;
    #endif
}

inline int score_to_value(int v){
    #if EVALUATION_STEP_WIDTH_MODE == 0
        return v;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        return v / 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        return v * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        return v * 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        return v * 8;
    #endif
}

inline int score_to_value(double v){
    #if EVALUATION_STEP_WIDTH_MODE == 0
        return v;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        return v / 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        return v * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        return v * 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        return v * 8;
    #endif
}