#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "mobility.hpp"
#include "transpose_table.hpp"

#define N_MOVE_ORDERING_PATTERNS 10
#define MAX_MOVE_ORDERING_EVALUATE_IDX 65536
#define MOVE_ORDERING_PHASE_DIV 10
#define N_MOVE_ORDERING_PHASE 6

#define p40 1
#define p41 4
#define p42 16
#define p43 64
#define p44 256
#define p45 1024
#define p46 4096
#define p47 16384
#define p48 65536

#define W_BEST1_MOVE 900000000
#define W_BEST2_MOVE 800000000
#define W_BEST3_MOVE 700000000

#define W_CACHE_HIT 1000000
#define W_CACHE_HIGH 10000

#define N_MID_WEIGHT 21 - MID_TO_END_DEPTH + 1

constexpr int W_CACHE_VALUE[N_MID_WEIGHT] = {74, 74, 74, 74, 74, 74, 74, 74, 74, 74};
constexpr int W_CELL_WEIGHT[N_MID_WEIGHT] = {1, 1, 1, 1, 4, 5, 5, 5, 5, 5};
constexpr int W_EVALUATE[N_MID_WEIGHT] = {5, 5, 5, 10, 18, 20, 20, 22, 20, 20};
constexpr int W_MOBILITY[N_MID_WEIGHT] = {61, 61, 59, 61, 61, 60, 63, 61, 61, 61};
constexpr int W_SURROUND[N_MID_WEIGHT] = {5, 5, 4, 2, 11, 12, 2, 5, 5, 5};
constexpr int W_PARITY[N_MID_WEIGHT] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

#define W_END_MOBILITY 29
#define W_END_SURROUND 10
#define W_END_PARITY 13

/*
short move_ordering_pattern_arr[N_MOVE_ORDERING_PHASE][N_MOVE_ORDERING_PATTERNS][MAX_MOVE_ORDERING_EVALUATE_IDX];

inline bool move_ordering_init(){
    FILE* fp;
    if (fopen_s(&fp, "resources/move_ordering.egmo", "rb") != 0){
        cerr << "can't open eval.egev" << endl;
        return false;
    }
    int phase_idx, pattern_idx;
    const size_t eval_sizes[N_MOVE_ORDERING_PATTERNS] = {65536, 65536, 65536, 65536, 64, 256, 1024, 4096, 16384, 65536};
    for (phase_idx = 0; phase_idx < N_MOVE_ORDERING_PHASE; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_MOVE_ORDERING_PATTERNS; ++pattern_idx){
            if (fread(move_ordering_pattern_arr[phase_idx][pattern_idx], 2, eval_sizes[pattern_idx], fp) < eval_sizes[pattern_idx]){
                cerr << "move_ordering.egmo broken" << endl;
                fclose(fp);
                return false;
            }
        }
    }
    cerr << "move ordering initialized" << endl;
    return true;
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2, const int p3){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44);
}


inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45) + 
        ((pop_digit(f, p6) * 3 + pop_digit(o, p6) * 2 + pop_digit(p, p6)) * p46);
}

inline int pick_pattern(unsigned long long p, unsigned long long o, unsigned long long f, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return 
        ((pop_digit(f, p0) * 3 + pop_digit(o, p0) * 2 + pop_digit(p, p0)) * p40) + 
        ((pop_digit(f, p1) * 3 + pop_digit(o, p1) * 2 + pop_digit(p, p1)) * p41) + 
        ((pop_digit(f, p2) * 3 + pop_digit(o, p2) * 2 + pop_digit(p, p2)) * p42) + 
        ((pop_digit(f, p3) * 3 + pop_digit(o, p3) * 2 + pop_digit(p, p3)) * p43) + 
        ((pop_digit(f, p4) * 3 + pop_digit(o, p4) * 2 + pop_digit(p, p4)) * p44) + 
        ((pop_digit(f, p5) * 3 + pop_digit(o, p5) * 2 + pop_digit(p, p5)) * p45) + 
        ((pop_digit(f, p6) * 3 + pop_digit(o, p6) * 2 + pop_digit(p, p6)) * p46) + 
        ((pop_digit(f, p7) * 3 + pop_digit(o, p7) * 2 + pop_digit(p, p7)) * p47);
}

inline int move_evaluate(Board *board, Mobility *mob, int move_ordering_phase){
    unsigned long long p, o;
    if (board->p == BLACK){
        p = board->b;
        o = board->w & (~mob->flip);
    } else{
        p = board->w;
        o = board->b & (~mob->flip);
    }
    return 
        move_ordering_pattern_arr[move_ordering_phase][0][pick_pattern(p, o, mob->flip, 0, 1, 2, 3, 4, 5, 6, 7)] + 
        move_ordering_pattern_arr[move_ordering_phase][0][pick_pattern(p, o, mob->flip, 0, 8, 16, 24, 32, 40, 48, 56)] + 
        move_ordering_pattern_arr[move_ordering_phase][0][pick_pattern(p, o, mob->flip, 7, 15, 23, 31, 39, 47, 55, 63)] + 
        move_ordering_pattern_arr[move_ordering_phase][0][pick_pattern(p, o, mob->flip, 56, 57, 58, 59, 60, 61, 62, 63)] + 

        move_ordering_pattern_arr[move_ordering_phase][1][pick_pattern(p, o, mob->flip, 8, 9, 10, 11, 12, 13, 14, 15)] + 
        move_ordering_pattern_arr[move_ordering_phase][1][pick_pattern(p, o, mob->flip, 1, 9, 17, 25, 33, 41, 49, 57)] + 
        move_ordering_pattern_arr[move_ordering_phase][1][pick_pattern(p, o, mob->flip, 6, 14, 22, 30, 38, 46, 54, 62)] + 
        move_ordering_pattern_arr[move_ordering_phase][1][pick_pattern(p, o, mob->flip, 48, 49, 50, 51, 52, 53, 54, 55)] + 

        move_ordering_pattern_arr[move_ordering_phase][2][pick_pattern(p, o, mob->flip, 16, 17, 18, 19, 20, 21, 22, 23)] + 
        move_ordering_pattern_arr[move_ordering_phase][2][pick_pattern(p, o, mob->flip, 2, 10, 18, 26, 34, 42, 50, 58)] + 
        move_ordering_pattern_arr[move_ordering_phase][2][pick_pattern(p, o, mob->flip, 5, 13, 21, 29, 37, 45, 53, 61)] + 
        move_ordering_pattern_arr[move_ordering_phase][2][pick_pattern(p, o, mob->flip, 40, 41, 42, 43, 44, 45, 46, 47)] + 

        move_ordering_pattern_arr[move_ordering_phase][3][pick_pattern(p, o, mob->flip, 24, 25, 26, 27, 28, 29, 30, 31)] + 
        move_ordering_pattern_arr[move_ordering_phase][3][pick_pattern(p, o, mob->flip, 3, 11, 19, 27, 35, 43, 51, 59)] + 
        move_ordering_pattern_arr[move_ordering_phase][3][pick_pattern(p, o, mob->flip, 4, 12, 20, 28, 36, 44, 52, 60)] + 
        move_ordering_pattern_arr[move_ordering_phase][3][pick_pattern(p, o, mob->flip, 32, 33, 34, 35, 36, 37, 38, 39)] + 
        
        move_ordering_pattern_arr[move_ordering_phase][4][pick_pattern(p, o, mob->flip, 5, 14, 23)] + 
        move_ordering_pattern_arr[move_ordering_phase][4][pick_pattern(p, o, mob->flip, 2, 9, 16)] + 
        move_ordering_pattern_arr[move_ordering_phase][4][pick_pattern(p, o, mob->flip, 40, 49, 58)] + 
        move_ordering_pattern_arr[move_ordering_phase][4][pick_pattern(p, o, mob->flip, 61, 54, 47)] + 

        move_ordering_pattern_arr[move_ordering_phase][5][pick_pattern(p, o, mob->flip, 4, 13, 22, 31)] + 
        move_ordering_pattern_arr[move_ordering_phase][5][pick_pattern(p, o, mob->flip, 3, 10, 17, 24)] + 
        move_ordering_pattern_arr[move_ordering_phase][5][pick_pattern(p, o, mob->flip, 32, 41, 50, 59)] + 
        move_ordering_pattern_arr[move_ordering_phase][5][pick_pattern(p, o, mob->flip, 60, 53, 46, 39)] + 

        move_ordering_pattern_arr[move_ordering_phase][6][pick_pattern(p, o, mob->flip, 3, 12, 21, 30, 39)] + 
        move_ordering_pattern_arr[move_ordering_phase][6][pick_pattern(p, o, mob->flip, 4, 11, 18, 25, 32)] + 
        move_ordering_pattern_arr[move_ordering_phase][6][pick_pattern(p, o, mob->flip, 24, 33, 42, 51, 60)] + 
        move_ordering_pattern_arr[move_ordering_phase][6][pick_pattern(p, o, mob->flip, 59, 52, 45, 38, 31)] + 

        move_ordering_pattern_arr[move_ordering_phase][7][pick_pattern(p, o, mob->flip, 2, 11, 20, 29, 38, 47)] + 
        move_ordering_pattern_arr[move_ordering_phase][7][pick_pattern(p, o, mob->flip, 5, 12, 19, 26, 33, 40)] + 
        move_ordering_pattern_arr[move_ordering_phase][7][pick_pattern(p, o, mob->flip, 16, 25, 34, 43, 52, 61)] + 
        move_ordering_pattern_arr[move_ordering_phase][7][pick_pattern(p, o, mob->flip, 58, 51, 44, 37, 30, 23)] + 

        move_ordering_pattern_arr[move_ordering_phase][8][pick_pattern(p, o, mob->flip, 1, 10, 19, 28, 37, 46, 55)] + 
        move_ordering_pattern_arr[move_ordering_phase][8][pick_pattern(p, o, mob->flip, 6, 13, 20, 27, 34, 41, 48)] + 
        move_ordering_pattern_arr[move_ordering_phase][8][pick_pattern(p, o, mob->flip, 8, 17, 26, 35, 44, 53, 62)] + 
        move_ordering_pattern_arr[move_ordering_phase][8][pick_pattern(p, o, mob->flip, 57, 50, 43, 36, 29, 22, 15)] + 

        move_ordering_pattern_arr[move_ordering_phase][9][pick_pattern(p, o, mob->flip, 0, 9, 18, 27, 36, 45, 54, 63)] + 
        move_ordering_pattern_arr[move_ordering_phase][9][pick_pattern(p, o, mob->flip, 7, 14, 21, 28, 35, 42, 49, 56)];
}
*/
inline void move_evaluate_simple(Search *search, Mobility *mob, const int best_moves[], const int weight_idx){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else{
        mob->value += cell_weight[mob->pos] * W_CELL_WEIGHT[weight_idx];
        if (search->board.parity & cell_div4[mob->pos])
            mob->value += W_PARITY[weight_idx];
        search->board.move(mob);
            int l, u;
            parent_transpose_table.get_prev(&search->board, search->board.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
            if (u != INF)
                mob->value += W_CACHE_HIT + W_CACHE_HIGH - u * W_CACHE_VALUE[weight_idx];
            else if (l != -INF)
                mob->value += W_CACHE_HIT - l * W_CACHE_VALUE[weight_idx];
            mob->value += -mid_evaluate(&search->board) * W_EVALUATE[weight_idx];
            if (search->board.p == BLACK)
                mob->value += calc_surround(search->board.b, ~(search->board.b | search->board.w)) * W_SURROUND[weight_idx];
            else
                mob->value += calc_surround(search->board.w, ~(search->board.b | search->board.w)) * W_SURROUND[weight_idx];
            /*
            int stab0, stab1;
            calc_stability_fast(&search->board, &stab0, &stab1);
            if (search->board.p == BLACK)
                mob->value += stab1 * W_STABILITY;
            else
                mob->value += stab0 * W_STABILITY;
            */
            mob->value -= pop_count_ull(search->board.mobility_ull()) * W_MOBILITY[weight_idx];
        search->board.undo(mob);
    }
}

bool cmp_move_ordering(Mobility &a, Mobility &b){
    return a.value > b.value;
}
/*
inline void move_ordering(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    int move_ordering_phase = (search->board.n - 4) / MOVE_ORDERING_PHASE_DIV;
    for (Mobility &mob: move_list){
        if (mob.pos == best_moves[0])
            mob.value = W_BEST1_MOVE;
        else if (mob.pos == best_moves[1])
            mob.value = W_BEST2_MOVE;
        else if (mob.pos == best_moves[2])
            mob.value = W_BEST3_MOVE;
        else
            mob.value = move_evaluate(&search->board, &mob, move_ordering_phase);
    }
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
*/

inline void move_ordering(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    const int weight_idx = HW2 - search->board.n <= 20 ? HW2 - search->board.n - MID_TO_END_DEPTH + 1 : 0;
    for (Mobility &mob: move_list)
        move_evaluate_simple(search, &mob, best_moves, weight_idx);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

/*
inline void move_ordering(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    const int depth = HW2 - search->board.n;
    const int weight_idx = depth <= 20 ? depth - MID_TO_END_DEPTH + 1 : 0;
    int move_ordering_phase = (search->board.n - 4) / MOVE_ORDERING_PHASE_DIV;
    for (Mobility &mob: move_list){
        move_evaluate_simple(search, &mob, best_moves, weight_idx);
        mob.value += move_evaluate(&search->board, &mob, move_ordering_phase) / 32;
    }
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
*/

inline void move_evaluate_fast_first(Search *search, Mobility *mob, const int best_moves[]){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else{
        //mob->value += cell_weight[mob->pos] * W_END_CELL_WEIGHT;
        if (search->board.parity & cell_div4[mob->pos])
            mob->value += W_END_PARITY;
        search->board.move(mob);
            //mob->value += -mid_evaluate(&search->board) * W_END_EVALUATE;
            if (search->board.p == BLACK)
                mob->value += calc_surround(search->board.b, ~(search->board.b | search->board.w)) * W_END_SURROUND;
            else
                mob->value += calc_surround(search->board.w, ~(search->board.b | search->board.w)) * W_END_SURROUND;
            /*
            int stab0, stab1;
            calc_stability_fast(&search->board, &stab0, &stab1);
            if (search->board.p == BLACK)
                mob->value += stab1 * W_END_STABILITY;
            else
                mob->value += stab0 * W_END_STABILITY;
            */
            mob->value -= pop_count_ull(search->board.mobility_ull()) * W_END_MOBILITY;
        search->board.undo(mob);
    }
}

inline void move_ordering_fast_first(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    for (Mobility &mob: move_list)
        move_evaluate_fast_first(search, &mob, best_moves);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first_fast(Search *search, Mobility *mob, const int best_moves[]){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else if (search->board.parity & cell_div4[mob->pos])
        mob->value += W_END_PARITY;
}

inline void move_ordering_fast_first_fast(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    if (!child_transpose_table.get_now(&search->board, hash_code, best_moves))
        child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    for (Mobility &mob: move_list)
        move_evaluate_fast_first_fast(search, &mob, best_moves);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_ordering_value(vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
