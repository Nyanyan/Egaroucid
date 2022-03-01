#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"

using namespace std;

#define N_PATTERNS 16
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STABILITY 65
#define MAX_STONE_NUM 65
#define N_CANPUT_PATTERNS 4
#define MAX_EVALUATE_IDX 59049

#define STEP 256
#define STEP_2 128

#define P31 3
#define P32 9
#define P33 27
#define P34 81
#define P35 243
#define P36 729
#define P37 2187
#define P38 6561
#define P39 19683
#define P310 59049
#define P31m 2
#define P32m 8
#define P33m 26
#define P34m 80
#define P35m 242
#define P36m 728
#define P37m 2186
#define P38m 6560
#define P39m 19682
#define P310m 59048

#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

uint_fast16_t pow3[11];
unsigned long long stability_edge_arr[N_8BIT][N_8BIT][2];
short pattern_arr[N_PHASES][2][N_PATTERNS][MAX_EVALUATE_IDX];
short eval_sur0_sur1_arr[N_PHASES][2][MAX_SURROUND][MAX_SURROUND];
short eval_canput0_canput1_arr[N_PHASES][2][MAX_CANPUT][MAX_CANPUT];
short eval_stab0_stab1_arr[N_PHASES][2][MAX_STABILITY][MAX_STABILITY];
short eval_num0_num1_arr[N_PHASES][2][MAX_STONE_NUM][MAX_STONE_NUM];
short eval_canput_pattern[N_PHASES][2][N_CANPUT_PATTERNS][P48];
short tmp_eval_canput_pattern[N_CANPUT_PATTERNS][P48];

string create_line(int b, int w){
    string res = "";
    for (int i = 0; i < HW; ++i){
        if ((b >> i) & 1)
            res += "X";
        else if ((w >> i) & 1)
            res += "O";
        else
            res += ".";
    }
    return res;
}

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i >= 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w, int ob, int ow){
    int i, nb, nw, res = 0b11111111;
    res &= b & ob;
    res &= w & ow;
    for (i = 0; i < HW; ++i){
        if ((1 & (b >> i)) == 0 && (1 & (w >> i)) == 0){
            probably_move_line(b, w, i, &nb, &nw);
            res &= calc_stability_line(nb, nw, ob, ow);
            probably_move_line(w, b, i, &nw, &nb);
            res &= calc_stability_line(nb, nw, ob, ow);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int idx, place, b, w, stab;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    /*
    for (place = 0; place < HW; ++place)
        stab[place] = true;
    b = create_one_color(712, 0);
    w = create_one_color(712, 1);
    cerr << create_line(b, w) << " ";
    calc_stability_line(b, w, stab);
    for (place = 0; place < HW; ++place)
        cerr << stab[place];
    cerr << endl;
    exit(0);
    */
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < HW; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * HW);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
        }
    }
}

inline int convert_canput_line(int idx){
    int res = 0;
    for (int i = 0; i < HW * 2; i += 2)
        res |= (1 & (idx >> i)) << (i / 2);
    for (int i = 1; i < HW * 2; i += 2)
        res |= (1 & (idx >> i)) << ((i / 2) + HW);
    return res;
}

inline bool init_evaluation_calc(){
    FILE* fp;
    #ifdef _WIN64
    if (fopen_s(&fp, "resources/eval.egev", "rb") != 0){
        cerr << "can't open eval.egev" << endl;
        return false;
    }
    #else
    fp = fopen("resources/eval.egev", "rb");
    if (fp == NULL){
        cerr << "can't open eval.egev" << endl;
        return false;
    }
    #endif
    int phase_idx, player_idx, pattern_idx;
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10};
    //constexpr int n_models = N_PHASES * 2;
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            //cerr << "loading evaluation parameter " << ((phase_idx * 2 + player_idx) * 100 / n_models) << "%" << endl;
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
                if (fread(pattern_arr[phase_idx][player_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                    cerr << "eval.egev broken" << endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(eval_sur0_sur1_arr[phase_idx][player_idx], 2, MAX_SURROUND * MAX_SURROUND, fp) < MAX_SURROUND * MAX_SURROUND){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_canput0_canput1_arr[phase_idx][player_idx], 2, MAX_CANPUT * MAX_CANPUT, fp) < MAX_CANPUT * MAX_CANPUT){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_stab0_stab1_arr[phase_idx][player_idx], 2, MAX_STABILITY * MAX_STABILITY, fp) < MAX_STABILITY * MAX_STABILITY){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_num0_num1_arr[phase_idx][player_idx], 2, MAX_STONE_NUM * MAX_STONE_NUM, fp) < MAX_STONE_NUM * MAX_STONE_NUM){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(tmp_eval_canput_pattern, 2, N_CANPUT_PATTERNS * P48, fp) < N_CANPUT_PATTERNS * P48){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            for (int i = 0; i < N_CANPUT_PATTERNS; ++i){
                for (int j = 0; j < P48; ++j)
                    eval_canput_pattern[phase_idx][player_idx][i][convert_canput_line(j)] = tmp_eval_canput_pattern[i][j];
            }
        }
    }
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(){
    init_evaluation_base();
    #if !EVAL_MODE
        return init_evaluation_calc();
    #else
        return true;
    #endif
    //cerr << "evaluation function initialized" << endl;
}

inline unsigned long long calc_surround_part(const unsigned long long player, const int dr){
    return (player << dr | player >> dr);
}

inline int calc_surround(const unsigned long long player, const unsigned long long empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline void calc_stability(Board *b, int *stab0, int *stab1){
    unsigned long long full_h, full_v, full_d7, full_d9;
    unsigned long long edge_stability = 0, black_stability = 0, white_stability = 0, n_stability;
    unsigned long long h, v, d7, d9;
    const unsigned long long black_mask = b->b & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const unsigned long long white_mask = b->w & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    int bk, wt;
    bk = b->b & 0b11111111;
    wt = b->w & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0] << 56;
    bk = (b->b >> 56) & 0b11111111;
    wt = (b->w >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0];
    bk = join_v_line(b->b, 0);
    wt = join_v_line(b->w, 0);
    edge_stability |= stability_edge_arr[bk][wt][1] << 7;
    bk = join_v_line(b->b, 7);
    wt = join_v_line(b->w, 7);
    edge_stability |= stability_edge_arr[bk][wt][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

    n_stability = (edge_stability & b->b) | (full_h & full_v & full_d7 & full_d9 & black_mask);
    while (n_stability & ~black_stability){
        black_stability |= n_stability;
        h = (black_stability >> 1) | (black_stability << 1) | full_h;
        v = (black_stability >> HW) | (black_stability << HW) | full_v;
        d7 = (black_stability >> HW_M1) | (black_stability << HW_M1) | full_d7;
        d9 = (black_stability >> HW_P1) | (black_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & black_mask;
    }

    n_stability = (edge_stability & b->w) | (full_h & full_v & full_d7 & full_d9 & white_mask);
    while (n_stability & ~white_stability){
        white_stability |= n_stability;
        h = (white_stability >> 1) | (white_stability << 1) | full_h;
        v = (white_stability >> HW) | (white_stability << HW) | full_v;
        d7 = (white_stability >> HW_M1) | (white_stability << HW_M1) | full_d7;
        d9 = (white_stability >> HW_P1) | (white_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & white_mask;
    }

    *stab0 = pop_count_ull(black_stability);
    *stab1 = pop_count_ull(white_stability);
}

inline void calc_stability_fast(Board *b, int *stab0, int *stab1){
    unsigned long long edge_stability = 0;
    int bk, wt;
    bk = b->b & 0b11111111;
    wt = b->w & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0] << 56;
    bk = (b->b >> 56) & 0b11111111;
    wt = (b->w >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0];
    bk = join_v_line(b->b, 0);
    wt = join_v_line(b->w, 0);
    edge_stability |= stability_edge_arr[bk][wt][1] << 7;
    bk = join_v_line(b->b, 7);
    wt = join_v_line(b->w, 7);
    edge_stability |= stability_edge_arr[bk][wt][1];
    *stab0 = pop_count_ull(edge_stability & b->b);
    *stab1 = pop_count_ull(edge_stability & b->w);
}


inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P34 + b_arr[p1] * P33 + b_arr[p2] * P32 + b_arr[P3] * P31 + b_arr[P4]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P35 + b_arr[p1] * P34 + b_arr[p2] * P33 + b_arr[P3] * P32 + b_arr[P4] * P31 + b_arr[p5]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P36 + b_arr[p1] * P35 + b_arr[p2] * P34 + b_arr[P3] * P33 + b_arr[P4] * P32 + b_arr[p5] * P31 + b_arr[p6]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P37 + b_arr[p1] * P36 + b_arr[p2] * P35 + b_arr[P3] * P34 + b_arr[P4] * P33 + b_arr[p5] * P32 + b_arr[p6] * P31 + b_arr[p7]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P38 + b_arr[p1] * P37 + b_arr[p2] * P36 + b_arr[P3] * P35 + b_arr[P4] * P34 + b_arr[p5] * P33 + b_arr[p6] * P32 + b_arr[p7] * P31 + b_arr[p8]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * P39 + b_arr[p1] * P38 + b_arr[p2] * P37 + b_arr[P3] * P36 + b_arr[P4] * P35 + b_arr[p5] * P34 + b_arr[p6] * P33 + b_arr[p7] * P32 + b_arr[p8] * P31 + b_arr[p9]];
}

inline int calc_pattern(const int phase_idx, Board *b){
    int b_arr[HW2];
    b->translate_to_arr(b_arr);
    return 
        pick_pattern(phase_idx, b->p, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, b->p, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, b->p, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, b->p, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, b->p, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, b->p, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, b->p, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, b->p, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, b->p, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, b->p, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, b->p, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, b->p, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, b->p, 3, b_arr, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, b->p, 3, b_arr, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, b->p, 3, b_arr, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, b->p, 3, b_arr, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, b->p, 4, b_arr, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, b->p, 4, b_arr, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, b->p, 4, b_arr, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, b->p, 4, b_arr, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, b->p, 5, b_arr, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, b->p, 5, b_arr, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, b->p, 5, b_arr, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, b->p, 5, b_arr, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, b->p, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, b->p, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, b->p, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, b->p, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7) + 
        pick_pattern(phase_idx, b->p, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24) + pick_pattern(phase_idx, b->p, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31) + pick_pattern(phase_idx, b->p, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39) + pick_pattern(phase_idx, b->p, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32) + 
        pick_pattern(phase_idx, b->p, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, b->p, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, b->p, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, b->p, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, b->p, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, b->p, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, b->p, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, b->p, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, b->p, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, b->p, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, b->p, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, b->p, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, b->p, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13) + pick_pattern(phase_idx, b->p, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41) + pick_pattern(phase_idx, b->p, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53) + pick_pattern(phase_idx, b->p, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22) + 
        pick_pattern(phase_idx, b->p, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, b->p, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, b->p, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, b->p, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24) + 
        pick_pattern(phase_idx, b->p, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27) + pick_pattern(phase_idx, b->p, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28) + pick_pattern(phase_idx, b->p, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35) + pick_pattern(phase_idx, b->p, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36) + 
        pick_pattern(phase_idx, b->p, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33) + pick_pattern(phase_idx, b->p, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38) + pick_pattern(phase_idx, b->p, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25) + pick_pattern(phase_idx, b->p, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
}
/*
inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P34 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P33 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P32 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P31 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4, const int p5){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P35 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P34 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P33 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P32 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2) * P31 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P36 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P35 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P34 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P33 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2) * P32 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * P31 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P37 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P36 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P35 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P34 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2) * P33 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * P32 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * P31 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P38 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P37 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P36 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P35 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2) * P34 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * P33 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * P32 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2) * P31 + 
        (2 - pop_digit(wt, p8) - pop_digit(bk, p8) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * P39 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * P38 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * P37 + 
        (2 - pop_digit(wt, P3) - pop_digit(bk, P3) * 2) * P36 + 
        (2 - pop_digit(wt, P4) - pop_digit(bk, P4) * 2) * P35 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * P34 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * P33 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2) * P32 + 
        (2 - pop_digit(wt, p8) - pop_digit(bk, p8) * 2) * P31 + 
        (2 - pop_digit(wt, p9) - pop_digit(bk, p9) * 2)
        ];
}

inline int calc_pattern(const int phase_idx, Board *b){
    return 
        pick_pattern(phase_idx, b->p, 0, b->b, b->w, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, b->p, 0, b->b, b->w, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, b->p, 0, b->b, b->w, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, b->p, 0, b->b, b->w, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, b->p, 1, b->b, b->w, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, b->p, 1, b->b, b->w, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, b->p, 1, b->b, b->w, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, b->p, 1, b->b, b->w, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, b->p, 2, b->b, b->w, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, b->p, 2, b->b, b->w, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, b->p, 2, b->b, b->w, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, b->p, 2, b->b, b->w, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, b->p, 3, b->b, b->w, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, b->p, 3, b->b, b->w, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, b->p, 3, b->b, b->w, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, b->p, 3, b->b, b->w, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, b->p, 4, b->b, b->w, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, b->p, 4, b->b, b->w, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, b->p, 4, b->b, b->w, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, b->p, 4, b->b, b->w, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, b->p, 5, b->b, b->w, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, b->p, 5, b->b, b->w, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, b->p, 5, b->b, b->w, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, b->p, 5, b->b, b->w, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, b->p, 6, b->b, b->w, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, b->p, 6, b->b, b->w, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, b->p, 7, b->b, b->w, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, b->p, 7, b->b, b->w, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, b->p, 7, b->b, b->w, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, b->p, 7, b->b, b->w, 54, 63, 55, 47, 39, 31, 23, 15, 7) + 
        pick_pattern(phase_idx, b->p, 8, b->b, b->w, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24) + pick_pattern(phase_idx, b->p, 8, b->b, b->w, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31) + pick_pattern(phase_idx, b->p, 8, b->b, b->w, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39) + pick_pattern(phase_idx, b->p, 8, b->b, b->w, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32) + 
        pick_pattern(phase_idx, b->p, 9, b->b, b->w, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, b->p, 9, b->b, b->w, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, b->p, 9, b->b, b->w, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, b->p, 9, b->b, b->w, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, b->p, 10, b->b, b->w, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, b->p, 10, b->b, b->w, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, b->p, 10, b->b, b->w, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, b->p, 10, b->b, b->w, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, b->p, 11, b->b, b->w, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, b->p, 11, b->b, b->w, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, b->p, 11, b->b, b->w, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, b->p, 11, b->b, b->w, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, b->p, 12, b->b, b->w, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13) + pick_pattern(phase_idx, b->p, 12, b->b, b->w, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41) + pick_pattern(phase_idx, b->p, 12, b->b, b->w, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53) + pick_pattern(phase_idx, b->p, 12, b->b, b->w, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22) + 
        pick_pattern(phase_idx, b->p, 13, b->b, b->w, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, b->p, 13, b->b, b->w, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, b->p, 13, b->b, b->w, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, b->p, 13, b->b, b->w, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24) + 
        pick_pattern(phase_idx, b->p, 14, b->b, b->w, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27) + pick_pattern(phase_idx, b->p, 14, b->b, b->w, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28) + pick_pattern(phase_idx, b->p, 14, b->b, b->w, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35) + pick_pattern(phase_idx, b->p, 14, b->b, b->w, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36) + 
        pick_pattern(phase_idx, b->p, 15, b->b, b->w, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33) + pick_pattern(phase_idx, b->p, 15, b->b, b->w, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38) + pick_pattern(phase_idx, b->p, 15, b->b, b->w, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25) + pick_pattern(phase_idx, b->p, 15, b->b, b->w, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
}
*/

inline int create_canput_line_h(unsigned long long b, unsigned long long w, int t){
    return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v(unsigned long long b, unsigned long long w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}

inline int calc_canput_pattern(const int phase_idx, Board *b, const unsigned long long black_mobility, const unsigned long long white_mobility){
    return 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line_h(black_mobility, white_mobility, 0)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line_h(black_mobility, white_mobility, 7)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line_v(black_mobility, white_mobility, 0)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line_v(black_mobility, white_mobility, 7)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line_h(black_mobility, white_mobility, 1)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line_h(black_mobility, white_mobility, 6)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line_v(black_mobility, white_mobility, 1)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line_v(black_mobility, white_mobility, 6)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line_h(black_mobility, white_mobility, 2)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line_h(black_mobility, white_mobility, 5)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line_v(black_mobility, white_mobility, 2)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line_v(black_mobility, white_mobility, 5)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line_h(black_mobility, white_mobility, 3)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line_h(black_mobility, white_mobility, 4)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line_v(black_mobility, white_mobility, 3)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line_v(black_mobility, white_mobility, 4)];
}

inline int end_evaluate(Board *b){
    return b->score();
}

inline int mid_evaluate(Board *b){
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    unsigned long long black_mobility, white_mobility, empties;
    black_mobility = get_mobility(b->b, b->w);
    white_mobility = get_mobility(b->w, b->b);
    empties = ~(b->b | b->w);
    canput0 = min(MAX_CANPUT - 1, pop_count_ull(black_mobility));
    canput1 = min(MAX_CANPUT - 1, pop_count_ull(white_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(b);
    phase_idx = b->phase();
    sur0 = min(MAX_SURROUND - 1, calc_surround(b->b, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround(b->w, empties));
    calc_stability(b, &stab0, &stab1);
    num0 = pop_count_ull(b->b);
    num1 = pop_count_ull(b->w);
    //cerr << sur0 << " " << sur1 << " " << canput0 << " " << canput1 << " " << stab0 << " " << stab1 << " " << num0 << " " << num1 << endl;
    int res = (b->p ? -1 : 1) * (
        calc_pattern(phase_idx, b) + 
        eval_sur0_sur1_arr[phase_idx][b->p][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][b->p][canput0][canput1] + 
        eval_stab0_stab1_arr[phase_idx][b->p][stab0][stab1] + 
        eval_num0_num1_arr[phase_idx][b->p][num0][num1] + 
        calc_canput_pattern(phase_idx, b, black_mobility, white_mobility)
        );
    if (res > 0)
        res += STEP_2;
    else if (res < 0)
        res -= STEP_2;
    res /= STEP;
    return max(-HW2, min(HW2, res));
}