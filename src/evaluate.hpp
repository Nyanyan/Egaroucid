#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"

using namespace std;

#define n_patterns 16
#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65
#define n_canput_patterns 4
#define max_evaluate_idx 59049

#define step 256
#define step_2 128

#define p31 3
#define p32 9
#define p33 27
#define p34 81
#define p35 243
#define p36 729
#define p37 2187
#define p38 6561
#define p39 19683
#define p310 59049
#define p31m 2
#define p32m 8
#define p33m 26
#define p34m 80
#define p35m 242
#define p36m 728
#define p37m 2186
#define p38m 6560
#define p39m 19682
#define p310m 59048

#define p41 4
#define p42 16
#define p43 64
#define p44 256
#define p45 1024
#define p46 4096
#define p47 16384
#define p48 65536

uint_fast16_t pow3[11];
unsigned long long stability_edge_arr[n_8bit][n_8bit][2];
short pattern_arr[n_phases][2][n_patterns][max_evaluate_idx];
short eval_sur0_sur1_arr[n_phases][2][max_surround][max_surround];
short eval_canput0_canput1_arr[n_phases][2][max_canput][max_canput];
short eval_stab0_stab1_arr[n_phases][2][max_stability][max_stability];
short eval_num0_num1_arr[n_phases][2][max_stone_num][max_stone_num];
short eval_canput_pattern[n_phases][2][n_canput_patterns][p48];

string create_line(int b, int w){
    string res = "";
    for (int i = 0; i < hw; ++i){
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
    for (i = place + 1; i < hw && (1 & (o >> i)); ++i);
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
    for (i = 0; i < hw; ++i){
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
    for (place = 0; place < hw; ++place)
        stab[place] = true;
    b = create_one_color(712, 0);
    w = create_one_color(712, 1);
    cerr << create_line(b, w) << " ";
    calc_stability_line(b, w, stab);
    for (place = 0; place < hw; ++place)
        cerr << stab[place];
    cerr << endl;
    exit(0);
    */
    for (b = 0; b < n_8bit; ++b) {
        for (w = b; w < n_8bit; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < hw; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * hw);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
            /*
            cerr << create_line(b, w) << " ";
            for (place = 0; place < hw; ++place)
                cerr << stab[place];
            cerr << " ";
            for (place = 0; place < hw; ++place)
                cerr << (1 & (stability_edge_arr[black][idx][0] >> place));
            cerr << " ";
            for (place = 0; place < hw; ++place)
                cerr << (1 & (stability_edge_arr[white][idx][0] >> place));
            cerr << endl;
            */
        }
    }
}

/*
inline bool init_evaluation_calc(){
    ifstream ifs("resources/param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        return false;
    }
    string line;
    int phase_idx, player_idx, pattern_idx, pattern_elem, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    constexpr int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10};
    constexpr int n_models = n_phases * 2;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            cerr << "loading " << ((phase_idx * 2 + player_idx) * 100 / n_models) << "%" << endl;
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                for (pattern_elem = 0; pattern_elem < pow3[pattern_sizes[pattern_idx]]; ++pattern_elem){
                    getline(ifs, line);
                    pattern_arr[phase_idx][player_idx][pattern_idx][pattern_elem] = stoi(line);
                }
            }
            for (sur0 = 0; sur0 < max_surround; ++sur0){
                for (sur1 = 0; sur1 < max_surround; ++sur1){
                    getline(ifs, line);
                    eval_sur0_sur1_arr[phase_idx][player_idx][sur0][sur1] = stoi(line);
                }
            }
            for (canput0 = 0; canput0 < max_canput; ++canput0){
                for (canput1 = 0; canput1 < max_canput; ++canput1){
                    getline(ifs, line);
                    eval_canput0_canput1_arr[phase_idx][player_idx][canput0][canput1] = stoi(line);
                }
            }
            for (stab0 = 0; stab0 < max_stability; ++stab0){
                for (stab1 = 0; stab1 < max_stability; ++stab1){
                    getline(ifs, line);
                    eval_stab0_stab1_arr[phase_idx][player_idx][stab0][stab1] = stoi(line);
                }
            }
            for (num0 = 0; num0 < max_stone_num; ++num0){
                for (num1 = 0; num1 < max_stone_num; ++num1){
                    getline(ifs, line);
                    eval_num0_num1_arr[phase_idx][player_idx][num0][num1] = stoi(line);
                }
            }
            for (pattern_idx = 0; pattern_idx < n_canput_patterns; ++pattern_idx){
                for (pattern_elem = 0; pattern_elem < p48; ++pattern_elem){
                    getline(ifs, line);
                    eval_canput_pattern[phase_idx][player_idx][pattern_idx][pattern_elem] = stoi(line);
                }
            }
        }
    }
    cerr << "evaluation function initialized" << endl;
    return true;
}
*/

inline int convert_canput_line(int idx){
    int res = 0;
    for (int i = 0; i < hw * 2; i += 2)
        res |= (1 & (idx >> i)) << (i / 2);
    for (int i = 1; i < hw * 2; i += 2)
        res |= (1 & (idx >> i)) << ((i / 2) + hw);
    return res;
}

inline bool init_evaluation_calc(){
    FILE* fp;
    if (fopen_s(&fp, "resources/eval.egev", "rb") != 0){
        cerr << "can't open eval.egev" << endl;
        return false;
    }
    int phase_idx, player_idx, pattern_idx;
    constexpr int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10};
    //constexpr int n_models = n_phases * 2;
    short tmp_eval_canput_pattern[n_canput_patterns][p48];
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            //cerr << "loading evaluation parameter " << ((phase_idx * 2 + player_idx) * 100 / n_models) << "%" << endl;
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                if (fread(pattern_arr[phase_idx][player_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                    cerr << "eval.egev broken" << endl;
                    fclose(fp);
                    return false;
                }
            }
            if (fread(eval_sur0_sur1_arr[phase_idx][player_idx], 2, max_surround * max_surround, fp) < max_surround * max_surround){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_canput0_canput1_arr[phase_idx][player_idx], 2, max_canput * max_canput, fp) < max_canput * max_canput){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_stab0_stab1_arr[phase_idx][player_idx], 2, max_stability * max_stability, fp) < max_stability * max_stability){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(eval_num0_num1_arr[phase_idx][player_idx], 2, max_stone_num * max_stone_num, fp) < max_stone_num * max_stone_num){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            if (fread(tmp_eval_canput_pattern, 2, n_canput_patterns * p48, fp) < n_canput_patterns * p48){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
            for (int i = 0; i < n_canput_patterns; ++i){
                for (int j = 0; j < p48; ++j)
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
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, hw) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, hw_m1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, hw_p1)
    ));
}

/*
inline int join_pattern(const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return b_arr[p0] * p37 + b_arr[p1] * p36 + b_arr[p2] * p35 + b_arr[p3] * p34 + b_arr[p4] * p33 + b_arr[p5] * p32 + b_arr[p6] * p31 + b_arr[p7];
}
*/

inline void calc_stability(board *b, int *stab0, int *stab1){
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
    
    //black_stability = all_stability & b->b;
    //white_stability = all_stability & b->w;
    
    n_stability = (edge_stability & b->b) | (full_h & full_v & full_d7 & full_d9 & black_mask);
    while (n_stability & ~black_stability){
        black_stability |= n_stability;
        h = (black_stability >> 1) | (black_stability << 1) | full_h;
        v = (black_stability >> hw) | (black_stability << hw) | full_v;
        d7 = (black_stability >> hw_m1) | (black_stability << hw_m1) | full_d7;
        d9 = (black_stability >> hw_p1) | (black_stability << hw_p1) | full_d9;
        n_stability = h & v & d7 & d9 & black_mask;
    }

    n_stability = (edge_stability & b->w) | (full_h & full_v & full_d7 & full_d9 & white_mask);
    while (n_stability & ~white_stability){
        white_stability |= n_stability;
        h = (white_stability >> 1) | (white_stability << 1) | full_h;
        v = (white_stability >> hw) | (white_stability << hw) | full_v;
        d7 = (white_stability >> hw_m1) | (white_stability << hw_m1) | full_d7;
        d9 = (white_stability >> hw_p1) | (white_stability << hw_p1) | full_d9;
        n_stability = h & v & d7 & d9 & white_mask;
    }

    /*
    for (int i = hw2_m1; i >= 0; --i){
        if (1 & (b->b >> i))
            cerr << '0';
        else if (1 & (b->w >> i))
            cerr << '1';
        else
            cerr << '.';
        if (i % hw == 0)
            cerr << endl;
    }
    cerr << endl;
    for (int i = hw2_m1; i >= 0; --i){
        if (1 & (black_stability >> i))
            cerr << '0';
        else
            cerr << '.';
        if (i % hw == 0)
            cerr << endl;
    }
    cerr << endl;
    for (int i = hw2_m1; i >= 0; --i){
        if (1 & (white_stability >> i))
            cerr << '1';
        else
            cerr << '.';
        if (i % hw == 0)
            cerr << endl;
    }
    cerr << endl;
    */
    *stab0 = pop_count_ull(black_stability);
    *stab1 = pop_count_ull(white_stability);
}

inline void calc_stability_fast(board *b, int *stab0, int *stab1){
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

inline int pop_digit(unsigned long long x, int place){
    return 1 & (x >> place);
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p34 + b_arr[p1] * p33 + b_arr[p2] * p32 + b_arr[p3] * p31 + b_arr[p4]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p35 + b_arr[p1] * p34 + b_arr[p2] * p33 + b_arr[p3] * p32 + b_arr[p4] * p31 + b_arr[p5]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p36 + b_arr[p1] * p35 + b_arr[p2] * p34 + b_arr[p3] * p33 + b_arr[p4] * p32 + b_arr[p5] * p31 + b_arr[p6]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p37 + b_arr[p1] * p36 + b_arr[p2] * p35 + b_arr[p3] * p34 + b_arr[p4] * p33 + b_arr[p5] * p32 + b_arr[p6] * p31 + b_arr[p7]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p38 + b_arr[p1] * p37 + b_arr[p2] * p36 + b_arr[p3] * p35 + b_arr[p4] * p34 + b_arr[p5] * p33 + b_arr[p6] * p32 + b_arr[p7] * p31 + b_arr[p8]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][p][pattern_idx][b_arr[p0] * p39 + b_arr[p1] * p38 + b_arr[p2] * p37 + b_arr[p3] * p36 + b_arr[p4] * p35 + b_arr[p5] * p34 + b_arr[p6] * p33 + b_arr[p7] * p32 + b_arr[p8] * p31 + b_arr[p9]];
}

inline int calc_pattern(const int phase_idx, board *b, const int b_arr[]){
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
inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p34 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p33 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p32 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p31 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p35 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p34 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p33 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p32 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2) * p31 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p36 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p35 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p34 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p33 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2) * p32 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * p31 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p37 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p36 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p35 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p34 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2) * p33 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * p32 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * p31 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p38 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p37 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p36 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p35 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2) * p34 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * p33 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * p32 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2) * p31 + 
        (2 - pop_digit(wt, p8) - pop_digit(bk, p8) * 2)
        ];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const unsigned long long bk, const unsigned long long wt, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][p][pattern_idx][
        (2 - pop_digit(wt, p0) - pop_digit(bk, p0) * 2) * p39 + 
        (2 - pop_digit(wt, p1) - pop_digit(bk, p1) * 2) * p38 + 
        (2 - pop_digit(wt, p2) - pop_digit(bk, p2) * 2) * p37 + 
        (2 - pop_digit(wt, p3) - pop_digit(bk, p3) * 2) * p36 + 
        (2 - pop_digit(wt, p4) - pop_digit(bk, p4) * 2) * p35 + 
        (2 - pop_digit(wt, p5) - pop_digit(bk, p5) * 2) * p34 + 
        (2 - pop_digit(wt, p6) - pop_digit(bk, p6) * 2) * p33 + 
        (2 - pop_digit(wt, p7) - pop_digit(bk, p7) * 2) * p32 + 
        (2 - pop_digit(wt, p8) - pop_digit(bk, p8) * 2) * p31 + 
        (2 - pop_digit(wt, p9) - pop_digit(bk, p9) * 2)
        ];
}

inline int calc_pattern(const int phase_idx, board *b){
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
/*
inline int create_canput_line(const int canput_arr[], const int a, const int b, const int c, const int d, const int e, const int f, const int g, const int h){
    return 
        canput_arr[a] * p47 + canput_arr[b] * p46 + canput_arr[c] * p45 + canput_arr[d] * p44 + 
        canput_arr[e] * p43 + canput_arr[f] * p42 + canput_arr[g] * p41 + canput_arr[h];
}
*/
/*
inline int create_canput_line(unsigned long long bk, unsigned long long wt, const int a, const int b, const int c, const int d, const int e, const int f, const int g, const int h){
    return 
        (pop_digit(wt, a) * 2 + pop_digit(bk, a)) * p47 + 
        (pop_digit(wt, b) * 2 + pop_digit(bk, b)) * p46 + 
        (pop_digit(wt, c) * 2 + pop_digit(bk, c)) * p45 + 
        (pop_digit(wt, d) * 2 + pop_digit(bk, d)) * p44 + 
        (pop_digit(wt, e) * 2 + pop_digit(bk, e)) * p43 + 
        (pop_digit(wt, f) * 2 + pop_digit(bk, f)) * p42 + 
        (pop_digit(wt, g) * 2 + pop_digit(bk, g)) * p41 + 
        (pop_digit(wt, h) * 2 + pop_digit(bk, h));
}
*/
inline int create_canput_line_h(unsigned long long b, unsigned long long w, int t){
    return (((w >> (hw * t)) & 0b11111111) << hw) | ((b >> (hw * t)) & 0b11111111);
}

inline int create_canput_line_v(unsigned long long b, unsigned long long w, int t){
    return (join_v_line(w, t) << hw) | join_v_line(b, t);
}

inline int calc_canput_pattern(const int phase_idx, board *b, const unsigned long long black_mobility, const unsigned long long white_mobility){
    /*
    int canput_arr[hw2];
    b->board_canput(canput_arr, black_mobility, white_mobility);
    return
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(canput_arr, 0, 1, 2, 3, 4, 5, 6, 7)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(canput_arr, 0, 8, 16, 24, 32, 40, 48, 56)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(canput_arr, 7, 15, 23, 31, 39, 47, 55, 63)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(canput_arr, 56, 57, 58, 59, 60, 61, 62, 63)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(canput_arr, 8, 9, 10, 11, 12, 13, 14, 15)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(canput_arr, 1, 9, 17, 25, 33, 41, 49, 57)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(canput_arr, 6, 14, 22, 30, 38, 46, 54, 62)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(canput_arr, 48, 49, 50, 51, 52, 53, 54, 55)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(canput_arr, 16, 17, 18, 19, 20, 21, 22, 23)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(canput_arr, 2, 10, 18, 26, 34, 42, 50, 58)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(canput_arr, 5, 13, 21, 29, 37, 45, 53, 61)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(canput_arr, 40, 41, 42, 43, 44, 45, 46, 47)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(canput_arr, 24, 25, 26, 27, 28, 29, 30, 31)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(canput_arr, 3, 11, 19, 27, 35, 43, 51, 59)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(canput_arr, 4, 12, 20, 28, 36, 44, 52, 60)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(canput_arr, 32, 33, 34, 35, 36, 37, 38, 39)];
    */
    /*
    return
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(black_mobility, white_mobility, 0, 1, 2, 3, 4, 5, 6, 7)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(black_mobility, white_mobility, 0, 8, 16, 24, 32, 40, 48, 56)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(black_mobility, white_mobility, 7, 15, 23, 31, 39, 47, 55, 63)] + 
        eval_canput_pattern[phase_idx][b->p][0][create_canput_line(black_mobility, white_mobility, 56, 57, 58, 59, 60, 61, 62, 63)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(black_mobility, white_mobility, 8, 9, 10, 11, 12, 13, 14, 15)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(black_mobility, white_mobility, 1, 9, 17, 25, 33, 41, 49, 57)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(black_mobility, white_mobility, 6, 14, 22, 30, 38, 46, 54, 62)] + 
        eval_canput_pattern[phase_idx][b->p][1][create_canput_line(black_mobility, white_mobility, 48, 49, 50, 51, 52, 53, 54, 55)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(black_mobility, white_mobility, 16, 17, 18, 19, 20, 21, 22, 23)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(black_mobility, white_mobility, 2, 10, 18, 26, 34, 42, 50, 58)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(black_mobility, white_mobility, 5, 13, 21, 29, 37, 45, 53, 61)] + 
        eval_canput_pattern[phase_idx][b->p][2][create_canput_line(black_mobility, white_mobility, 40, 41, 42, 43, 44, 45, 46, 47)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(black_mobility, white_mobility, 24, 25, 26, 27, 28, 29, 30, 31)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(black_mobility, white_mobility, 3, 11, 19, 27, 35, 43, 51, 59)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(black_mobility, white_mobility, 4, 12, 20, 28, 36, 44, 52, 60)] + 
        eval_canput_pattern[phase_idx][b->p][3][create_canput_line(black_mobility, white_mobility, 32, 33, 34, 35, 36, 37, 38, 39)];
    */
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

inline int end_evaluate(board *b){
    return b->count();
}

inline int mid_evaluate(board *b){
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    unsigned long long black_mobility, white_mobility, empties;
    int b_arr[hw2];
    b->translate_to_arr(b_arr);
    black_mobility = get_mobility(b->b, b->w);
    white_mobility = get_mobility(b->w, b->b);
    empties = ~(b->b | b->w);
    canput0 = min(max_canput - 1, pop_count_ull(black_mobility));
    canput1 = min(max_canput - 1, pop_count_ull(white_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(b);
    phase_idx = b->phase();
    sur0 = min(max_surround - 1, calc_surround(b->b, empties));
    sur1 = min(max_surround - 1, calc_surround(b->w, empties));
    calc_stability(b, &stab0, &stab1);
    num0 = pop_count_ull(b->b);
    num1 = pop_count_ull(b->w);
    //cerr << sur0 << " " << sur1 << " " << canput0 << " " << canput1 << " " << stab0 << " " << stab1 << " " << num0 << " " << num1 << endl;
    int res = (b->p ? -1 : 1) * (
        calc_pattern(phase_idx, b, b_arr) + 
        eval_sur0_sur1_arr[phase_idx][b->p][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][b->p][canput0][canput1] + 
        eval_stab0_stab1_arr[phase_idx][b->p][stab0][stab1] + 
        eval_num0_num1_arr[phase_idx][b->p][num0][num1] + 
        calc_canput_pattern(phase_idx, b, black_mobility, white_mobility)
        );
    if (res > 0)
        res += step_2;
    else if (res < 0)
        res -= step_2;
    res /= step;
    return max(-hw2, min(hw2, res));
}