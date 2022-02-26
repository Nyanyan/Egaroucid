#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "board.hpp"

using namespace std;

#define black 0
#define white 1

#define n_phases 15
#define phase_n_stones 4

#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65

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

unsigned long long stability_edge_arr[n_line][2];
int pop_digit[n_line][hw];
int pow3[11];

inline int calc_phase_idx(const board *b){
    return (b->n - 4) / phase_n_stones;
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
    int idx, place, b, w, stab, i, j;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            pop_digit[i][j] = (i / pow3[hw_m1 - j]) % 3;
    }
    for (idx = 0; idx < n_line; ++idx) {
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        stab = calc_stability_line(b, w, b, w);
        stability_edge_arr[idx][0] = 0;
        stability_edge_arr[idx][1] = 0;
        for (place = 0; place < hw; ++place){
            if (1 & (stab >> place)){
                stability_edge_arr[idx][0] |= 1ULL << place;
                stability_edge_arr[idx][1] |= 1ULL << (place * hw);
            }
        }
    }
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

inline int join_pattern(const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return b_arr[p0] * p37 + b_arr[p1] * p36 + b_arr[p2] * p35 + b_arr[p3] * p34 + b_arr[p4] * p33 + b_arr[p5] * p32 + b_arr[p6] * p31 + b_arr[p7];
}

inline void calc_stability(board *b, const int b_arr[], int *stab0, int *stab1){
    unsigned long long full_h, full_v, full_d7, full_d9;
    unsigned long long edge_stability = 0, black_stability = 0, white_stability = 0, n_stability;
    unsigned long long h, v, d7, d9;
    const unsigned long long black_mask = b->b & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const unsigned long long white_mask = b->w & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    int edge;
    edge = join_pattern(b_arr, 0, 1, 2, 3, 4, 5, 6, 7);
    edge_stability |= stability_edge_arr[edge][0] << 56;
    edge = join_pattern(b_arr, 56, 57, 58, 59, 60, 61, 62, 63);
    edge_stability |= stability_edge_arr[edge][0];
    edge = join_pattern(b_arr, 0, 8, 16, 24, 32, 40, 48, 56);
    edge_stability |= stability_edge_arr[edge][1] << 7;
    edge = join_pattern(b_arr, 7, 15, 23, 31, 39, 47, 55, 63);
    edge_stability |= stability_edge_arr[edge][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);
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
    *stab0 = pop_count_ull(black_stability);
    *stab1 = pop_count_ull(white_stability);
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4){
    return b_arr[p0] * p34 + b_arr[p1] * p33 + b_arr[p2] * p32 + b_arr[p3] * p31 + b_arr[p4];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return b_arr[p0] * p35 + b_arr[p1] * p34 + b_arr[p2] * p33 + b_arr[p3] * p32 + b_arr[p4] * p31 + b_arr[p5];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return b_arr[p0] * p36 + b_arr[p1] * p35 + b_arr[p2] * p34 + b_arr[p3] * p33 + b_arr[p4] * p32 + b_arr[p5] * p31 + b_arr[p6];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return b_arr[p0] * p37 + b_arr[p1] * p36 + b_arr[p2] * p35 + b_arr[p3] * p34 + b_arr[p4] * p33 + b_arr[p5] * p32 + b_arr[p6] * p31 + b_arr[p7];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return b_arr[p0] * p38 + b_arr[p1] * p37 + b_arr[p2] * p36 + b_arr[p3] * p35 + b_arr[p4] * p34 + b_arr[p5] * p33 + b_arr[p6] * p32 + b_arr[p7] * p31 + b_arr[p8];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return b_arr[p0] * p39 + b_arr[p1] * p38 + b_arr[p2] * p37 + b_arr[p3] * p36 + b_arr[p4] * p35 + b_arr[p5] * p34 + b_arr[p6] * p33 + b_arr[p7] * p32 + b_arr[p8] * p31 + b_arr[p9];
}

inline int create_canput_line(const int canput_arr[], const int a, const int b, const int c, const int d, const int e, const int f, const int g, const int h){
    return 
        canput_arr[a] * p47 + canput_arr[b] * p46 + canput_arr[c] * p45 + canput_arr[d] * p44 + 
        canput_arr[e] * p43 + canput_arr[f] * p42 + canput_arr[g] * p41 + canput_arr[h];
}


inline void calc_idx(int phase_idx, board *b, int idxes[]){
    int b_arr[hw2];
    b->translate_to_arr(b_arr);
    idxes[0] = pick_pattern(phase_idx, b->p, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15);
    idxes[1] = pick_pattern(phase_idx, b->p, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57);
    idxes[2] = pick_pattern(phase_idx, b->p, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55);
    idxes[3] = pick_pattern(phase_idx, b->p, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62);
    idxes[4] = pick_pattern(phase_idx, b->p, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23);
    idxes[5] = pick_pattern(phase_idx, b->p, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58);
    idxes[6] = pick_pattern(phase_idx, b->p, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47);
    idxes[7] = pick_pattern(phase_idx, b->p, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61);
    idxes[8] = pick_pattern(phase_idx, b->p, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31);
    idxes[9] = pick_pattern(phase_idx, b->p, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59);
    idxes[10] = pick_pattern(phase_idx, b->p, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39);
    idxes[11] = pick_pattern(phase_idx, b->p, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60);
    idxes[12] = pick_pattern(phase_idx, b->p, 3, b_arr, 3, 12, 21, 30, 39);
    idxes[13] = pick_pattern(phase_idx, b->p, 3, b_arr, 4, 11, 18, 25, 32);
    idxes[14] = pick_pattern(phase_idx, b->p, 3, b_arr, 24, 33, 42, 51, 60);
    idxes[15] = pick_pattern(phase_idx, b->p, 3, b_arr, 59, 52, 45, 38, 31);
    idxes[16] = pick_pattern(phase_idx, b->p, 4, b_arr, 2, 11, 20, 29, 38, 47);
    idxes[17] = pick_pattern(phase_idx, b->p, 4, b_arr, 5, 12, 19, 26, 33, 40);
    idxes[18] = pick_pattern(phase_idx, b->p, 4, b_arr, 16, 25, 34, 43, 52, 61);
    idxes[19] = pick_pattern(phase_idx, b->p, 4, b_arr, 58, 51, 44, 37, 30, 23);
    idxes[20] = pick_pattern(phase_idx, b->p, 5, b_arr, 1, 10, 19, 28, 37, 46, 55);
    idxes[21] = pick_pattern(phase_idx, b->p, 5, b_arr, 6, 13, 20, 27, 34, 41, 48);
    idxes[22] = pick_pattern(phase_idx, b->p, 5, b_arr, 8, 17, 26, 35, 44, 53, 62);
    idxes[23] = pick_pattern(phase_idx, b->p, 5, b_arr, 57, 50, 43, 36, 29, 22, 15);
    idxes[24] = pick_pattern(phase_idx, b->p, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63);
    idxes[25] = pick_pattern(phase_idx, b->p, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56);
    idxes[26] = pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14);
    idxes[27] = pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49);
    idxes[28] = pick_pattern(phase_idx, b->p, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54);
    idxes[29] = pick_pattern(phase_idx, b->p, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7);
    idxes[30] = pick_pattern(phase_idx, b->p, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24);
    idxes[31] = pick_pattern(phase_idx, b->p, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31);
    idxes[32] = pick_pattern(phase_idx, b->p, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39);
    idxes[33] = pick_pattern(phase_idx, b->p, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32);
    idxes[34] = pick_pattern(phase_idx, b->p, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13);
    idxes[35] = pick_pattern(phase_idx, b->p, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41);
    idxes[36] = pick_pattern(phase_idx, b->p, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53);
    idxes[37] = pick_pattern(phase_idx, b->p, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46);
    idxes[38] = pick_pattern(phase_idx, b->p, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26);
    idxes[39] = pick_pattern(phase_idx, b->p, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29);
    idxes[40] = pick_pattern(phase_idx, b->p, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34);
    idxes[41] = pick_pattern(phase_idx, b->p, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37);
    idxes[42] = pick_pattern(phase_idx, b->p, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18);
    idxes[43] = pick_pattern(phase_idx, b->p, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21);
    idxes[44] = pick_pattern(phase_idx, b->p, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42);
    idxes[45] = pick_pattern(phase_idx, b->p, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45);
    idxes[46] = pick_pattern(phase_idx, b->p, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13);
    idxes[47] = pick_pattern(phase_idx, b->p, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41);
    idxes[48] = pick_pattern(phase_idx, b->p, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53);
    idxes[49] = pick_pattern(phase_idx, b->p, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22);
    idxes[50] = pick_pattern(phase_idx, b->p, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32);
    idxes[51] = pick_pattern(phase_idx, b->p, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39);
    idxes[52] = pick_pattern(phase_idx, b->p, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31);
    idxes[53] = pick_pattern(phase_idx, b->p, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24);
    idxes[54] = pick_pattern(phase_idx, b->p, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27);
    idxes[55] = pick_pattern(phase_idx, b->p, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28);
    idxes[56] = pick_pattern(phase_idx, b->p, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35);
    idxes[57] = pick_pattern(phase_idx, b->p, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36);
    idxes[58] = pick_pattern(phase_idx, b->p, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33);
    idxes[59] = pick_pattern(phase_idx, b->p, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38);
    idxes[60] = pick_pattern(phase_idx, b->p, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25);
    idxes[61] = pick_pattern(phase_idx, b->p, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
    idxes[62] = min(max_surround - 1, calc_surround(b->b, ~(b->b | b->w)));
    idxes[63] = min(max_surround - 1, calc_surround(b->w, ~(b->b | b->w)));
    unsigned long long black_mobility = get_mobility(b->b, b->w);
    unsigned long long white_mobility = get_mobility(b->w, b->b);
    idxes[64] = pop_count_ull(black_mobility);
    idxes[65] = pop_count_ull(white_mobility);
    calc_stability(b, b_arr, &idxes[66], &idxes[67]);
    idxes[68] = pop_count_ull(b->b);
    idxes[69] = pop_count_ull(b->w);
    int canput_arr[hw2];
    b->board_canput(canput_arr, black_mobility, white_mobility);
    idxes[70] = create_canput_line(canput_arr, 0, 1, 2, 3, 4, 5, 6, 7);
    idxes[71] = create_canput_line(canput_arr, 0, 8, 16, 24, 32, 40, 48, 56);
    idxes[72] = create_canput_line(canput_arr, 7, 15, 23, 31, 39, 47, 55, 63);
    idxes[73] = create_canput_line(canput_arr, 56, 57, 58, 59, 60, 61, 62, 63);
    idxes[74] = create_canput_line(canput_arr, 8, 9, 10, 11, 12, 13, 14, 15);
    idxes[75] = create_canput_line(canput_arr, 1, 9, 17, 25, 33, 41, 49, 57);
    idxes[76] = create_canput_line(canput_arr, 6, 14, 22, 30, 38, 46, 54, 62);
    idxes[77] = create_canput_line(canput_arr, 48, 49, 50, 51, 52, 53, 54, 55);
    idxes[78] = create_canput_line(canput_arr, 16, 17, 18, 19, 20, 21, 22, 23);
    idxes[79] = create_canput_line(canput_arr, 2, 10, 18, 26, 34, 42, 50, 58);
    idxes[80] = create_canput_line(canput_arr, 5, 13, 21, 29, 37, 45, 53, 61);
    idxes[81] = create_canput_line(canput_arr, 40, 41, 42, 43, 44, 45, 46, 47);
    idxes[82] = create_canput_line(canput_arr, 24, 25, 26, 27, 28, 29, 30, 31);
    idxes[83] = create_canput_line(canput_arr, 3, 11, 19, 27, 35, 43, 51, 59);
    idxes[84] = create_canput_line(canput_arr, 4, 12, 20, 28, 36, 44, 52, 60);
    idxes[85] = create_canput_line(canput_arr, 32, 33, 34, 35, 36, 37, 38, 39);
}

inline void convert_idx(string str){
    int i, j;
    unsigned long long bk = 0, wt = 0;
    char elem;
    board b;
    b.n = 0;
    b.parity = 0;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            elem = str[i * hw + j];
            if (elem != '.'){
                bk |= (unsigned long long)(elem == '0') << (i * hw + j);
                wt |= (unsigned long long)(elem == '1') << (i * hw + j);
                ++b.n;
            }
        }
    }
    for (i = 0; i < b_idx_num; ++i){
        b.b[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (bk >> global_place[i][j]))
                b.b[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (wt >> global_place[i][j]))
                b.b[i] -= pow3[hw_m1 - j];
        }
    }
    //b.print();
    int idxes[86];
    calc_idx(phase_idx, &b, idxes);
    cout << idxes[68] + idxes[69] << " " << ai_player << " ";
    for (i = 0; i < 86; ++i)
        cout << idxes[i] << " ";
    cout << score << endl;
}

#define start_file 0
#define n_files 122

int main(){
    board_init();
    init_evaluation_base();

    int t = 0;

    for (int i = start_file; i < n_files; ++i){
        cerr << "=";
        ostringstream sout;
        sout << setfill('0') << setw(7) << i;
        string file_name = sout.str();
        ifstream ifs("data/records5/" + file_name + ".txt");
        if (ifs.fail()){
            cerr << "evaluation file not exist" << endl;
            exit(1);
        }
        string line;
        while (getline(ifs, line)){
            ++t;
            convert_idx(line);
        }
        if (i % 20 == 9)
            cerr << endl;
    }
    cerr << t << endl;
    return 0;

}