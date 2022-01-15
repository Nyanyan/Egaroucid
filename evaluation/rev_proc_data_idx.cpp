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

#define max_surround 80
#define max_canput 50

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

int count_black_arr[n_line];
int count_both_arr[n_line];
int mobility_arr[2][n_line];
int mobility_arr2[2][n_line * n_line];
int surround_arr[2][n_line];
int stability_edge_arr[2][n_line];
int stability_corner_arr[2][n_line];

inline int calc_phase_idx(const board *b){
    return (b->n - 4) / phase_n_stones;
}

inline void init_evaluation_base() {
    int idx, place, b, w;
    bool full_black, full_white;
    for (idx = 0; idx < n_line; ++idx) {
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        mobility_arr[black][idx] = 0;
        mobility_arr[white][idx] = 0;
        surround_arr[black][idx] = 0;
        surround_arr[white][idx] = 0;
        count_black_arr[idx] = 0;
        count_both_arr[idx] = 0;
        stability_edge_arr[black][idx] = 0;
        stability_edge_arr[white][idx] = 0;
        stability_corner_arr[black][idx] = 0;
        stability_corner_arr[white][idx] = 0;
        for (place = 0; place < hw; ++place) {
            if (1 & (b >> place)){
                ++count_black_arr[idx];
                ++count_both_arr[idx];
            } else if (1 & (w >> place)){
                --count_black_arr[idx];
                ++count_both_arr[idx];
            }
            if (place > 0) {
                if ((1 & (b >> (place - 1))) == 0 && (1 & (w >> (place - 1))) == 0) {
                    if (1 & (b >> place))
                        ++surround_arr[black][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[white][idx];
                }
            }
            if (place < hw - 1) {
                if ((1 & (b >> (place + 1))) == 0 && (1 & (w >> (place + 1))) == 0) {
                    if (1 & (b >> place))
                        ++surround_arr[black][idx];
                    else if (1 & (w >> place))
                        ++surround_arr[white][idx];
                }
            }
        }
        for (place = 0; place < hw; ++place) {
            if (legal_arr[black][idx][place])
                ++mobility_arr[black][idx];
            if (legal_arr[white][idx][place])
                ++mobility_arr[white][idx];
        }
        if (count_both_arr[idx] == hw){
            stability_edge_arr[black][idx] = (count_black_arr[idx] + hw) / 2;
            stability_edge_arr[white][idx] += hw - stability_edge_arr[black][idx];
        } else{
            full_black = true;
            full_white = true;
            for (place = 0; place < hw; ++place){
                full_black &= 1 & (b >> place);
                full_white &= 1 & (w >> place);
                stability_edge_arr[black][idx] += full_black;
                stability_edge_arr[white][idx] += full_white;
            }
            full_black = true;
            full_white = true;
            for (place = hw_m1; place >= 0; --place){
                full_black &= 1 & (b >> place);
                full_white &= 1 & (w >> place);
                stability_edge_arr[black][idx] += full_black;
                stability_edge_arr[white][idx] += full_white;
            }
            if (1 & b){
                --stability_edge_arr[black][idx];
                ++stability_corner_arr[black][idx];
            }
            if (1 & (b >> hw_m1)){
                --stability_edge_arr[black][idx];
                ++stability_corner_arr[black][idx];
            }
            if (1 & w){
                --stability_edge_arr[white][idx];
                ++stability_corner_arr[white][idx];
            }
            if (1 & (w >> hw_m1)){
                --stability_edge_arr[white][idx];
                ++stability_corner_arr[white][idx];
            }
        }
    }
}

inline int sfill5(int b){
    return pop_digit[b][2] != 2 ? b - p35m : b;
}

inline int sfill4(int b){
    return pop_digit[b][3] != 2 ? b - p34m : b;
}

inline int sfill3(int b){
    return pop_digit[b][4] != 2 ? b - p33m : b;
}

inline int sfill2(int b){
    return pop_digit[b][5] != 2 ? b - p32m : b;
}

inline int sfill1(int b){
    return pop_digit[b][6] != 2 ? b - p31m : b;
}

inline int calc_canput(const board *b, int p){
    return 
        mobility_arr[p][b->b[0]] + mobility_arr[p][b->b[1]] + mobility_arr[p][b->b[2]] + mobility_arr[p][b->b[3]] + 
        mobility_arr[p][b->b[4]] + mobility_arr[p][b->b[5]] + mobility_arr[p][b->b[6]] + mobility_arr[p][b->b[7]] + 
        mobility_arr[p][b->b[8]] + mobility_arr[p][b->b[9]] + mobility_arr[p][b->b[10]] + mobility_arr[p][b->b[11]] + 
        mobility_arr[p][b->b[12]] + mobility_arr[p][b->b[13]] + mobility_arr[p][b->b[14]] + mobility_arr[p][b->b[15]] + 
        mobility_arr[p][b->b[16] - p35m] + mobility_arr[p][b->b[26] - p35m] + mobility_arr[p][b->b[27] - p35m] + mobility_arr[p][b->b[37] - p35m] + 
        mobility_arr[p][b->b[17] - p34m] + mobility_arr[p][b->b[25] - p34m] + mobility_arr[p][b->b[28] - p34m] + mobility_arr[p][b->b[36] - p34m] + 
        mobility_arr[p][b->b[18] - p33m] + mobility_arr[p][b->b[24] - p33m] + mobility_arr[p][b->b[29] - p33m] + mobility_arr[p][b->b[35] - p33m] + 
        mobility_arr[p][b->b[19] - p32m] + mobility_arr[p][b->b[23] - p32m] + mobility_arr[p][b->b[30] - p32m] + mobility_arr[p][b->b[34] - p32m] + 
        mobility_arr[p][b->b[20] - p31m] + mobility_arr[p][b->b[22] - p31m] + mobility_arr[p][b->b[31] - p31m] + mobility_arr[p][b->b[33] - p31m] + 
        mobility_arr[p][b->b[21]] + mobility_arr[p][b->b[32]];
}

inline int calc_surround(const board *b, int p){
    return 
        surround_arr[p][b->b[0]] + surround_arr[p][b->b[1]] + surround_arr[p][b->b[2]] + surround_arr[p][b->b[3]] + 
        surround_arr[p][b->b[4]] + surround_arr[p][b->b[5]] + surround_arr[p][b->b[6]] + surround_arr[p][b->b[7]] + 
        surround_arr[p][b->b[8]] + surround_arr[p][b->b[9]] + surround_arr[p][b->b[10]] + surround_arr[p][b->b[11]] + 
        surround_arr[p][b->b[12]] + surround_arr[p][b->b[13]] + surround_arr[p][b->b[14]] + surround_arr[p][b->b[15]] + 
        surround_arr[p][sfill5(b->b[16])] + surround_arr[p][sfill5(b->b[26])] + surround_arr[p][sfill5(b->b[27])] + surround_arr[p][sfill5(b->b[37])] + 
        surround_arr[p][sfill4(b->b[17])] + surround_arr[p][sfill4(b->b[25])] + surround_arr[p][sfill4(b->b[28])] + surround_arr[p][sfill4(b->b[36])] + 
        surround_arr[p][sfill3(b->b[18])] + surround_arr[p][sfill3(b->b[24])] + surround_arr[p][sfill3(b->b[29])] + surround_arr[p][sfill3(b->b[35])] + 
        surround_arr[p][sfill2(b->b[19])] + surround_arr[p][sfill2(b->b[23])] + surround_arr[p][sfill2(b->b[30])] + surround_arr[p][sfill2(b->b[34])] + 
        surround_arr[p][sfill1(b->b[20])] + surround_arr[p][sfill1(b->b[22])] + surround_arr[p][sfill1(b->b[31])] + surround_arr[p][sfill1(b->b[33])] + 
        surround_arr[p][b->b[21]] + surround_arr[p][b->b[32]];
}

// pop_digit[idx][left_just]
// pop_mid[idx][right, right - 1][left, right_just]

inline int edge_2x(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][1] * p39 + b->b[y] * p31 + pop_digit[b->b[x]][6];
}

inline int triangle0(int phase_idx, const board *b, int w, int x, int y, int z){
    return b->b[w] / p34 * p36 + b->b[x] / p35 * p33 + b->b[y] / p36 * p31 + b->b[z] / p37;
}

inline int triangle1(int phase_idx, const board *b, int w, int x, int y, int z){
    return reverse_board[b->b[w]] / p34 * p36 + reverse_board[b->b[x]] / p35 * p33 + reverse_board[b->b[y]] / p36 * p31 + reverse_board[b->b[z]] / p37;
}

inline int edge_block(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][0] * p39 + pop_mid[b->b[x]][6][2] * p35 + pop_digit[b->b[x]][7] * p34 + pop_mid[b->b[y]][6][2];
}

inline int cross0(int phase_idx, const board *b, int x, int y, int z){
    return b->b[x] / p34 * p36 + b->b[y] / p35 * p33 + b->b[z] / p35;
}

inline int cross1(int phase_idx, const board *b, int x, int y, int z){
    return reverse_board[b->b[x]] / p34 * p36 + pop_mid[reverse_board[b->b[y]]][7][4] * p33 + pop_mid[reverse_board[b->b[z]]][7][4];
}

inline int corner90(int phase_idx, const board *b, int x, int y, int z){
    return b->b[x] / p35 * p36 + b->b[y] / p35 * p33 + b->b[z] / p35;
}

inline int corner91(int phase_idx, const board *b, int x, int y, int z){
    return reverse_board[b->b[x]] / p35 * p36 + reverse_board[b->b[y]] / p35 * p33 + reverse_board[b->b[z]] / p35;
}

inline int edge_2y(int phase_idx, const board *b, int x, int y){
    return pop_digit[b->b[x]][2] * p39 + b->b[y] * p31 + pop_digit[b->b[x]][5];
}

inline int narrow_triangle0(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return b->b[v] / p33 * p35 + b->b[w] / p36 * p33 + b->b[x] / p37 * p32 + b->b[y] / p37 * p31 + b->b[z] / p37;
}

inline int narrow_triangle1(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return reverse_board[b->b[v]] / p33 * p35 + reverse_board[b->b[w]] / p36 * p33 + pop_digit[b->b[x]][7] * p32 + pop_digit[b->b[y]][7] / p37 * p31 + pop_digit[b->b[z]][7] / p37;
}

inline int fish0(int phase_idx, const board *b, int w, int x, int y, int z){
    return b->b[w] / p36 * p38 + b->b[x] / p34 * p34 + pop_mid[b->b[y]][7][5] * p32 + pop_digit[b->b[z]][1] * p31 + pop_digit[b->b[z]][3];
}

inline int fish1(int phase_idx, const board *b, int w, int x, int y, int z){
    return reverse_board[b->b[w]] / p36 * p38 + reverse_board[b->b[x]] / p34 * p34 + pop_mid[reverse_board[b->b[y]]][7][5] * p32 + pop_digit[b->b[z]][6] * p31 + pop_digit[b->b[z]][4];
}

inline int kite0(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return b->b[v] / p36 * p38 + b->b[w] / p33 * p33 + pop_digit[b->b[x]][1] * p32 + pop_digit[b->b[y]][1] * p31 + pop_digit[b->b[z]][1];
}

inline int kite1(int phase_idx, const board *b, int v, int w, int x, int y, int z){
    return reverse_board[b->b[v]] / p36 * p38 + reverse_board[b->b[w]] / p33 * p33 + pop_digit[b->b[x]][6] * p32 + pop_digit[b->b[y]][6] * p31 + pop_digit[b->b[z]][6];
}

inline int calc_stability(board *b, int p){
    return
        stability_edge_arr[p][b->b[0]] + stability_edge_arr[p][b->b[7]] + stability_edge_arr[p][b->b[8]] + stability_edge_arr[p][b->b[15]] + 
        stability_corner_arr[p][b->b[0]] + stability_corner_arr[p][b->b[7]];
}

inline int create_canput_line(int canput_arr[], int a, int b, int c, int d, int e, int f, int g, int h){
    return (canput_arr[a] << 7) + (canput_arr[b] << 6) + (canput_arr[c] << 5) + (canput_arr[d] << 4) + (canput_arr[e] << 3) + (canput_arr[f] << 2) + (canput_arr[g] << 1) + canput_arr[h];
}

inline void calc_idx(int phase_idx, board *b, int idxes[]){
    idxes[0] = b->b[1];
    idxes[1] = b->b[6];
    idxes[2] = b->b[9];
    idxes[3] = b->b[14];
    idxes[4] = b->b[2];
    idxes[5] = b->b[5];
    idxes[6] = b->b[10];
    idxes[7] = b->b[13];
    idxes[8] = b->b[3];
    idxes[9] = b->b[4];
    idxes[10] = b->b[11];
    idxes[11] = b->b[12];
    idxes[12] = b->b[18] / p33;
    idxes[13] = b->b[24] / p33;
    idxes[14] = b->b[29] / p33;
    idxes[15] = b->b[35] / p33;
    idxes[16] = b->b[19] / p32;
    idxes[17] = b->b[23] / p32;
    idxes[18] = b->b[30] / p32;
    idxes[19] = b->b[34] / p32;
    idxes[20] = b->b[20] / p31;
    idxes[21] = b->b[22] / p31;
    idxes[22] = b->b[31] / p31;
    idxes[23] = b->b[33] / p31;
    idxes[24] = b->b[21];
    idxes[25] = b->b[32];
    idxes[26] = edge_2x(phase_idx, b, 1, 0);
    idxes[27] = edge_2x(phase_idx, b, 6, 7);
    idxes[28] = edge_2x(phase_idx, b, 9, 8);
    idxes[29] = edge_2x(phase_idx, b, 14, 15);
    idxes[30] = triangle0(phase_idx, b, 0, 1, 2, 3);
    idxes[31] = triangle0(phase_idx, b, 7, 6, 5, 4);
    idxes[32] = triangle0(phase_idx, b, 15, 14, 13, 12);
    idxes[33] = triangle1(phase_idx, b, 15, 14, 13, 12);
    idxes[34] = edge_block(phase_idx, b, 0, 1);
    idxes[35] = edge_block(phase_idx, b, 7, 6);
    idxes[36] = edge_block(phase_idx, b, 8, 9);
    idxes[37] = edge_block(phase_idx, b, 15, 14);
    idxes[38] = cross0(phase_idx, b, 21, 20, 22);
    idxes[39] = cross1(phase_idx, b, 21, 20, 22);
    idxes[40] = cross0(phase_idx, b, 32, 31, 33);
    idxes[41] = cross1(phase_idx, b, 32, 31, 33);
    idxes[42] = corner90(phase_idx, b, 0, 1, 2);
    idxes[43] = corner91(phase_idx, b, 0, 1, 2);
    idxes[44] = corner90(phase_idx, b, 7, 6, 5);
    idxes[45] = corner91(phase_idx, b, 7, 6, 5);
    idxes[46] = edge_2y(phase_idx, b, 1, 0);
    idxes[47] = edge_2y(phase_idx, b, 6, 7);
    idxes[48] = edge_2y(phase_idx, b, 9, 8);
    idxes[49] = edge_2y(phase_idx, b, 14, 15);
    idxes[50] = narrow_triangle0(phase_idx, b, 0, 1, 2, 3, 4);
    idxes[51] = narrow_triangle0(phase_idx, b, 7, 6, 5, 4, 3);
    idxes[52] = narrow_triangle1(phase_idx, b, 0, 1, 2, 3, 4);
    idxes[53] = narrow_triangle1(phase_idx, b, 7, 6, 5, 4, 3);
    idxes[54] = fish0(phase_idx, b, 0, 1, 2, 3);
    idxes[55] = fish0(phase_idx, b, 7, 6, 5, 4);
    idxes[56] = fish1(phase_idx, b, 0, 1, 2, 3);
    idxes[57] = fish1(phase_idx, b, 7, 6, 5, 4);
    idxes[58] = kite0(phase_idx, b, 0, 1, 2, 3, 4);
    idxes[59] = kite0(phase_idx, b, 7, 6, 5, 4, 3);
    idxes[60] = kite1(phase_idx, b, 0, 1, 2, 3, 4);
    idxes[61] = kite1(phase_idx, b, 7, 6, 5, 4, 3);
    idxes[62] = min(max_surround - 1, calc_surround(b, black));
    idxes[63] = min(max_surround - 1, calc_surround(b, white));
    idxes[64] = calc_canput(b, black);
    idxes[65] = calc_canput(b, white);
    idxes[66] = calc_stability(b, black);
    idxes[67] = calc_stability(b, white);
    int count = 
        count_black_arr[b->b[0]] + count_black_arr[b->b[1]] + count_black_arr[b->b[2]] + count_black_arr[b->b[3]] + 
        count_black_arr[b->b[4]] + count_black_arr[b->b[5]] + count_black_arr[b->b[6]] + count_black_arr[b->b[7]];
    int filled = 
        count_both_arr[b->b[0]] + count_both_arr[b->b[1]] + count_both_arr[b->b[2]] + count_both_arr[b->b[3]] + 
        count_both_arr[b->b[4]] + count_both_arr[b->b[5]] + count_both_arr[b->b[6]] + count_both_arr[b->b[7]];
    idxes[68] = (filled + count) / 2;
    idxes[69] = (filled - count) / 2;
    int player = b->p;
    int canput_arr[hw2];
    b->p = black;
    for (int i = 0; i < hw2; ++i){
        if (b->legal(i))
            canput_arr[i] = 1;
        else
            canput_arr[i] = 0;
    }
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
    b->p = white;
    for (int i = 0; i < hw2; ++i){
        if (b->legal(i))
            canput_arr[i] = 1;
        else
            canput_arr[i] = 0;
    }
    idxes[86] = create_canput_line(canput_arr, 0, 1, 2, 3, 4, 5, 6, 7);
    idxes[87] = create_canput_line(canput_arr, 0, 8, 16, 24, 32, 40, 48, 56);
    idxes[88] = create_canput_line(canput_arr, 7, 15, 23, 31, 39, 47, 55, 63);
    idxes[89] = create_canput_line(canput_arr, 56, 57, 58, 59, 60, 61, 62, 63);
    idxes[90] = create_canput_line(canput_arr, 8, 9, 10, 11, 12, 13, 14, 15);
    idxes[91] = create_canput_line(canput_arr, 1, 9, 17, 25, 33, 41, 49, 57);
    idxes[92] = create_canput_line(canput_arr, 6, 14, 22, 30, 38, 46, 54, 62);
    idxes[93] = create_canput_line(canput_arr, 48, 49, 50, 51, 52, 53, 54, 55);
    idxes[94] = create_canput_line(canput_arr, 16, 17, 18, 19, 20, 21, 22, 23);
    idxes[95] = create_canput_line(canput_arr, 2, 10, 18, 26, 34, 42, 50, 58);
    idxes[96] = create_canput_line(canput_arr, 5, 13, 21, 29, 37, 45, 53, 61);
    idxes[97] = create_canput_line(canput_arr, 40, 41, 42, 43, 44, 45, 46, 47);
    idxes[98] = create_canput_line(canput_arr, 24, 25, 26, 27, 28, 29, 30, 31);
    idxes[99] = create_canput_line(canput_arr, 3, 11, 19, 27, 35, 43, 51, 59);
    idxes[100] = create_canput_line(canput_arr, 4, 12, 20, 28, 36, 44, 52, 60);
    idxes[101] = create_canput_line(canput_arr, 32, 33, 34, 35, 36, 37, 38, 39);
}

inline void convert_idx(string str){
    int i, j;
    istringstream iss(str);
    int phase_idx, ai_player, score;
    board b;
    iss >> phase_idx;
    iss >> ai_player;
    int arr[hw2];
    int tmp;
    for (i = 0; i < 12; ++i)
        iss >> tmp;
    iss >> tmp;
    arr[3] = pop_digit[tmp][3];
    arr[12] = pop_digit[tmp][4];
    arr[21] = pop_digit[tmp][5];
    arr[30] = pop_digit[tmp][6];
    arr[39] = pop_digit[tmp][7];
    iss >> tmp;
    arr[24] = pop_digit[tmp][3];
    arr[33] = pop_digit[tmp][4];
    arr[42] = pop_digit[tmp][5];
    arr[51] = pop_digit[tmp][6];
    arr[60] = pop_digit[tmp][7];
    iss >> tmp;
    iss >> tmp;
    iss >> tmp;
    arr[2] = pop_digit[tmp][2];
    arr[11] = pop_digit[tmp][3];
    arr[20] = pop_digit[tmp][4];
    arr[29] = pop_digit[tmp][5];
    arr[38] = pop_digit[tmp][6];
    arr[47] = pop_digit[tmp][7];
    iss >> tmp;
    arr[16] = pop_digit[tmp][2];
    arr[25] = pop_digit[tmp][3];
    arr[34] = pop_digit[tmp][4];
    arr[43] = pop_digit[tmp][5];
    arr[52] = pop_digit[tmp][6];
    arr[61] = pop_digit[tmp][7];
    iss >> tmp;
    iss >> tmp;
    iss >> tmp;
    arr[1] = pop_digit[tmp][1];
    arr[10] = pop_digit[tmp][2];
    arr[19] = pop_digit[tmp][3];
    arr[28] = pop_digit[tmp][4];
    arr[37] = pop_digit[tmp][5];
    arr[46] = pop_digit[tmp][6];
    arr[55] = pop_digit[tmp][7];
    iss >> tmp;
    arr[8] = pop_digit[tmp][1];
    arr[17] = pop_digit[tmp][2];
    arr[26] = pop_digit[tmp][3];
    arr[35] = pop_digit[tmp][4];
    arr[44] = pop_digit[tmp][5];
    arr[53] = pop_digit[tmp][6];
    arr[62] = pop_digit[tmp][7];
    iss >> tmp;
    iss >> tmp;
    iss >> tmp;
    arr[0] = pop_digit[tmp][0];
    arr[9] = pop_digit[tmp][1];
    arr[18] = pop_digit[tmp][2];
    arr[27] = pop_digit[tmp][3];
    arr[36] = pop_digit[tmp][4];
    arr[45] = pop_digit[tmp][5];
    arr[54] = pop_digit[tmp][6];
    arr[63] = pop_digit[tmp][7];
    for (i = 0; i < 6; ++i)
        iss >> tmp;
    iss >> tmp;
    arr[56] = tmp / p39;
    arr[57] = tmp / p38 % 3;
    arr[58] = tmp / p37 % 3;
    arr[59] = tmp / p36 % 3;
    arr[48] = tmp / p35 % 3;
    arr[49] = tmp / p34 % 3;
    arr[50] = tmp / p33 % 3;
    arr[40] = tmp / p32 % 3;
    arr[41] = tmp / p31 % 3;
    arr[32] = tmp % 3;
    iss >> tmp;
    arr[7] = tmp / p39;
    arr[15] = tmp / p38 % 3;
    arr[23] = tmp / p37 % 3;
    arr[31] = tmp / p36 % 3;
    arr[6] = tmp / p35 % 3;
    arr[14] = tmp / p34 % 3;
    arr[22] = tmp / p33 % 3;
    arr[5] = tmp / p32 % 3;
    arr[13] = tmp / p31 % 3;
    arr[4] = tmp % 3;
    /*
    for (i = 0; i < hw2; ++i){
        if (arr[i] == 2)
            cerr << ".";
        else
            cerr << arr[i];
    }
    */
    //cerr << endl;
    for (i = 0; i < 20; ++i)
        iss >> tmp;
    iss >> score;
    b.translate_from_arr(arr, ai_player);
    //b.print();
    int idxes[102];
    calc_idx(phase_idx, &b, idxes);
    cout << idxes[68] + idxes[69] << " " << ai_player << " ";
    for (i = 0; i < 102; ++i)
        cout << idxes[i] << " ";
    cout << score << endl;
}

int main(){
    board_init();
    init_evaluation_base();

    int t = 0;

    cerr << "=";
    ifstream ifs("records3_2.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    while (getline(ifs, line)){
        ++t;
        convert_idx(line);
        if ((t & 0b11111111111111) == 0)
            cerr << "\r " << t;
    }
    return 0;

}