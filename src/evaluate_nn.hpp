#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"

using namespace std;

typedef float eval_type;

#define n_patterns 11
#define n_dense0 32
#define n_dense1 32
#define n_dense2 32
#define n_add_input 3
#define n_add_dense0 8
#define n_add_dense1 8
#define n_add_dense2 8
#define max_canput 40
#define max_surround 80
#define max_evaluate_idx 59049

#define sc_w 6400
#define step 100

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
int pattern_arr[n_phases][2][n_patterns][max_evaluate_idx];
int add_arr[n_phases][2][max_canput + 1][max_surround + 1][max_surround + 1];

inline eval_type max(eval_type a, eval_type b){
    return a < b ? b : a;
}

inline eval_type min(eval_type a, eval_type b){
    return a > b ? b : a;
}

inline eval_type leaky_relu(eval_type x){
    return max(0.01 * x, x);
}

inline int predict(int pattern_size, eval_type in_arr[], eval_type dense0[n_dense0][20], eval_type bias0[n_dense0], eval_type dense1[n_dense1][n_dense0], eval_type bias1[n_dense1], eval_type dense2[n_dense2][n_dense1], eval_type bias2[n_dense2], eval_type dense3[n_dense2], eval_type bias3){
    eval_type hidden0[n_dense0], hidden1[n_dense1], hidden2, res;
    int i, j;
    for (i = 0; i < n_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < pattern_size * 2; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    for (i = 0; i < n_dense1; ++i){
        hidden1[i] = bias1[i];
        for (j = 0; j < n_dense0; ++j)
            hidden1[i] += hidden0[j] * dense1[i][j];
        hidden1[i] = leaky_relu(hidden1[i]);
    }
    res = bias3;
    for (i = 0; i < n_dense2; ++i){
        hidden2 = bias2[i];
        for (j = 0; j < n_dense1; ++j)
            hidden2 += hidden1[j] * dense2[i][j];
        hidden2 = leaky_relu(hidden2);
        res += hidden2 * dense3[i];
    }
    return round(res * sc_w);
}

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    if (pattern_idx <= 7){
        for (int i = 0; i < pattern_size; ++i)
            res += pow3[i] * calc_pop(idx, i, pattern_size);
    } else if (pattern_idx == 8){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 4, pattern_size);
        res += p37 * calc_pop(idx, 7, pattern_size);
        res += p36 * calc_pop(idx, 9, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 5, pattern_size);
        res += p33 * calc_pop(idx, 8, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 6, pattern_size);
        res += calc_pop(idx, 3, pattern_size);
    } else if (pattern_idx == 9){
        res += p39 * calc_pop(idx, 5, pattern_size);
        res += p38 * calc_pop(idx, 4, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 2, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 0, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 8, pattern_size);
        res += p31 * calc_pop(idx, 7, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    } else if (pattern_idx == 10){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 1, pattern_size);
        res += p37 * calc_pop(idx, 2, pattern_size);
        res += p36 * calc_pop(idx, 3, pattern_size);
        res += p35 * calc_pop(idx, 7, pattern_size);
        res += p34 * calc_pop(idx, 8, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 4, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    }
    return res;
}

inline void pre_evaluation(int pattern_idx, int phase_idx, int player_idx, int evaluate_idx, int pattern_size, eval_type dense0[n_dense0][20], eval_type bias0[n_dense0], eval_type dense1[n_dense1][n_dense0], eval_type bias1[n_dense1], eval_type dense2[n_dense2][n_dense1], eval_type bias2[n_dense2], eval_type dense3[n_dense2], eval_type bias3){
    int digit, idx, i;
    eval_type arr[20];
    int tmp_pattern_arr[max_evaluate_idx];
    for (idx = 0; idx < pow3[pattern_size]; ++idx){
        for (i = 0; i < pattern_size; ++i){
            digit = (idx / pow3[pattern_size - 1 - i]) % 3;
            if (digit == 0){
                arr[i] = 1.0;
                arr[pattern_size + i] = 0.0;
            } else if (digit == 1){
                arr[i] = 0.0;
                arr[pattern_size + i] = 1.0;
            } else{
                arr[i] = 0.0;
                arr[pattern_size + i] = 0.0;
            }
        }
        pattern_arr[phase_idx][player_idx][evaluate_idx][idx] = predict(pattern_size, arr, dense0, bias0, dense1, bias1, dense2, bias2, dense3, bias3);
        tmp_pattern_arr[calc_rev_idx(pattern_idx, pattern_size, idx)] = pattern_arr[phase_idx][player_idx][evaluate_idx][idx];
    }
    for (idx = 0; idx < pow3[pattern_size]; ++idx){
        pattern_arr[phase_idx][player_idx][evaluate_idx][idx] += tmp_pattern_arr[idx];
    }
}

inline int predict_add(int canput, int sur0, int sur1, eval_type dense0[n_add_dense0][n_add_input], eval_type bias0[n_add_dense0], eval_type dense1[n_add_dense1][n_add_dense0], eval_type bias1[n_add_dense1], eval_type dense2[n_add_dense2][n_add_dense1], eval_type bias2[n_add_dense2], eval_type dense3[n_add_dense2], eval_type bias3){
    eval_type hidden0[n_add_dense0], hidden1[n_add_dense1], in_arr[n_add_input], hidden2, res;
    int i, j;
    in_arr[0] = ((eval_type)canput - 15.0) / 15.0;
    in_arr[1] = ((eval_type)sur0 - 15.0) / 15.0;
    in_arr[2] = ((eval_type)sur1 - 15.0) / 15.0;
    for (i = 0; i < n_add_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < n_add_input; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    for (i = 0; i < n_add_dense1; ++i){
        hidden1[i] = bias1[i];
        for (j = 0; j < n_add_dense0; ++j)
            hidden1[i] += hidden0[j] * dense1[i][j];
        hidden1[i] = leaky_relu(hidden1[i]);
    }
    res = bias3;
    for (i = 0; i < n_add_dense2; ++i){
        hidden2 = bias2[i];
        for (j = 0; j < n_add_dense1; ++j)
            hidden2 += hidden1[j] * dense2[i][j];
        hidden2 = leaky_relu(hidden2);
        res += hidden2 * dense3[i];
    }
    return round(res * sc_w);
}

inline void pre_evaluation_add(int phase_idx, int player_idx, eval_type dense0[n_add_dense0][n_add_input], eval_type bias0[n_add_dense0], eval_type dense1[n_add_dense1][n_add_dense0], eval_type bias1[n_add_dense1], eval_type dense2[n_add_dense2][n_add_dense1], eval_type bias2[n_add_dense2], eval_type dense3[n_add_dense2], eval_type bias3){
    int canput, sur0, sur1;
    for (canput = 0; canput <= max_canput; ++canput){
        for (sur0 = 0; sur0 <= max_surround; ++sur0){
            for (sur1 = 0; sur1 <= max_surround; ++sur1)
                add_arr[phase_idx][player_idx][canput][sur0][sur1] = predict_add(canput, sur0, sur1, dense0, bias0, dense1, bias1, dense2, bias2, dense3, bias3);
        }
    }
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

inline void init_evaluation_pred(){
    ifstream ifs("resources/raw_param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int i, j, phase_idx, pattern_idx, player_idx;
    eval_type dense0[n_dense0][20];
    eval_type bias0[n_dense0];
    eval_type dense1[n_dense1][n_dense0];
    eval_type bias1[n_dense1];
    eval_type dense2[n_dense2][n_dense1];
    eval_type bias2[n_dense2];
    eval_type dense3[n_dense2];
    eval_type bias3;
    eval_type add_dense0[n_add_dense0][n_add_input];
    eval_type add_bias0[n_add_dense0];
    eval_type add_dense1[n_add_dense1][n_add_dense0];
    eval_type add_bias1[n_add_dense1];
    eval_type add_dense2[n_add_dense2][n_add_dense1];
    eval_type add_bias2[n_add_dense2];
    eval_type add_dense3[n_add_dense2];
    eval_type add_bias3;
    const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10};
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            cerr << "=";
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                for (i = 0; i < pattern_sizes[pattern_idx] * 2; ++i){
                    for (j = 0; j < n_dense0; ++j){
                        getline(ifs, line);
                        dense0[j][i] = stof(line);
                    }
                }
                for (i = 0; i < n_dense0; ++i){
                    getline(ifs, line);
                    bias0[i] = stof(line);
                }
                for (i = 0; i < n_dense0; ++i){
                    for (j = 0; j < n_dense1; ++j){
                        getline(ifs, line);
                        dense1[j][i] = stof(line);
                    }
                }
                for (i = 0; i < n_dense1; ++i){
                    getline(ifs, line);
                    bias1[i] = stof(line);
                }
                for (i = 0; i < n_dense1; ++i){
                    for (j = 0; j < n_dense2; ++j){
                        getline(ifs, line);
                        dense2[j][i] = stof(line);
                    }
                }
                for (i = 0; i < n_dense2; ++i){
                    getline(ifs, line);
                    bias2[i] = stof(line);
                }
                for (i = 0; i < n_dense2; ++i){
                    getline(ifs, line);
                    dense3[i] = stof(line);
                }
                getline(ifs, line);
                bias3 = stof(line);
                pre_evaluation(pattern_idx, phase_idx, player_idx, pattern_idx, pattern_sizes[pattern_idx], dense0, bias0, dense1, bias1, dense2, bias2, dense3, bias3);
            }
            for (i = 0; i < n_add_input; ++i){
                for (j = 0; j < n_add_dense0; ++j){
                    getline(ifs, line);
                    add_dense0[j][i] = stof(line);
                }
            }
            for (i = 0; i < n_add_dense0; ++i){
                getline(ifs, line);
                add_bias0[i] = stof(line);
            }
            for (i = 0; i < n_add_dense0; ++i){
                for (j = 0; j < n_add_dense1; ++j){
                    getline(ifs, line);
                    add_dense1[j][i] = stof(line);
                }
            }
            for (i = 0; i < n_add_dense1; ++i){
                getline(ifs, line);
                add_bias1[i] = stof(line);
            }
            for (i = 0; i < n_add_dense1; ++i){
                for (j = 0; j < n_add_dense2; ++j){
                    getline(ifs, line);
                    add_dense2[j][i] = stof(line);
                }
            }
            for (i = 0; i < n_add_dense2; ++i){
                getline(ifs, line);
                add_bias2[i] = stof(line);
            }
            for (i = 0; i < n_add_dense2; ++i){
                getline(ifs, line);
                add_dense3[i] = stof(line);
            }
            getline(ifs, line);
            add_bias3 = stof(line);
            pre_evaluation_add(phase_idx, player_idx, add_dense0, add_bias0, add_dense1, add_bias1, add_dense2, add_bias2, add_dense3, add_bias3);
        }
    }
    cerr << endl;
}

inline void evaluate_init(){
    init_evaluation_base();
    #if !EVAL_MODE
        init_evaluation_pred();
    #endif
    cerr << "evaluation function initialized" << endl;
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

inline int calc_canput(const board *b){
    return 
        mobility_arr[b->p][b->b[0]] + mobility_arr[b->p][b->b[1]] + mobility_arr[b->p][b->b[2]] + mobility_arr[b->p][b->b[3]] + 
        mobility_arr[b->p][b->b[4]] + mobility_arr[b->p][b->b[5]] + mobility_arr[b->p][b->b[6]] + mobility_arr[b->p][b->b[7]] + 
        mobility_arr[b->p][b->b[8]] + mobility_arr[b->p][b->b[9]] + mobility_arr[b->p][b->b[10]] + mobility_arr[b->p][b->b[11]] + 
        mobility_arr[b->p][b->b[12]] + mobility_arr[b->p][b->b[13]] + mobility_arr[b->p][b->b[14]] + mobility_arr[b->p][b->b[15]] + 
        mobility_arr[b->p][b->b[16] - p35m] + mobility_arr[b->p][b->b[26] - p35m] + mobility_arr[b->p][b->b[27] - p35m] + mobility_arr[b->p][b->b[37] - p35m] + 
        mobility_arr[b->p][b->b[17] - p34m] + mobility_arr[b->p][b->b[25] - p34m] + mobility_arr[b->p][b->b[28] - p34m] + mobility_arr[b->p][b->b[36] - p34m] + 
        mobility_arr[b->p][b->b[18] - p33m] + mobility_arr[b->p][b->b[24] - p33m] + mobility_arr[b->p][b->b[29] - p33m] + mobility_arr[b->p][b->b[35] - p33m] + 
        mobility_arr[b->p][b->b[19] - p32m] + mobility_arr[b->p][b->b[23] - p32m] + mobility_arr[b->p][b->b[30] - p32m] + mobility_arr[b->p][b->b[34] - p32m] + 
        mobility_arr[b->p][b->b[20] - p31m] + mobility_arr[b->p][b->b[22] - p31m] + mobility_arr[b->p][b->b[31] - p31m] + mobility_arr[b->p][b->b[33] - p31m] + 
        mobility_arr[b->p][b->b[21]] + mobility_arr[b->p][b->b[32]];
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

inline eval_type edge_2x(int phase_idx, const board *b, int x, int y){
    return pattern_arr[phase_idx][b->p][7][pop_digit[b->b[x]][1] * p39 + b->b[y] * p31 + pop_digit[b->b[x]][6]];
}

inline eval_type triangle0(int phase_idx, const board *b, int w, int x, int y, int z){
    return pattern_arr[phase_idx][b->p][8][b->b[w] / p34 * p36 + b->b[x] / p35 * p33 + b->b[y] / p36 * p31 + b->b[z] / p37];
}

inline eval_type triangle1(int phase_idx, const board *b, int w, int x, int y, int z){
    return pattern_arr[phase_idx][b->p][8][reverse_board[b->b[w]] / p34 * p36 + reverse_board[b->b[x]] / p35 * p33 + reverse_board[b->b[y]] / p36 * p31 + reverse_board[b->b[z]] / p37];
}

inline eval_type edge_block(int phase_idx, const board *b, int x, int y){
    return pattern_arr[phase_idx][b->p][9][pop_digit[b->b[x]][0] * p39 + pop_mid[b->b[x]][6][2] * p35 + pop_digit[b->b[x]][7] * p34 + pop_mid[b->b[y]][6][2]];
}

inline eval_type cross(int phase_idx, const board *b, int x, int y, int z){
    return 
        pattern_arr[phase_idx][b->p][10][b->b[x] / p34 * p36 + b->b[y] / p35 * p33 + b->b[z] / p35] + 
        pattern_arr[phase_idx][b->p][10][reverse_board[b->b[x]] / p34 * p36 + pop_mid[reverse_board[b->b[y]]][7][4] * p33 + pop_mid[reverse_board[b->b[z]]][7][4]];
}

inline eval_type calc_pattern(int phase_idx, const board *b){
    return 
        pattern_arr[phase_idx][b->p][0][b->b[1]] + pattern_arr[phase_idx][b->p][0][b->b[6]] + pattern_arr[phase_idx][b->p][0][b->b[9]] + pattern_arr[phase_idx][b->p][0][b->b[14]] + 
        pattern_arr[phase_idx][b->p][1][b->b[2]] + pattern_arr[phase_idx][b->p][1][b->b[5]] + pattern_arr[phase_idx][b->p][1][b->b[10]] + pattern_arr[phase_idx][b->p][1][b->b[13]] + 
        pattern_arr[phase_idx][b->p][2][b->b[3]] + pattern_arr[phase_idx][b->p][2][b->b[4]] + pattern_arr[phase_idx][b->p][2][b->b[11]] + pattern_arr[phase_idx][b->p][2][b->b[12]] + 
        pattern_arr[phase_idx][b->p][3][b->b[18] / p33] + pattern_arr[phase_idx][b->p][3][b->b[24] / p33] + pattern_arr[phase_idx][b->p][3][b->b[29] / p33] + pattern_arr[phase_idx][b->p][3][b->b[35] / p33] + 
        pattern_arr[phase_idx][b->p][4][b->b[19] / p32] + pattern_arr[phase_idx][b->p][4][b->b[23] / p32] + pattern_arr[phase_idx][b->p][4][b->b[30] / p32] + pattern_arr[phase_idx][b->p][4][b->b[34] / p32] + 
        pattern_arr[phase_idx][b->p][5][b->b[20] / p31] + pattern_arr[phase_idx][b->p][5][b->b[22] / p31] + pattern_arr[phase_idx][b->p][5][b->b[31] / p31] + pattern_arr[phase_idx][b->p][5][b->b[33] / p31] + 
        pattern_arr[phase_idx][b->p][6][b->b[21]] + pattern_arr[phase_idx][b->p][6][b->b[32]] + 
        edge_2x(phase_idx, b, 1, 0) + edge_2x(phase_idx, b, 6, 7) + edge_2x(phase_idx, b, 9, 8) + edge_2x(phase_idx, b, 14, 15) + 
        triangle0(phase_idx, b, 0, 1, 2, 3) + triangle0(phase_idx, b, 7, 6, 5, 4) + triangle0(phase_idx, b, 15, 14, 13, 12) + triangle1(phase_idx, b, 15, 14, 13, 12) + 
        edge_block(phase_idx, b, 0, 1) + edge_block(phase_idx, b, 7, 6) + edge_block(phase_idx, b, 8, 9) + edge_block(phase_idx, b, 15, 14) + 
        cross(phase_idx, b, 21, 20, 22) + cross(phase_idx, b, 32, 31, 33);
}

inline int mid_evaluate(const board *b){
    int phase_idx = calc_phase_idx(b), canput, sur0, sur1;
    canput = min(max_canput, calc_canput(b));
    sur0 = min(max_surround, calc_surround(b, black));
    sur1 = min(max_surround, calc_surround(b, white));
    return calc_pattern(phase_idx, b) + add_arr[phase_idx][b->p][canput][sur0][sur1];
}

inline int end_evaluate(const board *b){
    int count = (b->p ? -1 : 1) * 
        (count_black_arr[b->b[0]] + count_black_arr[b->b[1]] + count_black_arr[b->b[2]] + count_black_arr[b->b[3]] + 
        count_black_arr[b->b[4]] + count_black_arr[b->b[5]] + count_black_arr[b->b[6]] + count_black_arr[b->b[7]]);
    int empty = hw2 - 
        count_both_arr[b->b[0]] - count_both_arr[b->b[1]] - count_both_arr[b->b[2]] - count_both_arr[b->b[3]] - 
        count_both_arr[b->b[4]] - count_both_arr[b->b[5]] - count_both_arr[b->b[6]] - count_both_arr[b->b[7]];
    if (count > 0)
        count += empty;
    else if (count < 0)
        count -= empty;
    return count * step;
}