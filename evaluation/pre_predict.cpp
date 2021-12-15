#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

typedef float eval_type;

#define hw 8

#define n_phases 6
#define n_line 6561

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

int pattern_arr[n_phases][2][n_patterns][max_evaluate_idx];
int add_arr[n_phases][2][max_canput][max_surround][max_surround];

int pow3[12];

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
    for (canput = 0; canput < max_canput; ++canput){
        for (sur0 = 0; sur0 < max_surround; ++sur0){
            for (sur1 = 0; sur1 < max_surround; ++sur1)
                add_arr[phase_idx][player_idx][canput][sur0][sur1] = predict_add(canput, sur0, sur1, dense0, bias0, dense1, bias1, dense2, bias2, dense3, bias3);
        }
    }
}

inline void init_evaluation_pred(){
    ifstream ifs("raw_param.txt");
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

void output_param(){
    int phase_idx, player_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, tmp, i, j, k;
    const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10};
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (player_idx = 0; player_idx < 2; ++player_idx){
            cerr << "=";
            for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
                for (pattern_elem = 0; pattern_elem < pow3[pattern_sizes[pattern_idx]]; ++pattern_elem){
                    cout << pattern_arr[phase_idx][player_idx][pattern_idx][pattern_elem] << endl;
                }
            }
            
            for (canput = 0; canput < max_canput; canput += 2){
                for (sur0 = 0; sur0 < max_surround; sur0 += 2){
                    for (sur1 = 0; sur1 < max_surround; sur1 += 2){
                        tmp = 0;
                        for (i = 0; i < 2; ++i){
                            for (j = 0; j < 2; ++j){
                                for (k = 0; k < 2; ++k)
                                    tmp += add_arr[phase_idx][player_idx][canput + i][sur0 + j][sur1 + k];
                            }
                        }
                        //cerr << add_arr[phase_idx][player_idx][canput][sur0][sur1] << " " << tmp / 8 << endl;
                        cout << tmp / 8 << endl;
                    }
                }
            }
            
            /*
            for (canput = 0; canput < max_canput; ++canput){
                for (sur0 = 0; sur0 < max_surround; ++sur0){
                    for (sur1 = 0; sur1 < max_surround; ++sur1){
                        cout << add_arr[phase_idx][player_idx][canput][sur0][sur1] << endl;
                    }
                }
            }
            */
        }
    }
    cerr << endl;
}

int main(){
    pow3[0] = 1;
    for (int i = 1; i < 12; ++i)
        pow3[i] = pow3[i - 1] * 3;
    init_evaluation_pred();
    output_param();
    return 0;
}