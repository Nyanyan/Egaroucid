
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>
#include "board.hpp"

#define PHASE_N_STONES_LEARN 1

#ifndef N_SYMMETRY_PATTERNS
#define N_SYMMETRY_PATTERNS 66
#endif
#define N_CANPUT_PATTERNS 8
#define N_RAW_PARAMS 77

#define N_PATTERNS 16
#define N_EVAL (N_PATTERNS + 3 + 5)

#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STONE_NUM 65
#define MAX_EVALUATE_IDX 65536

int sa_phase, sa_player;

#define PNO 0

#define P30 1
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

#define P40 1
#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

#define P20 1
#define P21 2
#define P22 4
#define P23 8
#define P24 16
#define P25 32
#define P26 64
#define P27 128

#define STEP 256
#define STEP_2 128
#define SC_W (STEP * HW2)

#define N_DATA 70000000

double beta = 0.001;
unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

double alpha[N_EVAL][MAX_EVALUATE_IDX];

constexpr int pattern_sizes[N_EVAL] = {
    8, 8, 8, 5, 6, 7, 8,
    10, 10, 9, 10, 10, 10, 10,
    8, 7,
    -1, -1, -1,
    6, 6, 6, 6,
    6
};

constexpr int eval_sizes[N_EVAL] = {
    P38, P38, P38, P35, P36, P37, P38,
    P310, P310, P39, P310, P310, P310, P310,
    52488, 34992,
    MAX_SURROUND * MAX_SURROUND, MAX_CANPUT * MAX_CANPUT, MAX_STONE_NUM * MAX_STONE_NUM,
    4096, 4096, 4096, 4096,
    4096
};

constexpr int pattern_nums[N_RAW_PARAMS] = {
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12, 12, 12, 12,
    13, 13, 13, 13,
    14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15,

    16, 17, 18,

    19, 20, 21, 22,
    23, 23, 23, 23
};

double eval_arr[N_EVAL][MAX_EVALUATE_IDX];
vector<vector<int>> test_data;
int* test_data_kernel = 0;
double test_labels[N_DATA];
int nums;
double scores;
vector<int> test_memo[N_EVAL][MAX_EVALUATE_IDX];
vector<double> test_scores, pre_calc_scores;
unordered_set<int> used_idxes[N_EVAL];
vector<int> used_idxes_vector[N_EVAL];
int rev_idxes[N_EVAL][MAX_EVALUATE_IDX];
int rev_idxes_mobility[N_EVAL][MAX_EVALUATE_IDX][4];
int pow4[8];
int pow3[11];
int n_data_score[129];
int n_data_idx[N_EVAL][MAX_EVALUATE_IDX][129];
double bias[129];


#define PRE_CALC_SCORE_INTERVAL 524288
double pre_calc_scores_arr[PRE_CALC_SCORE_INTERVAL];

void initialize_param() {
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx) {
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem)
            eval_arr[pattern_idx][pattern_elem] = 0;
    }
}

void input_param_onephase(string file) {
    ifstream ifs(file);
    if (ifs.fail()) {
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t = 0;
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx) {
        cerr << "=";
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem) {
            ++t;
            getline(ifs, line);
            eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
    cerr << t << endl;
}

inline double calc_score(int phase, int i);

struct Adj_info {
    int n_data;
    int n_params;
    int use_params;
};

Adj_info input_test_data(int argc, char* argv[]) {
    Adj_info res;
    int i, j, k;
    /*
    ifstream ifs("big_data.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    */
    int phase, player, score;
    int t = 0, u = 0;
    nums = 0;
    for (j = 0; j < N_EVAL; ++j) {
        used_idxes[j].clear();
        for (k = 0; k < MAX_EVALUATE_IDX; ++k)
            test_memo[j][k].clear();
    }
    test_scores.clear();
    for (i = 0; i < N_EVAL; ++i) {
        for (j = 0; j < MAX_EVALUATE_IDX; ++j) {
            for (k = 0; k < 129; ++k)
                n_data_idx[i][j][k] = 0;
        }
    }
    for (i = 0; i < 129; ++i)
        n_data_score[i] = 0;
    int sur, canput, stab, num;
    FILE* fp;
    int file_idxes[N_RAW_PARAMS];
    for (int file_idx = 7; file_idx < 8 /*argc*/ ; ++file_idx) {
        /*
        cerr << argv[file_idx] << endl;
        if (fopen_s(&fp, argv[file_idx], "rb") != 0) {
            cerr << "can't open " << argv[file_idx] << endl;
            continue;
        }
        */
        
        if (fopen_s(&fp, "data3_06.dat", "rb") != 0) {
            cerr << "can't open " << endl;
            continue;
        }
        
        while (t < N_DATA) {
            ++t;
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                cerr << '\r' << t;
            if (fread(&phase, 4, 1, fp) < 1)
                break;
            phase = (phase - 4) / PHASE_N_STONES_LEARN;
            fread(&player, 4, 1, fp);
            fread(file_idxes, 4, N_RAW_PARAMS, fp);
            fread(&score, 4, 1, fp);
            if (phase == sa_phase) {
                vector<int> file_idxes_vector;
                for (int i = 0; i < N_RAW_PARAMS; ++i)
                    file_idxes_vector.emplace_back(file_idxes[i]);
                test_data.emplace_back(file_idxes_vector);
                ++u;
                for (i = 0; i < N_RAW_PARAMS; ++i)
                    used_idxes[pattern_nums[i]].emplace(test_data[nums][i]);
                test_labels[nums] = score * STEP;
                for (i = 0; i < N_RAW_PARAMS; ++i)
                    test_memo[pattern_nums[i]][test_data[nums][i]].push_back(nums);
                test_scores.push_back(0);
                pre_calc_scores.push_back(0);
                ++n_data_score[score + 64];
                for (i = 0; i < N_RAW_PARAMS; ++i)
                    ++n_data_idx[pattern_nums[i]][test_data[nums][i]][score + 64];
                ++nums;
            }
        }
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
    for (i = 0; i < N_EVAL; ++i) {
        for (auto elem : used_idxes[i])
            used_idxes_vector[i].push_back(elem);
    }

    cerr << "n_data " << u << endl;
    res.n_data = u;

    u = 0;
    for (i = 0; i < N_EVAL; ++i)
        u += eval_sizes[i];
    cerr << "n_all_param " << u << endl;
    res.n_params = u;
    u = 0;
    for (i = 0; i < N_EVAL; ++i) {
        u += (int)used_idxes[i].size();
    }
    cerr << "used_param " << u << endl;
    res.use_params = u;

    int zero_score_N_DATA = n_data_score[64];
    int wipeout_N_DATA = zero_score_N_DATA / 2;
    int modified_n_data;
    for (int i = 0; i < 129; ++i) {
        if (n_data_score[i] == 0)
            continue;
        if (i <= 64)
            modified_n_data = wipeout_N_DATA + (zero_score_N_DATA - wipeout_N_DATA) * i / 64;
        else
            modified_n_data = zero_score_N_DATA - (zero_score_N_DATA - wipeout_N_DATA) * (i - 64) / 64;
        bias[i] = 1.0; //(double)modified_n_data / n_data_score[i];
        //cerr << modified_n_data << " " << bias[i] << endl;
    }

    double n_weighted_data;
    for (i = 0; i < N_EVAL; ++i) {
        for (const int& used_idx : used_idxes_vector[i]) {
            n_weighted_data = 0.0;
            for (j = 0; j < 129; ++j)
                n_weighted_data += bias[j] * (double)n_data_idx[i][used_idx][j];
            if (used_idx < N_PATTERNS || N_PATTERNS + 4 <= used_idx) {
                for (j = 0; j < 129; ++j)
                    n_weighted_data += bias[j] * (double)n_data_idx[i][rev_idxes[i][used_idx]][j];
            }
            alpha[i][used_idx] = beta / max(50.0, n_weighted_data);
        }
    }
    return res;
}

void output_param_onephase() {
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx) {
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem) {
            cout << round(eval_arr[pattern_idx][pattern_elem]) << endl;
        }
    }
    cerr << endl;
}

inline double loss(double x, int siz) {
    //double sq_size = sqrt((double)siz);
    //double tmp = (double)x / sq_size;
    return (double)x / (double)siz * (double)x;
}

inline double calc_score(int phase, int i) {
    int res = 0;
    for (int j = 0; j < N_RAW_PARAMS; ++j)
        res += eval_arr[pattern_nums[j]][test_data[i][j]];
    return max(-SC_W, min(SC_W, res));
}

__global__ void calc_score_kernel(int phase, double* arr, int* test_data_arr, double* res_arr) {
    int i = threadIdx.x;
    constexpr int pattern_nums_local[N_RAW_PARAMS] = {
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10,
        11, 11, 11, 11,
        12, 12, 12, 12,
        13, 13, 13, 13,
        14, 14, 14, 14,
        15, 15, 15, 15, 15, 15, 15, 15,

        16, 17, 18,

        19, 20, 21, 22,
        23, 23, 23, 23
    };
    int res = 0;
    for (int j = 0; j < N_RAW_PARAMS; ++j)
        res += arr[pattern_nums_local[j] * MAX_EVALUATE_IDX + test_data_arr[i * N_RAW_PARAMS + j]];
    res_arr[i] = (double)max(-SC_W, min(SC_W, res));
}

inline int calc_pop(int a, int b, int s) {
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_pop4(int a, int b, int s) {
    return (a / pow4[s - 1 - b]) % 4;
}

inline int calc_pop2(int a, int b) {
    return 1 & (a >> b);
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx) {
    int res = 0;
    if (pattern_idx <= 7 || pattern_idx == 12) { // edge + 2X & middle edge
        for (int i = 0; i < pattern_size; ++i)
            res += pow3[i] * calc_pop(idx, i, pattern_size);
    }
    else if (pattern_idx == 8) { //edge block
        res += P39 * calc_pop(idx, 5, pattern_size);
        res += P38 * calc_pop(idx, 4, pattern_size);
        res += P37 * calc_pop(idx, 3, pattern_size);
        res += P36 * calc_pop(idx, 2, pattern_size);
        res += P35 * calc_pop(idx, 1, pattern_size);
        res += P34 * calc_pop(idx, 0, pattern_size);
        res += P33 * calc_pop(idx, 9, pattern_size);
        res += P32 * calc_pop(idx, 8, pattern_size);
        res += P31 * calc_pop(idx, 7, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    }
    else if (pattern_idx == 9) { // corner9
        res += P38 * calc_pop(idx, 0, pattern_size);
        res += P37 * calc_pop(idx, 3, pattern_size);
        res += P36 * calc_pop(idx, 6, pattern_size);
        res += P35 * calc_pop(idx, 1, pattern_size);
        res += P34 * calc_pop(idx, 4, pattern_size);
        res += P33 * calc_pop(idx, 7, pattern_size);
        res += P32 * calc_pop(idx, 2, pattern_size);
        res += P31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 8, pattern_size);
    }
    else if (pattern_idx == 10) { // triangle
        res += P39 * calc_pop(idx, 0, pattern_size);
        res += P38 * calc_pop(idx, 4, pattern_size);
        res += P37 * calc_pop(idx, 7, pattern_size);
        res += P36 * calc_pop(idx, 9, pattern_size);
        res += P35 * calc_pop(idx, 1, pattern_size);
        res += P34 * calc_pop(idx, 5, pattern_size);
        res += P33 * calc_pop(idx, 8, pattern_size);
        res += P32 * calc_pop(idx, 2, pattern_size);
        res += P31 * calc_pop(idx, 6, pattern_size);
        res += calc_pop(idx, 3, pattern_size);
    }
    else if (pattern_idx == 11) { // cross
        res += P39 * calc_pop(idx, 0, pattern_size);
        res += P38 * calc_pop(idx, 1, pattern_size);
        res += P37 * calc_pop(idx, 2, pattern_size);
        res += P36 * calc_pop(idx, 3, pattern_size);
        res += P35 * calc_pop(idx, 7, pattern_size);
        res += P34 * calc_pop(idx, 8, pattern_size);
        res += P33 * calc_pop(idx, 9, pattern_size);
        res += P32 * calc_pop(idx, 4, pattern_size);
        res += P31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    }
    else if (pattern_idx == 12) { // narrow triangle
        res += P39 * calc_pop(idx, 0, pattern_size);
        res += P38 * calc_pop(idx, 5, pattern_size);
        res += P37 * calc_pop(idx, 7, pattern_size);
        res += P36 * calc_pop(idx, 8, pattern_size);
        res += P35 * calc_pop(idx, 9, pattern_size);
        res += P34 * calc_pop(idx, 1, pattern_size);
        res += P33 * calc_pop(idx, 6, pattern_size);
        res += P32 * calc_pop(idx, 2, pattern_size);
        res += P31 * calc_pop(idx, 3, pattern_size);
        res += calc_pop(idx, 4, pattern_size);
    }
    else if (pattern_idx == 13) { // kite
        res += P39 * calc_pop(idx, 0, pattern_size);
        res += P38 * calc_pop(idx, 2, pattern_size);
        res += P37 * calc_pop(idx, 1, pattern_size);
        res += P36 * calc_pop(idx, 3, pattern_size);
        res += P35 * calc_pop(idx, 7, pattern_size);
        res += P34 * calc_pop(idx, 8, pattern_size);
        res += P33 * calc_pop(idx, 9, pattern_size);
        res += P32 * calc_pop(idx, 4, pattern_size);
        res += P31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    }
    else if (pattern_idx == 14) { // edge + 2Xa
        int line1 = idx % 8;
        idx /= 8;
        res += P37 * calc_pop(idx, 3, pattern_size);
        res += P36 * calc_pop(idx, 2, pattern_size);
        res += P35 * calc_pop(idx, 1, pattern_size);
        res += P34 * calc_pop(idx, 0, pattern_size);
        res += P33 * calc_pop(idx, 7, pattern_size);
        res += P32 * calc_pop(idx, 6, pattern_size);
        res += P31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 4, pattern_size);
        res *= 8;
        res += line1;
    }
    else if (pattern_idx == 19) { // mobility triangle0
        res |= calc_pop2(idx, 3);
        res |= calc_pop2(idx, 1) << 1;
        res |= calc_pop2(idx, 4) << 2;
        res |= calc_pop2(idx, 0) << 3;
        res |= calc_pop2(idx, 2) << 4;
        res |= calc_pop2(idx, 5) << 5;
        idx >>= 6;
        res |= calc_pop2(idx, 3) << 6;
        res |= calc_pop2(idx, 1) << 7;
        res |= calc_pop2(idx, 4) << 8;
        res |= calc_pop2(idx, 0) << 9;
        res |= calc_pop2(idx, 2) << 10;
        res |= calc_pop2(idx, 5) << 11;
    }
    else if (pattern_idx == 20) { // mobility triangle1
        res |= calc_pop2(idx, 5);
        res |= calc_pop2(idx, 4) << 1;
        res |= calc_pop2(idx, 2) << 2;
        res |= calc_pop2(idx, 3) << 3;
        res |= calc_pop2(idx, 1) << 4;
        res |= calc_pop2(idx, 0) << 5;
        idx >>= 6;
        res |= calc_pop2(idx, 5) << 6;
        res |= calc_pop2(idx, 4) << 7;
        res |= calc_pop2(idx, 2) << 8;
        res |= calc_pop2(idx, 3) << 9;
        res |= calc_pop2(idx, 1) << 10;
        res |= calc_pop2(idx, 0) << 11;
    }
    else if (pattern_idx == 21) { // mobility triangle2
        res |= calc_pop2(idx, 5);
        res |= calc_pop2(idx, 4) << 1;
        res |= calc_pop2(idx, 2) << 2;
        res |= calc_pop2(idx, 3) << 3;
        res |= calc_pop2(idx, 1) << 4;
        res |= calc_pop2(idx, 0) << 5;
        idx >>= 6;
        res |= calc_pop2(idx, 5) << 6;
        res |= calc_pop2(idx, 4) << 7;
        res |= calc_pop2(idx, 2) << 8;
        res |= calc_pop2(idx, 3) << 9;
        res |= calc_pop2(idx, 1) << 10;
        res |= calc_pop2(idx, 0) << 11;
    }
    else if (pattern_idx == 22) { // mobility triangle3
        res |= calc_pop2(idx, 0);
        res |= calc_pop2(idx, 3) << 1;
        res |= calc_pop2(idx, 5) << 2;
        res |= calc_pop2(idx, 1) << 3;
        res |= calc_pop2(idx, 4) << 4;
        res |= calc_pop2(idx, 2) << 5;
        idx >>= 6;
        res |= calc_pop2(idx, 0) << 6;
        res |= calc_pop2(idx, 3) << 7;
        res |= calc_pop2(idx, 5) << 8;
        res |= calc_pop2(idx, 1) << 9;
        res |= calc_pop2(idx, 4) << 10;
        res |= calc_pop2(idx, 2) << 11;
    }
    else if (pattern_idx == 23) { // mobility midedge
        res |= calc_pop2(idx, 5);
        res |= calc_pop2(idx, 4) << 1;
        res |= calc_pop2(idx, 3) << 2;
        res |= calc_pop2(idx, 2) << 3;
        res |= calc_pop2(idx, 1) << 4;
        res |= calc_pop2(idx, 0) << 5;
        idx >>= 6;
        res |= calc_pop2(idx, 5) << 6;
        res |= calc_pop2(idx, 4) << 7;
        res |= calc_pop2(idx, 3) << 8;
        res |= calc_pop2(idx, 2) << 9;
        res |= calc_pop2(idx, 1) << 10;
        res |= calc_pop2(idx, 0) << 11;
    }
    else {
        res = idx;
    }
    return res;
}

inline int round_score(int x) {
    if (x > 0)
        x += STEP_2;
    else if (x < 0)
        x -= STEP_2;
    return x / STEP;
}

inline void scoring_mae() {
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i) {
        score = pre_calc_scores[i];
        avg_score += fabs((double)round_score((int)test_labels[i]) - (double)round_score(score)) / nums;
    }
    cerr << " mae " << avg_score << " "; //"                   ";
    avg_score = 0;
    for (i = 0; i < nums; ++i) {
        score = pre_calc_scores[i];
        avg_score += (fabs(test_labels[i] - (double)score) / STEP) * (fabs(test_labels[i] - (double)score) / STEP) / nums;
    }
    cerr << "mse " << avg_score << "                   ";
}

inline void scoring_mae_cout() {
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i) {
        score = pre_calc_scores[i];
        avg_score += fabs((double)round_score((int)test_labels[i]) - (double)round_score(score)) / nums;
    }
    cout << " mae " << avg_score << " "; //"                   ";
    avg_score = 0;
    for (i = 0; i < nums; ++i) {
        score = pre_calc_scores[i];
        avg_score += (fabs(test_labels[i] - (double)score) / STEP) * (fabs(test_labels[i] - (double)score) / STEP) / nums;
    }
    cout << "mse " << avg_score;
}

inline double scoring_next_step(int pattern, int idx) {
    double score, res = 0.0, err;
    int data_size = nums;
    double raw_error;
    for (const int& i : test_memo[pattern][idx]) {
        score = pre_calc_scores[i];
        raw_error = test_labels[i] - score;
        //if (fabs(raw_error) < (double)STEP_2)
        //    raw_error = 0.0;
        err = raw_error * bias[(int)test_labels[i] / STEP + 64];
        res += err;
    }
    return res;
}


double* pre_calc_scores_kernel = 0;
double* eval_arr_kernel = 0;

cudaError_t calc_score_with_gpu() {
    cudaError_t cudaStatus;
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(eval_arr_kernel, eval_arr, N_EVAL * MAX_EVALUATE_IDX * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMemcpy failed!" << endl;
        return cudaStatus;
    }
    
    // Launch a kernel on the GPU with one thread for each element.
    const int interval = 1024;
    int actual_interval;
    for (int i = 0; i < nums; i += interval) {
        actual_interval = min(interval, nums - i);
        calc_score_kernel << <1, actual_interval >> > (sa_phase, eval_arr_kernel, test_data_kernel + i * N_RAW_PARAMS, pre_calc_scores_kernel + i);
    }
    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        cerr << "calc_score_kernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << endl;
        return cudaStatus;
    }

    //double* pre_calc_scores_arr = (double*)malloc(nums * sizeof(double));
    // Copy output vector from GPU buffer to host memory.
    for (int i = 0; i < nums; i += PRE_CALC_SCORE_INTERVAL) {
        actual_interval = min(PRE_CALC_SCORE_INTERVAL, nums - i);
        cudaStatus = cudaMemcpy(&pre_calc_scores_arr, pre_calc_scores_kernel + i, actual_interval * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            cerr << "cudaMemcpy failed!" << endl;
            return cudaStatus;
        }
        for (int j = 0; j < actual_interval; ++j)
            pre_calc_scores[i + j] = pre_calc_scores_arr[j];
    }

    //for (int i = 0; i < 10; ++i)
    //    cerr << pre_calc_scores[i] << " " << test_labels[i] << endl;

    return cudaStatus;
}

int copy_data_kernel() {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
        cudaFree(test_data_kernel);
        cudaFree(pre_calc_scores_kernel);
        cudaFree(eval_arr_kernel);
        return 1;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&pre_calc_scores_kernel, nums * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        cudaFree(test_data_kernel);
        cudaFree(pre_calc_scores_kernel);
        cudaFree(eval_arr_kernel);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&eval_arr_kernel, N_EVAL * MAX_EVALUATE_IDX * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        cudaFree(test_data_kernel);
        cudaFree(pre_calc_scores_kernel);
        cudaFree(eval_arr_kernel);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&test_data_kernel, nums * N_RAW_PARAMS * sizeof(int*));
    if (cudaStatus != cudaSuccess) {
        cerr << "cudaMalloc failed!" << endl;
        cudaFree(test_data_kernel);
        cudaFree(pre_calc_scores_kernel);
        cudaFree(eval_arr_kernel);
        return 1;
    }


    const int interval = 1024;
    int test_data_arr[N_RAW_PARAMS * interval];
    int actual_interval;
    for (int i = 0; i < nums; i += interval) {
        cerr << "\r" << i * 100 / nums;
        actual_interval = min(interval, nums - i);
        for (int j = 0; j < actual_interval; ++j) {
            for (int k = 0; k < N_RAW_PARAMS; ++k)
                test_data_arr[j * N_RAW_PARAMS + k] = test_data[i + j][k];
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(test_data_kernel + N_RAW_PARAMS * i, test_data_arr, actual_interval * N_RAW_PARAMS * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cerr << "cudaMemcpy failed!" << endl;
            cudaFree(test_data_kernel);
            cudaFree(pre_calc_scores_kernel);
            cudaFree(eval_arr_kernel);
            return 1;
        }
    }
    cerr << endl;

    return 0;
}

void free_kernel() {
    cudaFree(test_data_kernel);
}

inline void next_step() {
    int pattern, rev_idx;
    double err;
    //for (int i = 0; i < nums; ++i)
    //    pre_calc_scores[i] = calc_score(sa_phase, i);
    cudaError_t cudaStatus = calc_score_with_gpu();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }
    for (pattern = 0; pattern < N_EVAL; ++pattern) {
        for (const int& idx : used_idxes_vector[pattern]) {
            if (idx == rev_idxes[pattern][idx]) {
                err = scoring_next_step(pattern, idx);
                eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
            }
            else if (idx < rev_idxes[pattern][idx]) {
                rev_idx = rev_idxes[pattern][idx];
                err = scoring_next_step(pattern, idx) + scoring_next_step(pattern, rev_idx);
                eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
                eval_arr[pattern][rev_idx] += 2.0 * alpha[pattern][idx] * err;
            }
        }
    }
}

void sd(unsigned long long tl) {
    cerr << "alpha " << alpha << endl;
    unsigned long long strt = tim(), now = tim();
    int t = 0;
    for (;;) {
        ++t;
        next_step();
        if ((t & 0b0) == 0) {
            now = tim();
            if (now - strt > tl)
                break;
            cerr << "\r " << t << " " << (int)((double)(now - strt) / tl * 1000);
            scoring_mae();
        }
    }
    cerr << endl;
    cout << t;
    scoring_mae_cout();
    //cout << endl;
}

void init() {
    int i, j;
    pow4[0] = 1;
    for (i = 1; i < 8; ++i)
        pow4[i] = pow4[i - 1] * 4;
    pow3[0] = 1;
    for (i = 1; i < 11; ++i)
        pow3[i] = pow3[i - 1] * 3;
    for (i = 0; i < N_EVAL; ++i) {
        for (j = 0; j < eval_sizes[i]; ++j) {
            //cerr << i << " " << j << endl;
            rev_idxes[i][j] = calc_rev_idx(i, pattern_sizes[i], j);
        }
    }
}

int main(int argc, char* argv[]) {
    //eval_arr = new double*[N_EVAL];
    //for (int i = 0; i < N_EVAL; ++i)
    //    eval_arr[i] = new double[MAX_EVALUATE_IDX];

    sa_phase = 59; // atoi(argv[1]);
    hour = 0; // atoi(argv[2]);
    minute = 1; // atoi(argv[3]);
    second = 0; // atoi(argv[4]);
    beta = 0.01; // atof(argv[5]);
    /*
    sa_phase = atoi(argv[1]);
    hour = atoi(argv[2]);
    minute = atoi(argv[3]);
    second = atoi(argv[4]);
    beta = atof(argv[5]);
    */

    int i, j;
    
    minute += hour * 60;
    second += minute * 60;

    cerr << sa_phase << " " << second << " " << beta << endl;

    board_init();
    init();
    initialize_param();
    cerr << "initialized" << endl;
    //output_param_onephase();
    //input_param_onephase((string)(argv[6]));
    Adj_info info = input_test_data(argc, argv);

    if (copy_data_kernel())
        return 1;

    sd(second * 1000);
    free_kernel();
    cout << " n_data " << info.n_data << " n_param " << info.n_params << " n_used_params " << info.use_params << endl;

    output_param_onephase();

    return 0;
}
/*
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

*/