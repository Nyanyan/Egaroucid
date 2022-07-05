#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>
#include "new_util/board.hpp"

using namespace std;

#define N_PATTERNS 18
#define N_EVAL (N_PATTERNS + 3)
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 70
#endif
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STONE_NUM 65
#define MAX_EVALUATE_IDX 59049

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

#define STEP 256
#define STEP_2 128
#define SC_W (STEP * HW2)

#define N_DATA 50000000

#define N_RAW_PARAMS 73

double beta = 0.001;
unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

double alpha[N_EVAL][MAX_EVALUATE_IDX];

constexpr int pattern_sizes[N_EVAL] = {
    8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 10, 10, 
    8, 6, 6, 6, 4,
    -1, -1, -1
};
constexpr int eval_sizes[N_EVAL] = {
    P38, P38, P38, P35, P36, P37, P38, P310, P310, P310, P310, P310, P310, 
    52488, 46656, 46656, 46656, 41472, 
    MAX_SURROUND * MAX_SURROUND, MAX_CANPUT * MAX_CANPUT, MAX_STONE_NUM * MAX_STONE_NUM
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
    15, 15, 15, 15, 
    16, 16, 16, 16, 
    17, 17, 17, 17, 

    18, 19, 20
};

double eval_arr[N_EVAL][MAX_EVALUATE_IDX];
vector<vector<int>> test_data;
double test_labels[N_DATA];
int nums;
double scores;
vector<int> test_memo[N_EVAL][MAX_EVALUATE_IDX];
vector<double> test_scores, pre_calc_scores;
unordered_set<int> used_idxes[N_EVAL];
vector<int> used_idxes_vector[N_EVAL];
int rev_idxes[N_EVAL][MAX_EVALUATE_IDX];
int pow4[8];
int pow3[11];
int N_DATA_score[129];
int N_DATA_idx[N_EVAL][MAX_EVALUATE_IDX][129];
double bias[129];

void initialize_param(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem)
            eval_arr[pattern_idx][pattern_elem] = 0;
    }
}

void input_param_onephase(string file){
    ifstream ifs(file);
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t =0;
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx){
        cerr << "=";
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            ++t;
            getline(ifs, line);
            eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
    cerr << t << endl;
}

inline double calc_score(int phase, int i);

void input_test_data(int argc, char *argv[]){
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
    for (j = 0; j < N_EVAL; ++j){
        used_idxes[j].clear();
        for (k = 0; k < MAX_EVALUATE_IDX; ++k)
            test_memo[j][k].clear();
    }
    test_scores.clear();
    for (i = 0; i < N_EVAL; ++i){
        for (j = 0; j < MAX_EVALUATE_IDX; ++j){
            for (k = 0; k < 129; ++k)
                N_DATA_idx[i][j][k] = 0;
        }
    }
    for(i = 0; i < 129; ++i)
        N_DATA_score[i] = 0;
    int sur, canput, stab, num;
    FILE* fp;
    int file_idxes[N_RAW_PARAMS];
    for (int file_idx = 7; file_idx < argc; ++file_idx){
        cerr << argv[file_idx] << endl;
        if (fopen_s(&fp, argv[file_idx], "rb") != 0) {
            cerr << "can't open " << argv[file_idx] << endl;
            continue;
        }
        while (t < N_DATA){
            ++t;
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                cerr << '\r' << t;
            if (fread(&phase, 4, 1, fp) < 1)
                break;
            phase = (phase - 4) / PHASE_N_STONES;
            fread(&player, 4, 1, fp);
            fread(file_idxes, 4, N_RAW_PARAMS, fp);
            fread(&score, 4, 1, fp);
            if (phase == sa_phase){
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
                ++N_DATA_score[score + 64];
                for (i = 0; i < N_RAW_PARAMS; ++i)
                    ++N_DATA_idx[pattern_nums[i]][test_data[nums][i]][score + 64];
                ++nums;
            }
        }
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
    for (i = 0; i < N_EVAL; ++i){
        for (auto elem: used_idxes[i])
            used_idxes_vector[i].push_back(elem);
    }

    cerr << "N_DATA " << u << endl;

    u = 0;
    for (i = 0; i < N_EVAL; ++i)
        u += eval_sizes[i];
    cerr << "n_all_param " << u << endl;
    u = 0;
    for (i = 0; i < N_EVAL; ++i){
        u += (int)used_idxes[i].size();
    }
    cerr << "used_param " << u << endl;

    int zero_score_N_DATA = N_DATA_score[64];
    int wipeout_N_DATA = zero_score_N_DATA / 2;
    int modified_N_DATA;
    for (int i = 0; i < 129; ++i){
        if (N_DATA_score[i] == 0)
            continue;
        if (i <= 64)
            modified_N_DATA = wipeout_N_DATA + (zero_score_N_DATA - wipeout_N_DATA) * i / 64;
        else
            modified_N_DATA = zero_score_N_DATA - (zero_score_N_DATA - wipeout_N_DATA) * (i - 64) / 64;
        bias[i] = 1.0; //(double)modified_N_DATA / N_DATA_score[i];
        //cerr << modified_N_DATA << " " << bias[i] << endl;
    }

    double n_weighted_data;
    for (i = 0; i < N_EVAL; ++i){
        for (const int &used_idx: used_idxes_vector[i]){
            n_weighted_data = 0.0;
            for (j = 0; j < 129; ++j)
                n_weighted_data += bias[j] * (double)N_DATA_idx[i][used_idx][j];
            if (used_idx < N_PATTERNS || N_PATTERNS + 4 <= used_idx){
                for (j = 0; j < 129; ++j)
                    n_weighted_data += bias[j] * (double)N_DATA_idx[i][rev_idxes[i][used_idx]][j];
            }
            alpha[i][used_idx] = beta / max(100.0, n_weighted_data);
        }
    }
    
}

void output_param_onephase(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            cout << round(eval_arr[pattern_idx][pattern_elem]) << endl;
        }
    }
    cerr << endl;
}

inline double loss(double x, int siz){
    //double sq_size = sqrt((double)siz);
    //double tmp = (double)x / sq_size;
    return (double)x / (double)siz * (double)x;
}

inline double calc_score(int phase, int i){
    int res = 0;
    for (int j = 0; j < N_RAW_PARAMS; ++j)
        res += eval_arr[pattern_nums[j]][test_data[i][j]];
    return max(-SC_W, min(SC_W, res));
}

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_pop4(int a, int b, int s){
    return (a / pow4[s - 1 - b]) % 4;
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    if (pattern_idx <= 7){
        for (int i = 0; i < pattern_size; ++i)
            res += pow3[i] * calc_pop(idx, i, pattern_size);
    } else if (pattern_idx == 8){ //edge block
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
    } else if (pattern_idx == 9){ // corner10
        res += P39 * calc_pop(idx, 0, pattern_size);
        res += P38 * calc_pop(idx, 3, pattern_size);
        res += P37 * calc_pop(idx, 6, pattern_size);
        res += P36 * calc_pop(idx, 1, pattern_size);
        res += P35 * calc_pop(idx, 4, pattern_size);
        res += P34 * calc_pop(idx, 7, pattern_size);
        res += P33 * calc_pop(idx, 2, pattern_size);
        res += P32 * calc_pop(idx, 5, pattern_size);
        res += P31 * calc_pop(idx, 8, pattern_size);
        res += calc_pop(idx, 9, pattern_size);
    } else if (pattern_idx == 10){ // cross
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
    } else if (pattern_idx == 11){ // kite
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
    } else if (pattern_idx == 12){ // triangle
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
    } else if (pattern_idx == 13){ // edge + 2Xa
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
    } else if (pattern_idx == 14){ // 2edge + X
        int line1 = idx % 8;
        idx /= 8;
        int line2 = idx % 8;
        idx /= 8;
        res += P35 * calc_pop(idx, 5, pattern_size);
        res += P34 * calc_pop(idx, 4, pattern_size);
        res += P33 * calc_pop(idx, 2, pattern_size);
        res += P32 * calc_pop(idx, 3, pattern_size);
        res += P31 * calc_pop(idx, 1, pattern_size);
        res += calc_pop(idx, 0, pattern_size);
        res *= 64;
        res += line1 * 8 + line2;
    } else if (pattern_idx == 15){ // edge + midedge
        int line1 = idx % 8;
        idx /= 8;
        int line2 = idx % 8;
        idx /= 8;
        res += P35 * calc_pop(idx, 5, pattern_size);
        res += P34 * calc_pop(idx, 4, pattern_size);
        res += P33 * calc_pop(idx, 3, pattern_size);
        res += P32 * calc_pop(idx, 2, pattern_size);
        res += P31 * calc_pop(idx, 1, pattern_size);
        res += calc_pop(idx, 0, pattern_size);
        res *= 64;
        res += line2 * 8 + line1;
    } else if (pattern_idx == 16){ // 2edge + corner
        int line1 = idx % 8;
        idx /= 8;
        int line2 = idx % 8;
        idx /= 8;
        res += P35 * calc_pop(idx, 0, pattern_size);
        res += P34 * calc_pop(idx, 2, pattern_size);
        res += P33 * calc_pop(idx, 1, pattern_size);
        res += P32 * calc_pop(idx, 3, pattern_size);
        res += P31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 4, pattern_size);
        res *= 64;
        res += line1 * 8 + line2;
    } else if (pattern_idx == 17){ // corner + 3line
        int line1 = idx % 8;
        idx /= 8;
        int line2 = idx % 8;
        idx /= 8;
        int line3 = idx % 8;
        idx /= 8;
        res += P33 * calc_pop(idx, 0, pattern_size);
        res += P32 * calc_pop(idx, 2, pattern_size);
        res += P31 * calc_pop(idx, 1, pattern_size);
        res += calc_pop(idx, 3, pattern_size);
        res *= 512;
        res += line3 * 64 + line2 * 8 + line1;
    } else{
        res = idx;
    }
    return res;
}

inline int round_score(int x){
    if (x > 0)
        x += STEP_2;
    else if (x < 0)
        x -= STEP_2;
    return x / STEP;
}

inline void scoring_mae(){
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = pre_calc_scores[i];
        avg_score += fabs((double)round_score((int)test_labels[i]) - (double)round_score(score)) / nums;
    }
    cerr << " mae " << avg_score << " "; //"                   ";
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = pre_calc_scores[i];
        avg_score += (fabs(test_labels[i] - (double)score) / STEP) * (fabs(test_labels[i] - (double)score) / STEP) / nums;
    }
    cerr << "mse " << avg_score << "                   ";
}

inline void scoring_mae_cout(){
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = pre_calc_scores[i];
        avg_score += fabs((double)round_score((int)test_labels[i]) - (double)round_score(score)) / nums;
    }
    cout << " mae " << avg_score << " "; //"                   ";
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = pre_calc_scores[i];
        avg_score += (fabs(test_labels[i] - (double)score) / STEP) * (fabs(test_labels[i] - (double)score) / STEP) / nums;
    }
    cout << "mse " << avg_score;
}

inline double scoring_next_step(int pattern, int idx){
    double score, res = 0.0, err;
    int data_size = nums;
    double raw_error;
    for (const int &i: test_memo[pattern][idx]){
        score = pre_calc_scores[i];
        raw_error = test_labels[i] - score;
        if (fabs(raw_error) < (double)STEP_2)
            raw_error = 0.0;
        err = raw_error * bias[(int)test_labels[i] / STEP + 64];
        res += err;
    }
    return res;
}

inline void next_step(){
    int pattern, rev_idx;
    double err;
    for (int i = 0; i < nums; ++i)
        pre_calc_scores[i] = calc_score(sa_phase, i);
    for (pattern = 0; pattern < N_EVAL; ++pattern){
        for (const int &idx: used_idxes_vector[pattern]){
            if (pattern < N_PATTERNS || N_PATTERNS + 4 <= pattern){
                rev_idx = rev_idxes[pattern][idx];
                if (idx < rev_idx){
                    err = scoring_next_step(pattern, idx) + scoring_next_step(pattern, rev_idx);
                    eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
                    eval_arr[pattern][rev_idx] += 2.0 * alpha[pattern][idx] * err;
                } else if (idx == rev_idx){
                    err = scoring_next_step(pattern, idx);
                    eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
                }
            } else{
                err = scoring_next_step(pattern, idx);
                eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
            }
        }
    }
}

void sd(unsigned long long tl){
    cerr << "alpha " << alpha << endl;
    unsigned long long strt = tim(), now = tim();
    int t = 0;
    for (;;){
        ++t;
        next_step();
        if ((t & 0b0) == 0){
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
    cout << endl;
}

void init(){
    int i, j;
    pow4[0] = 1;
    for (i = 1; i < 8; ++i)
        pow4[i] = pow4[i - 1] * 4;
    pow3[0] = 1;
    for (i = 1; i < 11; ++i)
        pow3[i] = pow3[i - 1] * 3;
    for (i = 0; i < N_EVAL; ++i){
        for (j = 0; j < eval_sizes[i]; ++j){
            //cerr << i << " " << j << endl;
            rev_idxes[i][j] = calc_rev_idx(i, pattern_sizes[i], j);
        }
    }
}

int main(int argc, char *argv[]){
    //eval_arr = new double*[N_EVAL];
    //for (int i = 0; i < N_EVAL; ++i)
    //    eval_arr[i] = new double[MAX_EVALUATE_IDX];

    sa_phase = atoi(argv[1]);
    hour = atoi(argv[2]);
    minute = atoi(argv[3]);
    second = atoi(argv[4]);
    beta = atof(argv[5]);
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
    input_test_data(argc, argv);

    sd(second * 1000);

    output_param_onephase();

    return 0;
}