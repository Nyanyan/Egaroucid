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

#define n_phases 30
#define phase_n_stones 2
#define n_patterns 11
#define n_eval 21
#define max_evaluate_idx 65536

#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65

int sa_phase, sa_player;

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

#define p41 4
#define p42 16
#define p43 64
#define p44 256
#define p45 1024
#define p46 4096
#define p47 16384
#define p48 65536

#define step 256
#define sc_w (step * HW2)

#define n_data 10000000

#define n_raw_params 70

double beta = 0.001;
unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

double alpha[n_eval][max_evaluate_idx];

constexpr int pattern_sizes[n_eval] = {
    8, 8, 8, 5, 6, 7, 8, 10, 10, 9, 10, 
    0, 0, 0, 0, 
    8, 8, 8, 8, 
    -1
};
constexpr int eval_sizes[n_eval] = {
    p38, p38, p38, p35, p36, p37, p38, p310, p310, p39, p310, 
    max_surround * max_surround, max_canput * max_canput, max_stability * max_stability, max_stone_num * max_stone_num, 
    p48, p48, p48, p48, p48, 
    p34 * p44
};
constexpr int pattern_nums[n_raw_params] = {
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

    11, 12, 13, 14, 

    15, 15, 15, 15, 
    16, 16, 16, 16, 
    17, 17, 17, 17, 
    18, 18, 18, 18, 
    19, 19, 19, 19,

    20, 20, 20, 20
};
double eval_arr[n_eval][max_evaluate_idx];
//double **eval_arr;
vector<vector<int>> test_data;
double test_labels[n_data];
int nums;
double scores;
vector<int> test_memo[n_eval][max_evaluate_idx];
vector<double> test_scores, pre_calc_scores;
unordered_set<int> used_idxes[n_eval];
vector<int> used_idxes_vector[n_eval];
int rev_idxes[n_eval][max_evaluate_idx];
int pow4[8];
int pow3[11];
int n_data_score[129];
int n_data_idx[n_eval][max_evaluate_idx][129];
double bias[129];

void initialize_param(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
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
    for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
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
    for (j = 0; j < n_eval; ++j){
        used_idxes[j].clear();
        for (k = 0; k < max_evaluate_idx; ++k)
            test_memo[j][k].clear();
    }
    test_scores.clear();
    for (i = 0; i < n_eval; ++i){
        for (j = 0; j < max_evaluate_idx; ++j){
            for (k = 0; k < 129; ++k)
                n_data_idx[i][j][k] = 0;
        }
    }
    for(i = 0; i < 129; ++i)
        n_data_score[i] = 0;
    int sur, canput, stab, num;
    FILE* fp;
    int file_idxes[n_raw_params];
    for (int file_idx = 7; file_idx < argc; ++file_idx){
        cerr << argv[file_idx] << endl;
        if (fopen_s(&fp, argv[file_idx], "rb") != 0) {
            cerr << "can't open " << argv[file_idx] << endl;
            continue;
        }
        while (t < n_data){
            ++t;
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                cerr << '\r' << t;
            if (fread(&phase, 4, 1, fp) < 1)
                break;
            phase = (phase - 4) / phase_n_stones;
            fread(&player, 4, 1, fp);
            fread(file_idxes, 4, n_raw_params, fp);
            fread(&score, 4, 1, fp);
            if (phase == sa_phase){
                vector<int> file_idxes_vector;
                for (int i = 0; i < n_raw_params; ++i)
                    file_idxes_vector.emplace_back(file_idxes[i]);
                test_data.emplace_back(file_idxes_vector);
                ++u;
                for (i = 0; i < n_raw_params; ++i)
                    used_idxes[pattern_nums[i]].emplace(test_data[nums][i]);
                test_labels[nums] = score * step;
                for (i = 0; i < n_raw_params; ++i)
                    test_memo[pattern_nums[i]][test_data[nums][i]].push_back(nums);
                test_scores.push_back(0);
                pre_calc_scores.push_back(0);
                ++n_data_score[score + 64];
                for (i = 0; i < n_raw_params; ++i)
                    ++n_data_idx[pattern_nums[i]][test_data[nums][i]][score + 64];
                ++nums;
            }
        }
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
    for (i = 0; i < n_eval; ++i){
        for (auto elem: used_idxes[i])
            used_idxes_vector[i].push_back(elem);
    }

    cerr << "n_data " << u << endl;

    u = 0;
    for (i = 0; i < n_eval; ++i)
        u += eval_sizes[i];
    cerr << "n_all_param " << u << endl;
    u = 0;
    for (i = 0; i < n_eval; ++i){
        u += (int)used_idxes[i].size();
    }
    cerr << "used_param " << u << endl;

    int zero_score_n_data = n_data_score[64];
    int wipeout_n_data = zero_score_n_data / 2;
    int modified_n_data;
    for (int i = 0; i < 129; ++i){
        if (n_data_score[i] == 0)
            continue;
        if (i <= 64)
            modified_n_data = wipeout_n_data + (zero_score_n_data - wipeout_n_data) * i / 64;
        else
            modified_n_data = zero_score_n_data - (zero_score_n_data - wipeout_n_data) * (i - 64) / 64;
        bias[i] = 1.0; //(double)modified_n_data / n_data_score[i];
        //cerr << modified_n_data << " " << bias[i] << endl;
    }

    double n_weighted_data;
    for (i = 0; i < n_eval; ++i){
        for (const int &used_idx: used_idxes_vector[i]){
            n_weighted_data = 0.0;
            for (j = 0; j < 129; ++j)
                n_weighted_data += bias[j] * (double)n_data_idx[i][used_idx][j];
            alpha[i][used_idx] = beta / max(50.0, n_weighted_data);
        }
    }
    
}

void output_param_onephase(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
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
    for (int j = 0; j < n_raw_params; ++j)
        res += eval_arr[pattern_nums[j]][test_data[i][j]];
    return max(-sc_w, min(sc_w, res));
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
    } else if (pattern_idx == 9){ // corner9
        res += p38 * calc_pop(idx, 0, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 6, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 4, pattern_size);
        res += p33 * calc_pop(idx, 7, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 8, pattern_size);
    } else if (pattern_idx == 10){ // narrow triangle
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 5, pattern_size);
        res += p37 * calc_pop(idx, 7, pattern_size);
        res += p36 * calc_pop(idx, 8, pattern_size);
        res += p35 * calc_pop(idx, 9, pattern_size);
        res += p34 * calc_pop(idx, 1, pattern_size);
        res += p33 * calc_pop(idx, 6, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 3, pattern_size);
        res += calc_pop(idx, 4, pattern_size);
    } else if (n_patterns <= pattern_idx && pattern_idx < n_patterns + 4){
        res = idx;
    } else if (n_patterns + 4 <= pattern_idx && pattern_idx < n_patterns + 4 + 5){
        for (int i = 0; i < 8; ++i){
            res |= (1 & (idx >> i)) << (HW_M1 - i);
            res |= (1 & (idx >> (HW + i))) << (HW + HW_M1 - i);
        }
    } else if (pattern_idx == 20){
        int stone_idx = idx / 256;
        int mobility_idx = idx % 256;
        for (int i = 0; i < 4; ++i)
            res += pow3[i] * calc_pop(stone_idx, i, 4);
        res *= 256;
        for (int i = 0; i < 4; ++i){
            res |= (1 & (mobility_idx >> i)) << (3 - i);
            res |= (1 & (mobility_idx >> (4 + i))) << (7 - i);
        }
    }
    return res;
}

inline void scoring_mae(){
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = pre_calc_scores[i];
        avg_score += fabs(test_labels[i] - (double)score) / nums;
    }
    cerr << " " << avg_score << " " << avg_score / step << "                   ";
}

inline double scoring_next_step(int pattern, int idx){
    double score, res = 0.0, err;
    int data_size = nums;
    for (const int &i: test_memo[pattern][idx]){
        score = pre_calc_scores[i];
        err = (test_labels[i] - score) * bias[(int)test_labels[i] / step + 64];
        res += err;
    }
    return res;
}

inline void next_step(){
    int pattern, rev_idx;
    double err;
    for (int i = 0; i < nums; ++i)
        pre_calc_scores[i] = calc_score(sa_phase, i);
    for (pattern = 0; pattern < n_eval; ++pattern){
        for (const int &idx: used_idxes_vector[pattern]){
            if (pattern < n_patterns || n_patterns + 4 <= pattern){
                rev_idx = rev_idxes[pattern][idx];
                if (idx < rev_idx){
                    err = scoring_next_step(pattern, idx) + scoring_next_step(pattern, rev_idx);
                    eval_arr[pattern][idx] += (alpha[pattern][idx] + alpha[pattern][rev_idx]) * err;
                    eval_arr[pattern][rev_idx] += (alpha[pattern][idx] + alpha[pattern][rev_idx]) * err;
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
    cerr << t;
    scoring_mae();
    cerr << endl;
}

void init(){
    int i, j;
    pow4[0] = 1;
    for (i = 1; i < 8; ++i)
        pow4[i] = pow4[i - 1] * 4;
    pow3[0] = 1;
    for (i = 1; i < 11; ++i)
        pow3[i] = pow3[i - 1] * 3;
    for (i = 0; i < n_eval; ++i){
        for (j = 0; j < eval_sizes[i]; ++j){
            //cerr << i << " " << j << endl;
            rev_idxes[i][j] = calc_rev_idx(i, pattern_sizes[i], j);
        }
    }
}

int main(int argc, char *argv[]){
    //eval_arr = new double*[n_eval];
    //for (int i = 0; i < n_eval; ++i)
    //    eval_arr[i] = new double[max_evaluate_idx];

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
    //output_param_onephase();
    //input_param_onephase((string)(argv[6]));
    input_test_data(argc, argv);

    sd(second * 1000);

    output_param_onephase();

    return 0;
}