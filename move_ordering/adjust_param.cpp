#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>

using namespace std;

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

#define n_patterns 10
#define max_evaluate_idx 65536 //59049

#define step 4096
#define sc_w 4096

#define n_data 5000000

#define n_raw_params 38

#define beta 0.003
unsigned long long hour = 0;
unsigned long long minute = 15;
unsigned long long second = 0;

double alpha[n_patterns][max_evaluate_idx];

const int pattern_sizes[n_patterns] = {8, 8, 8, 8, 3, 4, 5, 6, 7, 8};
const int eval_sizes[n_patterns] = {65536, 65536, 65536, 65536, 64, 256, 1024, 4096, 16384, 65536};
double eval_arr[n_patterns][max_evaluate_idx];
int test_data[n_data][n_raw_params];
double test_labels[n_data];
int nums;
double scores;
vector<int> test_memo[n_patterns][max_evaluate_idx];
vector<double> test_scores, pre_calc_scores;
unordered_set<int> used_idxes[n_patterns];
vector<int> used_idxes_vector[n_patterns];
int rev_idxes[n_patterns][max_evaluate_idx];
int pow4[8];

void initialize_param(){
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem)
            eval_arr[pattern_idx][pattern_elem] = 0;
    }
    cerr << "initialized" << endl;
}

void input_param(string file){
    ifstream ifs(file);
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t =0;
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
        cerr << "=";
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            ++t;
            getline(ifs, line);
            eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
    cerr << t << endl;
}

void input_test_data(int strt){
    int i, j, k;
    ifstream ifs("big_data.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int phase, player, score;
    int t = 0, u = 0;
    nums = 0;
    const int pattern_nums[n_raw_params] = {
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6, 
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9
    };
    for (j = 0; j < n_patterns; ++j){
        used_idxes[j].clear();
        for (k = 0; k < max_evaluate_idx; ++k)
            test_memo[j][k].clear();
    }
    test_scores.clear();
    for (i = 0; i < strt; ++i)
        getline(ifs, line);
    while (getline(ifs, line) && t < n_data){
        ++t;
        if ((t & 0b1111111111111111) == 0b1111111111111111)
            cerr << '\r' << t;
        istringstream iss(line);
        ++u;
        for (i = 0; i < n_raw_params; ++i)
            iss >> test_data[nums][i];
        iss >> score;
        for (i = 0; i < n_raw_params; ++i)
            used_idxes[pattern_nums[i]].emplace(test_data[nums][i]);
        test_labels[nums] = score * step;
        for (i = 0; i < n_raw_params; ++i)
            test_memo[pattern_nums[i]][test_data[nums][i]].push_back(nums);
        test_scores.push_back(0);
        pre_calc_scores.push_back(0);
        ++nums;
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
    for (i = 0; i < n_patterns; ++i){
        for (auto elem: used_idxes[i])
            used_idxes_vector[i].push_back(elem);
    }

    cerr << "n_data " << u << endl;

    u = 0;
    for (i = 0; i < n_patterns; ++i)
        u += eval_sizes[i];
    cerr << "n_all_param " << u << endl;
    u = 0;
    for (i = 0; i < n_patterns; ++i){
        u += (int)used_idxes[i].size();
    }
    cerr << "used_param " << u << endl;
    
    for (i = 0; i < n_patterns; ++i){
        for (const int &used_idx: used_idxes_vector[i])
            alpha[i][used_idx] = beta / max(50, (int)test_memo[i][used_idx].size());
    }

    ifs.close();
}

void output_param(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            cout << round(eval_arr[pattern_idx][pattern_elem]) << endl;
        }
    }
    cerr << endl;
}

inline double loss(double x, int siz){
    return (double)x / (double)siz * (double)x;
}

inline double calc_score(int i){
    int res = 
        eval_arr[0][test_data[i][0]] + 
        eval_arr[0][test_data[i][1]] + 
        eval_arr[0][test_data[i][2]] + 
        eval_arr[0][test_data[i][3]] + 
        eval_arr[1][test_data[i][4]] + 
        eval_arr[1][test_data[i][5]] + 
        eval_arr[1][test_data[i][6]] + 
        eval_arr[1][test_data[i][7]] + 
        eval_arr[2][test_data[i][8]] + 
        eval_arr[2][test_data[i][9]] + 
        eval_arr[2][test_data[i][10]] + 
        eval_arr[2][test_data[i][11]] + 
        eval_arr[3][test_data[i][12]] + 
        eval_arr[3][test_data[i][13]] + 
        eval_arr[3][test_data[i][14]] + 
        eval_arr[3][test_data[i][15]] + 
        eval_arr[4][test_data[i][16]] + 
        eval_arr[4][test_data[i][17]] + 
        eval_arr[4][test_data[i][18]] + 
        eval_arr[4][test_data[i][19]] + 
        eval_arr[5][test_data[i][20]] + 
        eval_arr[5][test_data[i][21]] + 
        eval_arr[5][test_data[i][22]] + 
        eval_arr[5][test_data[i][23]] + 
        eval_arr[6][test_data[i][24]] + 
        eval_arr[6][test_data[i][25]] + 
        eval_arr[6][test_data[i][26]] + 
        eval_arr[6][test_data[i][27]] + 
        eval_arr[7][test_data[i][28]] + 
        eval_arr[7][test_data[i][29]] + 
        eval_arr[7][test_data[i][30]] + 
        eval_arr[7][test_data[i][31]] + 
        eval_arr[8][test_data[i][32]] + 
        eval_arr[8][test_data[i][33]] + 
        eval_arr[8][test_data[i][34]] + 
        eval_arr[8][test_data[i][35]] + 
        eval_arr[9][test_data[i][36]] + 
        eval_arr[9][test_data[i][37]];
    return max(-sc_w, min(sc_w, res));
}

inline int calc_pop4(int a, int b, int s){
    return (a / pow4[s - 1 - b]) % 4;
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    int t;
    if (pattern_idx <= 3 || pattern_idx == 9){
        t = 8;
    } else if (pattern_idx == 4){
        t = 3;
    } else if (pattern_idx == 5){
        t = 4;
    } else if (pattern_idx == 6){
        t = 5;
    } else if (pattern_idx == 7){
        t = 6;
    } else if (pattern_idx == 8){
        t = 7;
    }
    for (int i = 0; i < t; ++i){
        res += calc_pop4(idx, i, t) * pow4[i];
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
        err = test_labels[i] - score;
        res += err;
    }
    return res;
}

inline void next_step(){
    int pattern, rev_idx;
    double err;
    for (int i = 0; i < nums; ++i)
        pre_calc_scores[i] = calc_score(i);
    for (pattern = 0; pattern < n_patterns; ++pattern){
        for (const int &idx: used_idxes_vector[pattern]){
            rev_idx = rev_idxes[pattern][idx];
            if (idx < rev_idx){
                err = scoring_next_step(pattern, idx) + scoring_next_step(pattern, rev_idx);
                eval_arr[pattern][idx] += 2.0 * alpha[pattern][idx] * err;
                eval_arr[pattern][rev_idx] += 2.0 * alpha[pattern][idx] * err;
            } else if (idx == rev_idx){
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
    for (i = 0; i < n_patterns; ++i){
        for (j = 0; j < eval_sizes[i]; ++j)
            rev_idxes[i][j] = calc_rev_idx(i, pattern_sizes[i], j);
    }
}

int main(int argc, char *argv[]){
    int i, j;

    minute += hour * 60;
    second += minute * 60;

    init();
    initialize_param();
    //output_param();
    input_param((string)(argv[1]));
    input_test_data(0);

    sd(second * 1000);

    output_param();

    return 0;
}