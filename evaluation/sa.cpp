#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <unordered_set>
#include "board.hpp"

using namespace std;

#define n_phases 15
#define phase_n_stones 4
#define n_patterns 13
#define max_surround 80
#define max_evaluate_idx 59049

int sa_phase;

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

#define sc_w 6400
#define step 100

#define n_data 20000000

const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10};
const int eval_sizes[n_patterns + 1] = {p38, p38, p38, p35, p36, p37, p38, p310, p310, p310, p310, p39, p310, max_surround * max_surround};
int eval_arr[n_phases][n_patterns + 1][max_evaluate_idx];
int test_data[n_data / 6 + 10000][52];
double test_labels[n_data / 6 + 10000];
int nums;
double scores;
vector<int> test_memo[n_patterns + 1][max_evaluate_idx];
vector<double> test_scores;
unordered_set<int> used_idxes[n_patterns + 1];
vector<int> used_idxes_vector[n_patterns + 1];


inline unsigned long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int xorx=123456789, xory=362436069, xorz=521288629, xorw=88675123;
inline double myrandom(){
    int t = (xorx^(xorx<<11));
    xorx = xory;
    xory = xorz;
    xorz = xorw;
    xorw = (xorw^(xorw>>19))^(t^(t>>8));
    return (double)(xorw) / 2147483648.0;
}

inline int myrandrange(int s, int e){
    return s + (int)(myrandom() * (e - s));
}

#define time_num_step 10000
#define time_step_width (1.0 / time_num_step)
#define prob_num_step 1000000
#define prob_step_width (100.0 / prob_num_step)
double start_temp = 10.0;
double end_temp = 0.01;
double temperature_arr[time_num_step];
double prob_arr[prob_num_step];

double temperature_x(double x){
    return pow(start_temp, 1 - x) * pow(end_temp, x);
    //return start_temp + (end_temp - start_temp) * x;
}

void anneal_init(){
    double x;
    for (int idx = 0; idx < time_num_step; idx++){
        x = time_step_width * idx;
        temperature_arr[idx] = temperature_x(x);
    }
    for (int idx = 0; idx < prob_num_step; idx++){
        x = -prob_step_width * idx;
        prob_arr[idx] = exp(x);
    }
}

double calc_temperature(double strt, double now, double tl){
    return temperature_x((now - strt) / tl);
}

double prob(double p_score, double n_score, double strt, double now, double tl){
    double dis = p_score - n_score;
    if (dis >= 0)
        return 1.0;
    return exp(dis / calc_temperature(strt, now, tl));
}

double prob_fast(double p_score, double n_score, double strt, double now, double tl){
    double dis = p_score - n_score;
    if (dis >= 0)
        return 1.0;
    int temperature_idx = (int)((now - strt) / tl / time_step_width);
    int prob_idx = (int)min((double)(prob_num_step - 1), -dis / temperature_arr[temperature_idx] / prob_step_width);
    return prob_arr[prob_idx];
}

void input_param(){
    ifstream ifs("f_param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t =0;
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        cerr << "=";
        for (pattern_idx = 0; pattern_idx < n_patterns + 1; ++pattern_idx){
            for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
                ++t;
                getline(ifs, line);
                eval_arr[phase_idx][pattern_idx][pattern_elem] = stoi(line);
            }
        }
    }
    cerr << t << endl;
}

void input_param_onephase(){
    ifstream ifs("f_param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t =0;
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < n_patterns + 1; ++pattern_idx){
        cerr << "=";
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            ++t;
            getline(ifs, line);
            eval_arr[sa_phase][pattern_idx][pattern_elem] = stoi(line);
        }
    }
    cerr << t << endl;
}

inline int calc_add_idx(int arr[]){
    //return arr[42] * max_surround * max_surround + arr[43] * max_surround + arr[44];
    //return arr[42] / 2 * max_surround_2 * max_surround_2 + arr[43] / 2 * max_surround_2 + arr[44] / 2;
    //return arr[42] * max_surround_2 * max_surround_2 + arr[43] / 2 * max_surround_2 + arr[44] / 2;
    return arr[50] * max_surround + arr[51];
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
    const int pattern_nums[54] = {
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
        13, 13, 13, 13
    };
    for (j = 0; j < n_patterns + 1; ++j){
        used_idxes[j].clear();
        for (k = 0; k < max_evaluate_idx; ++k)
            test_memo[j][k].clear();
    }
    test_scores.clear();
    for (i = 0; i < strt; ++i)
        getline(ifs, line);
    while (getline(ifs, line) && t < n_data){
        ++t;
        if ((t & 0b1111111111111) == 0b1111111111111)
            cerr << '\r' << t;
        istringstream iss(line);
        iss >> phase;
        if (phase == sa_phase){
            ++u;
            iss >> player;
            for (i = 0; i < 52; ++i)
                iss >> test_data[nums][i];
            iss >> score;
            for (i = 0; i < 50; ++i)
                used_idxes[pattern_nums[i]].emplace(test_data[nums][i]);
            used_idxes[13].emplace(calc_add_idx(test_data[nums]));
            test_labels[nums] = score * step;
            for (i = 0; i < 50; ++i)
                test_memo[pattern_nums[i]][test_data[nums][i]].push_back(nums);
            test_memo[13][calc_add_idx(test_data[nums])].push_back(nums);
            test_scores.push_back(0);
            ++nums;
        }
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
    for (i = 0; i < n_patterns + 1; ++i){
        for (auto elem: used_idxes[i])
            used_idxes_vector[i].push_back(elem);
    }

    cerr << "n_data " << u << endl;

    u = 0;
    for (i = 0; i < n_patterns + 1; ++i)
        u += eval_sizes[i];
    cerr << "n_all_param " << u << endl;
    u = 0;
    for (i = 0; i < n_patterns + 1; ++i){
        u += (int)used_idxes[i].size();
    }
    cerr << "used_param " << u << endl;
    
    ifs.close();
}

void output_param(){
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        cerr << "=";
        for (pattern_idx = 0; pattern_idx < n_patterns + 1; ++pattern_idx){
            for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
                cout << eval_arr[phase_idx][pattern_idx][pattern_elem] << endl;
            }
        }
    }
    cerr << endl;
}

void output_param_onephase(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < n_patterns + 1; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            cout << eval_arr[sa_phase][pattern_idx][pattern_elem] << endl;
        }
    }
    cerr << endl;
}

inline double loss(int x, int siz){
    //double sq_size = sqrt((double)siz);
    //double tmp = (double)x / sq_size;
    return (double)x / (double)siz * (double)x;
}

inline int calc_score(int phase, int i){
    /*
    cerr << phase << " " << i << endl;
    cerr << test_data.size() << endl;
    cerr << nums[phase] << endl;
    cerr << test_data[phase][i].size() << endl;
    */
    return
        eval_arr[phase][0][test_data[i][0]] + 
        eval_arr[phase][0][test_data[i][1]] + 
        eval_arr[phase][0][test_data[i][2]] + 
        eval_arr[phase][0][test_data[i][3]] + 
        eval_arr[phase][1][test_data[i][4]] + 
        eval_arr[phase][1][test_data[i][5]] + 
        eval_arr[phase][1][test_data[i][6]] + 
        eval_arr[phase][1][test_data[i][7]] + 
        eval_arr[phase][2][test_data[i][8]] + 
        eval_arr[phase][2][test_data[i][9]] + 
        eval_arr[phase][2][test_data[i][10]] + 
        eval_arr[phase][2][test_data[i][11]] + 
        eval_arr[phase][3][test_data[i][12]] + 
        eval_arr[phase][3][test_data[i][13]] + 
        eval_arr[phase][3][test_data[i][14]] + 
        eval_arr[phase][3][test_data[i][15]] + 
        eval_arr[phase][4][test_data[i][16]] + 
        eval_arr[phase][4][test_data[i][17]] + 
        eval_arr[phase][4][test_data[i][18]] + 
        eval_arr[phase][4][test_data[i][19]] + 
        eval_arr[phase][5][test_data[i][20]] + 
        eval_arr[phase][5][test_data[i][21]] + 
        eval_arr[phase][5][test_data[i][22]] + 
        eval_arr[phase][5][test_data[i][23]] + 
        eval_arr[phase][6][test_data[i][24]] + 
        eval_arr[phase][6][test_data[i][25]] + 
        eval_arr[phase][7][test_data[i][26]] + 
        eval_arr[phase][7][test_data[i][27]] + 
        eval_arr[phase][7][test_data[i][28]] + 
        eval_arr[phase][7][test_data[i][29]] + 
        eval_arr[phase][8][test_data[i][30]] + 
        eval_arr[phase][8][test_data[i][31]] + 
        eval_arr[phase][8][test_data[i][32]] + 
        eval_arr[phase][8][test_data[i][33]] + 
        eval_arr[phase][9][test_data[i][34]] + 
        eval_arr[phase][9][test_data[i][35]] + 
        eval_arr[phase][9][test_data[i][36]] + 
        eval_arr[phase][9][test_data[i][37]] + 
        eval_arr[phase][10][test_data[i][38]] + 
        eval_arr[phase][10][test_data[i][39]] + 
        eval_arr[phase][10][test_data[i][40]] + 
        eval_arr[phase][10][test_data[i][41]] + 
        eval_arr[phase][11][test_data[i][42]] + 
        eval_arr[phase][11][test_data[i][43]] + 
        eval_arr[phase][11][test_data[i][44]] + 
        eval_arr[phase][11][test_data[i][45]] + 
        eval_arr[phase][12][test_data[i][46]] + 
        eval_arr[phase][12][test_data[i][47]] + 
        eval_arr[phase][12][test_data[i][48]] + 
        eval_arr[phase][12][test_data[i][49]] + 
        //eval_arr[phase][13][test_data[i][50]] + 
        //eval_arr[phase][13][test_data[i][51]] + 
        //eval_arr[phase][13][test_data[i][52]] + 
        //eval_arr[phase][13][test_data[i][53]] + 
        eval_arr[phase][13][calc_add_idx(test_data[i])];
}

inline double first_scoring(){
    int i, j, score;
    double avg_score, res = 0.0, err;
    avg_score = 0.0;
    for (i = 0; i < nums; ++i){
        score = calc_score(sa_phase, i);
        err = loss(test_labels[i] - score, nums);
        avg_score += err;
        test_scores[i] = err;
    }
    scores = avg_score;
    res = avg_score;
    return res;
}

inline double scoring(int pattern, int idx){
    int i, j, score;
    double avg_score, res = 0.0, err;
    int data_size = nums;
    avg_score = scores;
    for (i = 0; i < test_memo[pattern][idx].size(); ++i){
        avg_score -= test_scores[test_memo[pattern][idx][i]];
        score = calc_score(sa_phase, test_memo[pattern][idx][i]);
        err = loss(test_labels[test_memo[pattern][idx][i]] - score, data_size);
        test_scores[test_memo[pattern][idx][i]] = err;
        avg_score += err;
    }
    scores = avg_score;
    res = scores;
    return res;
}

inline void scoring_mae(){
    int i, j, score;
    double avg_score, res = 0.0;
    avg_score = 0;
    for (i = 0; i < nums; ++i){
        score = calc_score(sa_phase, i);
        avg_score += fabs(test_labels[i] - (double)score) / nums;
    }
    cerr << " " << avg_score << "                      ";
}

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    if (pattern_idx <= 7 || pattern_idx == 12){
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
    } else if (pattern_idx == 11){
        res += p38 * calc_pop(idx, 0, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 6, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 4, pattern_size);
        res += p33 * calc_pop(idx, 7, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 8, pattern_size);
    } else if (pattern_idx == 13){
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
    }
    return res;
}

void sa(unsigned long long tl){
    unsigned long long strt = tim(), now = tim();
    double score = first_scoring(), n_score;
    int phase, pattern, idx, rev_idx, f_val;
    int t = 0, u = 0;
    cerr << score << " ";
    scoring_mae();
    cerr << endl;
    for (;;){
        ++t;
        phase = sa_phase; //myrandrange(0, n_phases);
        pattern = myrandrange(0, n_patterns + 1);
        idx = used_idxes_vector[pattern][myrandrange(0, (int)used_idxes_vector[pattern].size())];
        if (pattern < n_patterns){
            f_val = eval_arr[phase][pattern][idx];
            eval_arr[phase][pattern][idx] += myrandrange(-50, 51);
            rev_idx = calc_rev_idx(pattern, pattern_sizes[pattern], idx);
            eval_arr[phase][pattern][rev_idx] = eval_arr[phase][pattern][idx];
            scoring(pattern, idx);
            n_score = scoring(pattern, rev_idx);
        } else{
            f_val = eval_arr[phase][pattern][idx];
            eval_arr[phase][pattern][idx] += myrandrange(-50, 51);
            n_score = scoring(pattern, idx);
        }
        if (n_score < score){
            score = n_score;
            ++u;
        } else{
            eval_arr[phase][pattern][idx] = f_val;
            scoring(pattern, idx);
            if (pattern < n_patterns){
                eval_arr[phase][pattern][rev_idx] = f_val;
                scoring(pattern, rev_idx);
            }
        }
        if ((t & 0b11111111111) == 0){
            now = tim();
            if (now - strt > tl)
                break;
            cerr << '\r' << (int)((double)(now - strt) / tl * 1000) << " " << t << " " << u << " ";
            cerr << score << " ";
            scoring_mae();
        }
    }
    cerr << '\r';
    cerr << score << " ";
    scoring_mae();
    //cerr << endl;
    //cerr << first_scoring() << endl;
    cerr << t << " " << u << endl;
}

void sa2(unsigned long long tl){
    unsigned long long strt = tim(), now = tim();
    double score = first_scoring(), n_score;
    int phase, pattern, idx1, idx2, rev_idx1, rev_idx2, f_val1, f_val2;
    int t = 0, u = 0;
    cerr << score << " ";
    scoring_mae();
    cerr << endl;
    for (;;){
        ++t;
        phase = sa_phase; //myrandrange(0, n_phases);
        pattern = myrandrange(0, n_patterns + 1);
        idx1 = used_idxes_vector[pattern][myrandrange(0, (int)used_idxes_vector[pattern].size())];
        idx2 = used_idxes_vector[pattern][myrandrange(0, (int)used_idxes_vector[pattern].size())];
        if (pattern < n_patterns){
            f_val1 = eval_arr[phase][pattern][idx1];
            f_val2 = eval_arr[phase][pattern][idx2];
            eval_arr[phase][pattern][idx1] += myrandrange(-10, 11);
            eval_arr[phase][pattern][idx2] += myrandrange(-10, 11);
            rev_idx1 = calc_rev_idx(pattern, pattern_sizes[pattern], idx1);
            rev_idx2 = calc_rev_idx(pattern, pattern_sizes[pattern], idx2);
            eval_arr[phase][pattern][rev_idx1] = eval_arr[phase][pattern][idx1];
            eval_arr[phase][pattern][rev_idx2] = eval_arr[phase][pattern][idx2];
            scoring(pattern, idx1);
            scoring(pattern, idx2);
            scoring(pattern, rev_idx1);
            n_score = scoring(pattern, rev_idx2);
        } else{
            f_val1 = eval_arr[phase][pattern][idx1];
            f_val2 = eval_arr[phase][pattern][idx2];
            eval_arr[phase][pattern][idx1] += myrandrange(-10, 11);
            eval_arr[phase][pattern][idx2] += myrandrange(-10, 11);
            scoring(pattern, idx1);
            n_score = scoring(pattern, idx2);
        }
        //if (prob(score, n_score, strt, now, tl) > myrandom()){
        if (n_score < score){
            score = n_score;
            ++u;
        } else{
            eval_arr[phase][pattern][idx1] = f_val1;
            eval_arr[phase][pattern][idx2] = f_val2;
            scoring(pattern, idx1);
            scoring(pattern, idx2);
            if (pattern < n_patterns){
                eval_arr[phase][pattern][rev_idx1] = f_val1;
                eval_arr[phase][pattern][rev_idx2] = f_val2;
                scoring(pattern, rev_idx1);
                scoring(pattern, rev_idx2);
            }
        }
        if ((t & 0b11111111) == 0){
            now = tim();
            if (now - strt > tl)
                break;
            cerr << '\r' << (int)((double)(now - strt) / tl * 1000) << " " << t << " " << u << " ";
            cerr << score << " ";
            scoring_mae();
        }
    }
    cerr << '\r';
    cerr << score << " ";
    scoring_mae();
    //cerr << endl;
    //cerr << first_scoring() << endl;
    cerr << t << " " << u << endl;
}

int main(int argc, char *argv[]){
    sa_phase = atoi(argv[1]);
    int i, j;

    unsigned long long hour = 0;
    unsigned long long minute = 1;
    unsigned long long second = 0;
    minute += hour * 60;
    second += minute * 60;

    board_init();
    input_param();

    input_test_data(0);
    sa2(second * 1000);

    output_param_onephase();

    return 0;
}