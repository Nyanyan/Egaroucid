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
#define n_patterns 16
#define n_eval (n_patterns + 4 + 4)
#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65
#define max_evaluate_idx 65536 //59049

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
#define sc_w (step * hw2)

#define n_data 100000

#define n_raw_params 86

#define beta 0.00001
unsigned long long hour = 0;
unsigned long long minute = 2;
unsigned long long second = 0;

double alpha[n_eval][max_evaluate_idx];

const int pattern_sizes[n_eval] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10, 0, 0, 0, 0, 8, 8, 8, 8};
const int eval_sizes[n_eval] = {p38, p38, p38, p35, p36, p37, p38, p310, p310, p310, p310, p39, p310, p310, p310, p310, max_surround * max_surround, max_canput * max_canput, max_stability * max_stability, max_stone_num * max_stone_num, p48, p48, p48, p48};
double eval_arr[n_phases][n_eval][max_evaluate_idx];
int test_data[n_phases][n_data / n_phases][n_raw_params];
int n_test_data[n_phases];
double test_labels[n_phases][n_data / n_phases];
int nums;
double scores;
vector<int> test_memo[n_eval][max_evaluate_idx];
int pow4[8];
int pow3[11];

void initialize_param(){
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        cerr << "=";
        for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
            for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem)
                eval_arr[phase_idx][pattern_idx][pattern_elem] = 0;
        }
    }
    cerr << endl;
}

void input_param(){
    ifstream ifs("learned_data/param.txt");
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        exit(1);
    }
    string line;
    int t =0;
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        cerr << "=";
        for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
            for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
                ++t;
                getline(ifs, line);
                eval_arr[phase_idx][pattern_idx][pattern_elem] = stoi(line);
            }
        }
    }
    cerr << t << endl;
}

inline int calc_sur0_sur1(int arr[]){
    return arr[62] * max_surround + arr[63];
}

inline int calc_canput0_canput1(int arr[]){
    return arr[64] * max_canput + arr[65];
}

inline int calc_stab0_stab1(int arr[]){
    return arr[66] * max_stability + arr[67];
}

inline int calc_num0_num1(int arr[]){
    return arr[68] * max_stone_num + arr[69];
}

inline double calc_score(int phase, int i);

void input_test_data(){
    int i, j, k;
    int phase, player, score;
    int t = 0, u = 0;
    nums = 0;
    const int pattern_nums[62] = {
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
        15, 15, 15, 15
    };
    for (i = 0; i < n_phases; ++i)
        n_test_data[i] = 0;
    int sur, canput, stab, num;
    FILE* fp;
    if (fopen_s(&fp, "big_data_flat.dat", "rb") != 0) {
        cerr << "can't open " << endl;
        return;
    }
    while (t < n_data){
        ++t;
        if ((t & 0b1111111111111111) == 0b1111111111111111)
            cerr << '\r' << t;
        if (fread(&phase, 4, 1, fp) < 1)
            break;
        phase = (phase - 4) / phase_n_stones;
        fread(&player, 4, 1, fp);
        fread(test_data[phase][n_test_data[phase]], 4, n_raw_params, fp);
        fread(&score, 4, 1, fp);
        test_labels[phase][n_test_data[phase]] = score * step;
        ++n_test_data[phase];
    }
    cerr << '\r' << t << endl;
    cerr << "loaded data" << endl;
}

inline double calc_score(int phase, int i){
    int res = 
        eval_arr[phase][0][test_data[phase][i][0]] + 
        eval_arr[phase][0][test_data[phase][i][1]] + 
        eval_arr[phase][0][test_data[phase][i][2]] + 
        eval_arr[phase][0][test_data[phase][i][3]] + 
        eval_arr[phase][1][test_data[phase][i][4]] + 
        eval_arr[phase][1][test_data[phase][i][5]] + 
        eval_arr[phase][1][test_data[phase][i][6]] + 
        eval_arr[phase][1][test_data[phase][i][7]] + 
        eval_arr[phase][2][test_data[phase][i][8]] + 
        eval_arr[phase][2][test_data[phase][i][9]] + 
        eval_arr[phase][2][test_data[phase][i][10]] + 
        eval_arr[phase][2][test_data[phase][i][11]] + 
        eval_arr[phase][3][test_data[phase][i][12]] + 
        eval_arr[phase][3][test_data[phase][i][13]] + 
        eval_arr[phase][3][test_data[phase][i][14]] + 
        eval_arr[phase][3][test_data[phase][i][15]] + 
        eval_arr[phase][4][test_data[phase][i][16]] + 
        eval_arr[phase][4][test_data[phase][i][17]] + 
        eval_arr[phase][4][test_data[phase][i][18]] + 
        eval_arr[phase][4][test_data[phase][i][19]] + 
        eval_arr[phase][5][test_data[phase][i][20]] + 
        eval_arr[phase][5][test_data[phase][i][21]] + 
        eval_arr[phase][5][test_data[phase][i][22]] + 
        eval_arr[phase][5][test_data[phase][i][23]] + 
        eval_arr[phase][6][test_data[phase][i][24]] + 
        eval_arr[phase][6][test_data[phase][i][25]] + 
        eval_arr[phase][7][test_data[phase][i][26]] + 
        eval_arr[phase][7][test_data[phase][i][27]] + 
        eval_arr[phase][7][test_data[phase][i][28]] + 
        eval_arr[phase][7][test_data[phase][i][29]] + 
        eval_arr[phase][8][test_data[phase][i][30]] + 
        eval_arr[phase][8][test_data[phase][i][31]] + 
        eval_arr[phase][8][test_data[phase][i][32]] + 
        eval_arr[phase][8][test_data[phase][i][33]] + 
        eval_arr[phase][9][test_data[phase][i][34]] + 
        eval_arr[phase][9][test_data[phase][i][35]] + 
        eval_arr[phase][9][test_data[phase][i][36]] + 
        eval_arr[phase][9][test_data[phase][i][37]] + 
        eval_arr[phase][10][test_data[phase][i][38]] + 
        eval_arr[phase][10][test_data[phase][i][39]] + 
        eval_arr[phase][10][test_data[phase][i][40]] + 
        eval_arr[phase][10][test_data[phase][i][41]] + 
        eval_arr[phase][11][test_data[phase][i][42]] + 
        eval_arr[phase][11][test_data[phase][i][43]] + 
        eval_arr[phase][11][test_data[phase][i][44]] + 
        eval_arr[phase][11][test_data[phase][i][45]] + 
        eval_arr[phase][12][test_data[phase][i][46]] + 
        eval_arr[phase][12][test_data[phase][i][47]] + 
        eval_arr[phase][12][test_data[phase][i][48]] + 
        eval_arr[phase][12][test_data[phase][i][49]] + 
        eval_arr[phase][13][test_data[phase][i][50]] + 
        eval_arr[phase][13][test_data[phase][i][51]] + 
        eval_arr[phase][13][test_data[phase][i][52]] + 
        eval_arr[phase][13][test_data[phase][i][53]] + 
        eval_arr[phase][14][test_data[phase][i][54]] + 
        eval_arr[phase][14][test_data[phase][i][55]] + 
        eval_arr[phase][14][test_data[phase][i][56]] + 
        eval_arr[phase][14][test_data[phase][i][57]] + 
        eval_arr[phase][15][test_data[phase][i][58]] + 
        eval_arr[phase][15][test_data[phase][i][59]] + 
        eval_arr[phase][15][test_data[phase][i][60]] + 
        eval_arr[phase][15][test_data[phase][i][61]] + 
        eval_arr[phase][16][calc_sur0_sur1(test_data[phase][i])] + 
        eval_arr[phase][17][calc_canput0_canput1(test_data[phase][i])] + 
        eval_arr[phase][18][calc_stab0_stab1(test_data[phase][i])] + 
        eval_arr[phase][19][calc_num0_num1(test_data[phase][i])] + 
        eval_arr[phase][20][test_data[phase][i][70]] + 
        eval_arr[phase][20][test_data[phase][i][71]] + 
        eval_arr[phase][20][test_data[phase][i][72]] + 
        eval_arr[phase][20][test_data[phase][i][73]] + 
        eval_arr[phase][21][test_data[phase][i][74]] + 
        eval_arr[phase][21][test_data[phase][i][75]] + 
        eval_arr[phase][21][test_data[phase][i][76]] + 
        eval_arr[phase][21][test_data[phase][i][77]] + 
        eval_arr[phase][22][test_data[phase][i][78]] + 
        eval_arr[phase][22][test_data[phase][i][79]] + 
        eval_arr[phase][22][test_data[phase][i][80]] + 
        eval_arr[phase][22][test_data[phase][i][81]] + 
        eval_arr[phase][23][test_data[phase][i][82]] + 
        eval_arr[phase][23][test_data[phase][i][83]] + 
        eval_arr[phase][23][test_data[phase][i][84]] + 
        eval_arr[phase][23][test_data[phase][i][85]];
    return max(-64 * step, min(64 * step, res));
}

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}

inline int calc_pop4(int a, int b, int s){
    return (a / pow4[s - 1 - b]) % 4;
}

void init(){
    int i, j;
    pow4[0] = 1;
    for (i = 1; i < 8; ++i)
        pow4[i] = pow4[i - 1] * 4;
    pow3[0] = 1;
    for (i = 1; i < 11; ++i)
        pow3[i] = pow3[i - 1] * 3;
}

void calc_sd(double sds[]){
    int ans, pred;
    double sd, err;
    for (int phase = 0; phase < n_phases; ++phase){
        sd = 0.0;
        for (int i = 0; i < n_test_data[phase]; ++i){
            ans = test_labels[phase][i];
            pred = calc_score(phase, i);
            //cerr << ans << " " << pred << endl;
            err = (double)(ans - pred) / step;
            sd += err / n_test_data[phase] * err;
        }
        sd = sqrt(sd);
        sds[phase] = sd;
        cerr << phase << " " << sd << endl;
    }
}

double f(double x, double sd){
    constexpr double sqrt_2pi = sqrt(2.0 * 3.14159265);
    return exp(-x * x / 2.0 / sd / sd) / sqrt_2pi / sd * 2.0;
}

void calc_comb_sd(double sds[], double n_sds[n_phases][n_phases]){
    double p1, p2, sd, err;
    for (int phase1 = 0; phase1 < n_phases; ++phase1){
        for (int phase2 = 0; phase2 < n_phases; ++phase2){
            sd = 0.0;
            for (int s = -64; s <= 64; s += 2){
                for (int t = -64; t <= 64; t += 2){
                    p1 = f((double)s, sds[phase1]);
                    p2 = f((double)t, sds[phase2]);
                    err = (double)(s - t);
                    sd += err * err * p1 * p2;
                }
            }
            sd = sqrt(sd);
            n_sds[phase1][phase2] = sd;
            cout << phase1 << " " << phase2 << " " << sd << endl;
        }
    }
}

int main(int argc, char *argv[]){
    int i, j;

    board_init();
    init();
    input_param();
    input_test_data();

    double sds[n_phases], n_sds[n_phases][n_phases];
    calc_sd(sds);

    calc_comb_sd(sds, n_sds);

    return 0;
}