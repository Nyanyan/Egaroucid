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
#define sc_w (step * HW2)

#define n_raw_params 86

double beta = 0.001;
unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

double alpha[n_eval][max_evaluate_idx];

const int pattern_sizes[n_eval] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10, 0, 0, 0, 0, 8, 8, 8, 8};
const int eval_sizes[n_eval] = {p38, p38, p38, p35, p36, p37, p38, p310, p310, p310, p310, p39, p310, p310, p310, p310, max_surround * max_surround, max_canput * max_canput, max_stability * max_stability, max_stone_num * max_stone_num, p48, p48, p48, p48};
double eval_arr[n_phases][n_eval][max_evaluate_idx];
int test_data[n_raw_params + 3];
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

constexpr double accept_ratio[65] = {
    0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 64
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 56
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 48
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 40
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 32
    1.0, 1.0, 1.0, 1.0, 0.99, 0.98, 0.97, 0.96, // 24
    0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, //16
    0.79, 0.77, 0.75, 0.73, 0.65, 0.63, 0.62, 0.61, // 8
    0.6, // 0
};

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
    ofstream fout;
    fout.open("big_data.dat", ios::out|ios::binary|ios::trunc);
    if (!fout){
        cerr << "can't open" << endl;
        return;
    }
    FILE* fp;
    for (int file_idx = 1; file_idx < argc; ++file_idx){
        cerr << argv[file_idx] << endl;
        if (fopen_s(&fp, argv[file_idx], "rb") != 0) {
            cerr << "can't open " << argv[file_idx] << endl;
            continue;
        }
        while (true){
            ++t;
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                cerr << '\r' << t;
            if (fread(test_data, 4, n_raw_params + 3, fp) < n_raw_params + 3)
                break;
            if (myrandom() < accept_ratio[HW2 - abs(test_data[n_raw_params + 2])]){
                ++n_data_score[test_data[n_raw_params + 2] + HW2];
                for (i = 0; i < n_raw_params + 3; ++i)
                    fout.write((char*)&test_data[i], 4);
            }
        }
    }
    for (i = 0; i < 129; i += 2)
        cout << i - HW2 << " " << n_data_score[i] << endl;
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

int main(int argc, char *argv[]){
    board_init();
    init();
    input_test_data(argc, argv);

    return 0;
}