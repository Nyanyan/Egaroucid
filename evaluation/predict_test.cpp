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

#define n_data 50000000

#define n_raw_params 86

double beta = 0.001;
unsigned long long hour = 0;
unsigned long long minute = 3;
unsigned long long second = 0;

double alpha[n_eval][max_evaluate_idx];

const int pattern_sizes[n_eval] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10, 0, 0, 0, 0, 8, 8, 8, 8};
const int eval_sizes[n_eval] = {p38, p38, p38, p35, p36, p37, p38, p310, p310, p310, p310, p39, p310, p310, p310, p310, max_surround * max_surround, max_canput * max_canput, max_stability * max_stability, max_stone_num * max_stone_num, p48, p48, p48, p48};
double eval_arr[n_phases][n_eval][max_evaluate_idx];
int test_data[n_data / n_phases][n_raw_params];
double test_labels[n_data / n_phases];
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
            eval_arr[sa_phase][pattern_idx][pattern_elem] = stoi(line);
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

inline double calc_score(int phase, int data[]);

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
    FILE* fp;
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
            fread(test_data[nums], 4, n_raw_params, fp);
            fread(&score, 4, 1, fp);
            if (phase == sa_phase){
                ++u;
                for (i = 0; i < 62; ++i)
                    used_idxes[pattern_nums[i]].emplace(test_data[nums][i]);
                sur = calc_sur0_sur1(test_data[nums]);
                canput = calc_canput0_canput1(test_data[nums]);
                stab = calc_stab0_stab1(test_data[nums]);
                num = calc_num0_num1(test_data[nums]);
                used_idxes[16].emplace(sur);
                used_idxes[17].emplace(canput);
                used_idxes[18].emplace(stab);
                used_idxes[19].emplace(num);
                for (i = 0; i < 16; ++i)
                    used_idxes[20 + i / 4].emplace(test_data[nums][70 + i]);
                test_labels[nums] = score * step;
                for (i = 0; i < 62; ++i)
                    test_memo[pattern_nums[i]][test_data[nums][i]].push_back(nums);
                test_memo[16][sur].push_back(nums);
                test_memo[17][canput].push_back(nums);
                test_memo[18][stab].push_back(nums);
                test_memo[19][num].push_back(nums);
                for (i = 0; i < 16; ++i)
                    test_memo[20 + i / 4][test_data[nums][70 + i]].push_back(nums);
                test_scores.push_back(0);
                pre_calc_scores.push_back(0);
                ++n_data_score[score + 64];
                for (i = 0; i < 62; ++i)
                    ++n_data_idx[pattern_nums[i]][test_data[nums][i]][score + 64];
                ++n_data_idx[16][sur][score + 64];
                ++n_data_idx[17][canput][score + 64];
                ++n_data_idx[18][stab][score + 64];
                ++n_data_idx[19][num][score + 64];
                for (i = 0; i < 16; ++i)
                    ++n_data_idx[20 + i / 4][test_data[nums][70 + i]][score + 64];
                /*
                if (nums == 0){
                    for (i = 0; i < n_raw_params; ++i)
                        cerr << test_data[nums][i] << " ";
                    cerr << score << " " << calc_score(sa_phase, nums) << endl;
                }
                */
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

void output_param(){
    int phase_idx, pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        cerr << "=";
        for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
            for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
                cout << round(eval_arr[phase_idx][pattern_idx][pattern_elem]) << endl;
            }
        }
    }
    cerr << endl;
}

void output_param_onephase(){
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j;
    cerr << "=";
    for (pattern_idx = 0; pattern_idx < n_eval; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            cout << round(eval_arr[sa_phase][pattern_idx][pattern_elem]) << endl;
        }
    }
    cerr << endl;
}

inline double loss(double x, int siz){
    //double sq_size = sqrt((double)siz);
    //double tmp = (double)x / sq_size;
    return (double)x / (double)siz * (double)x;
}

inline double calc_score(int phase, int data[]){
    cerr << eval_arr[phase][0][data[0]] << endl;
    cerr << eval_arr[phase][0][data[1]] << endl;
    cerr << eval_arr[phase][0][data[2]] << endl;
    cerr << eval_arr[phase][0][data[3]] << endl;
    cerr << eval_arr[phase][1][data[4]] << endl;
    cerr << eval_arr[phase][1][data[5]] << endl;
    cerr << eval_arr[phase][1][data[6]] << endl;
    cerr << eval_arr[phase][1][data[7]] << endl;
    cerr << eval_arr[phase][2][data[8]] << endl;
    cerr << eval_arr[phase][2][data[9]] << endl;
    cerr << eval_arr[phase][2][data[10]] << endl;
    cerr << eval_arr[phase][2][data[11]] << endl;
    cerr << eval_arr[phase][3][data[12]] << endl;
    cerr << eval_arr[phase][3][data[13]] << endl;
    cerr << eval_arr[phase][3][data[14]] << endl;
    cerr << eval_arr[phase][3][data[15]] << endl;
    cerr << eval_arr[phase][4][data[16]] << endl;
    cerr << eval_arr[phase][4][data[17]] << endl;
    cerr << eval_arr[phase][4][data[18]] << endl;
    cerr << eval_arr[phase][4][data[19]] << endl;
    cerr << eval_arr[phase][5][data[20]] << endl;
    cerr << eval_arr[phase][5][data[21]] << endl;
    cerr << eval_arr[phase][5][data[22]] << endl;
    cerr << eval_arr[phase][5][data[23]] << endl;
    cerr << eval_arr[phase][6][data[24]] << endl;
    cerr << eval_arr[phase][6][data[25]] << endl;
    cerr << eval_arr[phase][7][data[26]] << endl;
    cerr << eval_arr[phase][7][data[27]] << endl;
    cerr << eval_arr[phase][7][data[28]] << endl;
    cerr << eval_arr[phase][7][data[29]] << endl;
    cerr << eval_arr[phase][8][data[30]] << endl;
    cerr << eval_arr[phase][8][data[31]] << endl;
    cerr << eval_arr[phase][8][data[32]] << endl;
    cerr << eval_arr[phase][8][data[33]] << endl;
    cerr << eval_arr[phase][9][data[34]] << endl;
    cerr << eval_arr[phase][9][data[35]] << endl;
    cerr << eval_arr[phase][9][data[36]] << endl;
    cerr << eval_arr[phase][9][data[37]] << endl;
    cerr << eval_arr[phase][10][data[38]] << endl;
    cerr << eval_arr[phase][10][data[39]] << endl;
    cerr << eval_arr[phase][10][data[40]] << endl;
    cerr << eval_arr[phase][10][data[41]] << endl;
    cerr << eval_arr[phase][11][data[42]] << endl;
    cerr << eval_arr[phase][11][data[43]] << endl;
    cerr << eval_arr[phase][11][data[44]] << endl;
    cerr << eval_arr[phase][11][data[45]] << endl;
    cerr << eval_arr[phase][12][data[46]] << endl;
    cerr << eval_arr[phase][12][data[47]] << endl;
    cerr << eval_arr[phase][12][data[48]] << endl;
    cerr << eval_arr[phase][12][data[49]] << endl;
    cerr << eval_arr[phase][13][data[50]] << endl;
    cerr << eval_arr[phase][13][data[51]] << endl;
    cerr << eval_arr[phase][13][data[52]] << endl;
    cerr << eval_arr[phase][13][data[53]] << endl;
    cerr << eval_arr[phase][14][data[54]] << endl;
    cerr << eval_arr[phase][14][data[55]] << endl;
    cerr << eval_arr[phase][14][data[56]] << endl;
    cerr << eval_arr[phase][14][data[57]] << endl;
    cerr << eval_arr[phase][15][data[58]] << endl;
    cerr << eval_arr[phase][15][data[59]] << endl;
    cerr << eval_arr[phase][15][data[60]] << endl;
    cerr << eval_arr[phase][15][data[61]] << endl;
    cerr << eval_arr[phase][16][calc_sur0_sur1(data)] << endl;
    cerr << eval_arr[phase][17][calc_canput0_canput1(data)] << endl;
    cerr << eval_arr[phase][18][calc_stab0_stab1(data)] << endl;
    cerr << eval_arr[phase][19][calc_num0_num1(data)] << endl;
    cerr << eval_arr[phase][20][data[70]] << endl;
    cerr << eval_arr[phase][20][data[71]] << endl;
    cerr << eval_arr[phase][20][data[72]] << endl;
    cerr << eval_arr[phase][20][data[73]] << endl;
    cerr << eval_arr[phase][21][data[74]] << endl;
    cerr << eval_arr[phase][21][data[75]] << endl;
    cerr << eval_arr[phase][21][data[76]] << endl;
    cerr << eval_arr[phase][21][data[77]] << endl;
    cerr << eval_arr[phase][22][data[78]] << endl;
    cerr << eval_arr[phase][22][data[79]] << endl;
    cerr << eval_arr[phase][22][data[80]] << endl;
    cerr << eval_arr[phase][22][data[81]] << endl;
    cerr << eval_arr[phase][23][data[82]] << endl;
    cerr << eval_arr[phase][23][data[83]] << endl;
    cerr << eval_arr[phase][23][data[84]] << endl;
    cerr << eval_arr[phase][23][data[85]] << endl;
    int res = 
        eval_arr[phase][0][data[0]] + 
        eval_arr[phase][0][data[1]] + 
        eval_arr[phase][0][data[2]] + 
        eval_arr[phase][0][data[3]] + 
        eval_arr[phase][1][data[4]] + 
        eval_arr[phase][1][data[5]] + 
        eval_arr[phase][1][data[6]] + 
        eval_arr[phase][1][data[7]] + 
        eval_arr[phase][2][data[8]] + 
        eval_arr[phase][2][data[9]] + 
        eval_arr[phase][2][data[10]] + 
        eval_arr[phase][2][data[11]] + 
        eval_arr[phase][3][data[12]] + 
        eval_arr[phase][3][data[13]] + 
        eval_arr[phase][3][data[14]] + 
        eval_arr[phase][3][data[15]] + 
        eval_arr[phase][4][data[16]] + 
        eval_arr[phase][4][data[17]] + 
        eval_arr[phase][4][data[18]] + 
        eval_arr[phase][4][data[19]] + 
        eval_arr[phase][5][data[20]] + 
        eval_arr[phase][5][data[21]] + 
        eval_arr[phase][5][data[22]] + 
        eval_arr[phase][5][data[23]] + 
        eval_arr[phase][6][data[24]] + 
        eval_arr[phase][6][data[25]] + 
        eval_arr[phase][7][data[26]] + 
        eval_arr[phase][7][data[27]] + 
        eval_arr[phase][7][data[28]] + 
        eval_arr[phase][7][data[29]] + 
        eval_arr[phase][8][data[30]] + 
        eval_arr[phase][8][data[31]] + 
        eval_arr[phase][8][data[32]] + 
        eval_arr[phase][8][data[33]] + 
        eval_arr[phase][9][data[34]] + 
        eval_arr[phase][9][data[35]] + 
        eval_arr[phase][9][data[36]] + 
        eval_arr[phase][9][data[37]] + 
        eval_arr[phase][10][data[38]] + 
        eval_arr[phase][10][data[39]] + 
        eval_arr[phase][10][data[40]] + 
        eval_arr[phase][10][data[41]] + 
        eval_arr[phase][11][data[42]] + 
        eval_arr[phase][11][data[43]] + 
        eval_arr[phase][11][data[44]] + 
        eval_arr[phase][11][data[45]] + 
        eval_arr[phase][12][data[46]] + 
        eval_arr[phase][12][data[47]] + 
        eval_arr[phase][12][data[48]] + 
        eval_arr[phase][12][data[49]] + 
        eval_arr[phase][13][data[50]] + 
        eval_arr[phase][13][data[51]] + 
        eval_arr[phase][13][data[52]] + 
        eval_arr[phase][13][data[53]] + 
        eval_arr[phase][14][data[54]] + 
        eval_arr[phase][14][data[55]] + 
        eval_arr[phase][14][data[56]] + 
        eval_arr[phase][14][data[57]] + 
        eval_arr[phase][15][data[58]] + 
        eval_arr[phase][15][data[59]] + 
        eval_arr[phase][15][data[60]] + 
        eval_arr[phase][15][data[61]] + 
        eval_arr[phase][16][calc_sur0_sur1(data)] + 
        eval_arr[phase][17][calc_canput0_canput1(data)] + 
        eval_arr[phase][18][calc_stab0_stab1(data)] + 
        eval_arr[phase][19][calc_num0_num1(data)] + 
        eval_arr[phase][20][data[70]] + 
        eval_arr[phase][20][data[71]] + 
        eval_arr[phase][20][data[72]] + 
        eval_arr[phase][20][data[73]] + 
        eval_arr[phase][21][data[74]] + 
        eval_arr[phase][21][data[75]] + 
        eval_arr[phase][21][data[76]] + 
        eval_arr[phase][21][data[77]] + 
        eval_arr[phase][22][data[78]] + 
        eval_arr[phase][22][data[79]] + 
        eval_arr[phase][22][data[80]] + 
        eval_arr[phase][22][data[81]] + 
        eval_arr[phase][23][data[82]] + 
        eval_arr[phase][23][data[83]] + 
        eval_arr[phase][23][data[84]] + 
        eval_arr[phase][23][data[85]];
        /*
        + 
        eval_arr[phase][24][test_data[i][86]] + 
        eval_arr[phase][24][test_data[i][87]] + 
        eval_arr[phase][24][test_data[i][88]] + 
        eval_arr[phase][24][test_data[i][89]] + 
        eval_arr[phase][25][test_data[i][90]] + 
        eval_arr[phase][25][test_data[i][91]] + 
        eval_arr[phase][25][test_data[i][92]] + 
        eval_arr[phase][25][test_data[i][93]] + 
        eval_arr[phase][26][test_data[i][94]] + 
        eval_arr[phase][26][test_data[i][95]] + 
        eval_arr[phase][26][test_data[i][96]] + 
        eval_arr[phase][26][test_data[i][97]] + 
        eval_arr[phase][27][test_data[i][98]] + 
        eval_arr[phase][27][test_data[i][99]] + 
        eval_arr[phase][27][test_data[i][100]] + 
        eval_arr[phase][27][test_data[i][101]];
        */
    /*
    if (res > 0)
        res += step / 2;
    else if (res < 0)
        res -= step / 2;
    res /= step;
    res = max(-64, min(64, res));
    res *= step;
    */
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
    } else if (pattern_idx == 14){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 2, pattern_size);
        res += p37 * calc_pop(idx, 1, pattern_size);
        res += p36 * calc_pop(idx, 3, pattern_size);
        res += p35 * calc_pop(idx, 6, pattern_size);
        res += p34 * calc_pop(idx, 8, pattern_size);
        res += p33 * calc_pop(idx, 4, pattern_size);
        res += p32 * calc_pop(idx, 7, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 9, pattern_size);
    } else if (pattern_idx == 15){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 2, pattern_size);
        res += p37 * calc_pop(idx, 1, pattern_size);
        res += p36 * calc_pop(idx, 3, pattern_size);
        res += p35 * calc_pop(idx, 7, pattern_size);
        res += p34 * calc_pop(idx, 8, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 4, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    } else if (pattern_idx >= n_patterns + 4){
        for (int i = 0; i < 8; ++i){
            res |= (1 & (idx >> i)) << (HW_M1 - i);
            res |= (1 & (idx >> (HW + i))) << (HW + HW_M1 - i);
        }
    }
    return res;
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
        for (j = 0; j < eval_sizes[i]; ++j)
            rev_idxes[i][j] = calc_rev_idx(i, pattern_sizes[i], j);
    }
}

#define max_surround 100
#define max_canput 50
#define max_stability 65
#define max_stone_num 65

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

#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#define P48 65536

uint64_t stability_edge_arr[N_8BIT][N_8BIT][2];

inline int calc_phase_idx(const Board *b){
    return (b->n - 4) / PHASE_N_STONES;
}

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i >= 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w, int ob, int ow){
    int i, nb, nw, res = 0b11111111;
    res &= b & ob;
    res &= w & ow;
    for (i = 0; i < HW; ++i){
        if ((1 & (b >> i)) == 0 && (1 & (w >> i)) == 0){
            probably_move_line(b, w, i, &nb, &nw);
            res &= calc_stability_line(nb, nw, ob, ow);
            probably_move_line(w, b, i, &nw, &nb);
            res &= calc_stability_line(nb, nw, ob, ow);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int idx, place, b, w, stab;
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < HW; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * HW);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
        }
    }
}

inline uint64_t calc_surround_part(const uint64_t player, const int dr){
    return (player << dr | player >> dr);
}

inline int calc_surround(const uint64_t player, const uint64_t empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline void calc_stability(Board *b, int *stab0, int *stab1){
    uint64_t full_h, full_v, full_d7, full_d9;
    uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
    uint64_t h, v, d7, d9;
    const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    int pl, op;
    pl = b->player & 0b11111111;
    op = b->opponent & 0b11111111;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (b->player >> 56) & 0b11111111;
    op = (b->opponent >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

    n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
    while (n_stability & ~player_stability){
        player_stability |= n_stability;
        h = (player_stability >> 1) | (player_stability << 1) | full_h;
        v = (player_stability >> HW) | (player_stability << HW) | full_v;
        d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
        d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & player_mask;
    }

    n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
    while (n_stability & ~opponent_stability){
        opponent_stability |= n_stability;
        h = (opponent_stability >> 1) | (opponent_stability << 1) | full_h;
        v = (opponent_stability >> HW) | (opponent_stability << HW) | full_v;
        d7 = (opponent_stability >> HW_M1) | (opponent_stability << HW_M1) | full_d7;
        d9 = (opponent_stability >> HW_P1) | (opponent_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & opponent_mask;
    }

    *stab0 = pop_count_ull(player_stability);
    *stab1 = pop_count_ull(opponent_stability);
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4){
    return b_arr[p0] * P34 + b_arr[p1] * P33 + b_arr[p2] * P32 + b_arr[P3] * P31 + b_arr[P4];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5){
    return b_arr[p0] * P35 + b_arr[p1] * P34 + b_arr[p2] * P33 + b_arr[P3] * P32 + b_arr[P4] * P31 + b_arr[p5];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6){
    return b_arr[p0] * P36 + b_arr[p1] * P35 + b_arr[p2] * P34 + b_arr[P3] * P33 + b_arr[P4] * P32 + b_arr[p5] * P31 + b_arr[p6];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7){
    return b_arr[p0] * P37 + b_arr[p1] * P36 + b_arr[p2] * P35 + b_arr[P3] * P34 + b_arr[P4] * P33 + b_arr[p5] * P32 + b_arr[p6] * P31 + b_arr[p7];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8){
    return b_arr[p0] * P38 + b_arr[p1] * P37 + b_arr[p2] * P36 + b_arr[P3] * P35 + b_arr[P4] * P34 + b_arr[p5] * P33 + b_arr[p6] * P32 + b_arr[p7] * P31 + b_arr[p8];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int P3, const int P4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return b_arr[p0] * P39 + b_arr[p1] * P38 + b_arr[p2] * P37 + b_arr[P3] * P36 + b_arr[P4] * P35 + b_arr[p5] * P34 + b_arr[p6] * P33 + b_arr[p7] * P32 + b_arr[p8] * P31 + b_arr[p9];
}

inline int create_canput_line_h(uint64_t b, uint64_t w, int t){
    return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v(uint64_t b, uint64_t w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}


inline void calc_idx(int phase_idx, Board *b, int idxes[]){
    uint_fast8_t b_arr[HW2];
    b->translate_to_arr_player(b_arr);
    idxes[0] = pick_pattern(phase_idx, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15);
    idxes[1] = pick_pattern(phase_idx, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57);
    idxes[2] = pick_pattern(phase_idx, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55);
    idxes[3] = pick_pattern(phase_idx, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62);
    idxes[4] = pick_pattern(phase_idx, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23);
    idxes[5] = pick_pattern(phase_idx, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58);
    idxes[6] = pick_pattern(phase_idx, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47);
    idxes[7] = pick_pattern(phase_idx, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61);
    idxes[8] = pick_pattern(phase_idx, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31);
    idxes[9] = pick_pattern(phase_idx, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59);
    idxes[10] = pick_pattern(phase_idx, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39);
    idxes[11] = pick_pattern(phase_idx, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60);
    idxes[12] = pick_pattern(phase_idx, 3, b_arr, 3, 12, 21, 30, 39);
    idxes[13] = pick_pattern(phase_idx, 3, b_arr, 4, 11, 18, 25, 32);
    idxes[14] = pick_pattern(phase_idx, 3, b_arr, 24, 33, 42, 51, 60);
    idxes[15] = pick_pattern(phase_idx, 3, b_arr, 59, 52, 45, 38, 31);
    idxes[16] = pick_pattern(phase_idx, 4, b_arr, 2, 11, 20, 29, 38, 47);
    idxes[17] = pick_pattern(phase_idx, 4, b_arr, 5, 12, 19, 26, 33, 40);
    idxes[18] = pick_pattern(phase_idx, 4, b_arr, 16, 25, 34, 43, 52, 61);
    idxes[19] = pick_pattern(phase_idx, 4, b_arr, 58, 51, 44, 37, 30, 23);
    idxes[20] = pick_pattern(phase_idx, 5, b_arr, 1, 10, 19, 28, 37, 46, 55);
    idxes[21] = pick_pattern(phase_idx, 5, b_arr, 6, 13, 20, 27, 34, 41, 48);
    idxes[22] = pick_pattern(phase_idx, 5, b_arr, 8, 17, 26, 35, 44, 53, 62);
    idxes[23] = pick_pattern(phase_idx, 5, b_arr, 57, 50, 43, 36, 29, 22, 15);
    idxes[24] = pick_pattern(phase_idx, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63);
    idxes[25] = pick_pattern(phase_idx, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56);
    idxes[26] = pick_pattern(phase_idx, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14);
    idxes[27] = pick_pattern(phase_idx, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49);
    idxes[28] = pick_pattern(phase_idx, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54);
    idxes[29] = pick_pattern(phase_idx, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7, 14);
    idxes[30] = pick_pattern(phase_idx, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24);
    idxes[31] = pick_pattern(phase_idx, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31);
    idxes[32] = pick_pattern(phase_idx, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39);
    idxes[33] = pick_pattern(phase_idx, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32);
    idxes[34] = pick_pattern(phase_idx, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13);
    idxes[35] = pick_pattern(phase_idx, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41);
    idxes[36] = pick_pattern(phase_idx, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53);
    idxes[37] = pick_pattern(phase_idx, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46);
    idxes[38] = pick_pattern(phase_idx, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26);
    idxes[39] = pick_pattern(phase_idx, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29);
    idxes[40] = pick_pattern(phase_idx, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34);
    idxes[41] = pick_pattern(phase_idx, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37);
    idxes[42] = pick_pattern(phase_idx, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18);
    idxes[43] = pick_pattern(phase_idx, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21);
    idxes[44] = pick_pattern(phase_idx, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42);
    idxes[45] = pick_pattern(phase_idx, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45);
    idxes[46] = pick_pattern(phase_idx, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13);
    idxes[47] = pick_pattern(phase_idx, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41);
    idxes[48] = pick_pattern(phase_idx, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53);
    idxes[49] = pick_pattern(phase_idx, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22);
    idxes[50] = pick_pattern(phase_idx, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32);
    idxes[51] = pick_pattern(phase_idx, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39);
    idxes[52] = pick_pattern(phase_idx, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31);
    idxes[53] = pick_pattern(phase_idx, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24);
    idxes[54] = pick_pattern(phase_idx, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27);
    idxes[55] = pick_pattern(phase_idx, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28);
    idxes[56] = pick_pattern(phase_idx, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35);
    idxes[57] = pick_pattern(phase_idx, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36);
    idxes[58] = pick_pattern(phase_idx, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33);
    idxes[59] = pick_pattern(phase_idx, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38);
    idxes[60] = pick_pattern(phase_idx, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25);
    idxes[61] = pick_pattern(phase_idx, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
    idxes[62] = min(max_surround - 1, calc_surround(b->player, ~(b->player | b->opponent)));
    idxes[63] = min(max_surround - 1, calc_surround(b->opponent, ~(b->player | b->opponent)));
    uint64_t player_mobility = calc_legal(b->player, b->opponent);
    uint64_t opponent_mobility = calc_legal(b->opponent, b->player);
    idxes[64] = pop_count_ull(player_mobility);
    idxes[65] = pop_count_ull(opponent_mobility);
    calc_stability(b, &idxes[66], &idxes[67]);
    idxes[68] = pop_count_ull(b->player);
    idxes[69] = pop_count_ull(b->opponent);
    idxes[70] = create_canput_line_h(player_mobility, opponent_mobility, 0);
    idxes[71] = create_canput_line_h(player_mobility, opponent_mobility, 7);
    idxes[72] = create_canput_line_v(player_mobility, opponent_mobility, 0);
    idxes[73] = create_canput_line_v(player_mobility, opponent_mobility, 7);
    idxes[74] = create_canput_line_h(player_mobility, opponent_mobility, 1);
    idxes[75] = create_canput_line_h(player_mobility, opponent_mobility, 6);
    idxes[76] = create_canput_line_v(player_mobility, opponent_mobility, 1);
    idxes[77] = create_canput_line_v(player_mobility, opponent_mobility, 6);
    idxes[78] = create_canput_line_h(player_mobility, opponent_mobility, 2);
    idxes[79] = create_canput_line_h(player_mobility, opponent_mobility, 5);
    idxes[80] = create_canput_line_v(player_mobility, opponent_mobility, 2);
    idxes[81] = create_canput_line_v(player_mobility, opponent_mobility, 5);
    idxes[82] = create_canput_line_h(player_mobility, opponent_mobility, 3);
    idxes[83] = create_canput_line_h(player_mobility, opponent_mobility, 4);
    idxes[84] = create_canput_line_v(player_mobility, opponent_mobility, 3);
    idxes[85] = create_canput_line_v(player_mobility, opponent_mobility, 4);
}

inline int convert_idx(string str, int idxes[]){
    cerr << str << endl;
    int i, j;
    unsigned long long bk = 0, wt = 0;
    char elem;
    Board b;
    b.n = 0;
    b.parity = 0;
    for (i = 0; i < HW; ++i){
        for (j = 0; j < HW; ++j){
            elem = str[i * HW + j];
            if (elem != '.'){
                bk |= (unsigned long long)(elem == '0') << (HW2_M1 - i * HW - j);
                wt |= (unsigned long long)(elem == '1') << (HW2_M1 - i * HW - j);
                ++b.n;
            }
        }
    }
    int ai_player, score;
    ai_player = (str[65] == '0' ? 0 : 1);
    if (ai_player == 0){
        b.player = bk;
        b.opponent = wt;
    } else{
        b.player = wt;
        b.opponent = bk;
    }
    score = stoi(str.substr(67));
    if (ai_player == 1)
        score = -score;
    b.print();
    calc_idx(0, &b, idxes);
    return (pop_count_ull(bk) + pop_count_ull(wt) - 4) / 2;
}

int main(int argc, char *argv[]){
    int i, j;

    minute += hour * 60;
    second += minute * 60;

    cerr << sa_phase << " " << second << " " << beta << endl;

    board_init();
    init_evaluation_base();
    init();
    input_param();
    string line;
    getline(cin, line);
    int data[86];
    int phase = convert_idx(line, data);

    cerr << "value: " << calc_score(phase, data) << endl;

    return 0;
}