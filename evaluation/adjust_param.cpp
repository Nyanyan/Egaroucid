#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include "evaluation_definition.hpp"

struct Data{
    uint16_t features[N_FEATURES];
    int score;
};

#define N_MAX_DATA 70000000

double eval_arr[N_EVAL][MAX_EVALUATE_IDX];
double alpha[N_EVAL][MAX_EVALUATE_IDX];
std::vector<Data> test_data;
std::vector<int> preds;
uint16_t rev_idxes[N_EVAL][MAX_EVALUATE_IDX];
std::vector<std::vector<std::vector<int>>> feature_idx_to_data(N_EVAL, std::vector<std::vector<int>>(MAX_EVALUATE_IDX, std::vector<int>(0)));

inline double predict(int problem_idx){
    double res = 0.0;
    for (int i = 0; i < N_FEATURES; ++i)
        res += eval_arr[feature_to_eval_idx[i]][test_data[problem_idx].features[i]];
    if (res > (double)SCORE_MAX)
        res = (double)SCORE_MAX;
    else if (res < -(double)SCORE_MAX)
        res = -(double)SCORE_MAX;
    return res;
}

int round_score(double score){
    return (int)round(score / STEP);
}

void calc_score(){
    double err;
    double mse = 0.0;
    double mae = 0.0;
    double double_data_size = (double)test_data.size();
    for (int i = 0; i < (int)test_data.size(); ++i){
        preds[i] = round_score(predict(i));
        err = (double)abs(preds[i] - test_data[i].score);
        mse += err / double_data_size * err;
        mae += err / double_data_size;
    }
    std::cerr << " mse " << mse << " mae " << mae << "                             ";
}

double calc_error(int feature, int idx){
    double res = 0.0;
    double raw_err;
    for (const int &i: feature_idx_to_data[feature][idx]){
        raw_err = (double)(test_data[i].score - preds[i]);
        res += raw_err;
    }
    return res;
}

void next_step(){
    double err;
    for (int feature = 0; feature < N_EVAL; ++feature){
        for (int idx = 0; idx < eval_sizes[feature]; ++idx){
            if (idx == rev_idxes[feature][idx]){
                err = calc_error(feature, idx);
                eval_arr[feature][idx] += 2.0 * alpha[feature][idx] * err;
            } else if (idx < rev_idxes[feature][idx]){
                err = calc_error(feature, idx) + calc_error(feature, rev_idxes[feature][idx]);
                eval_arr[feature][idx] += 2.0 * alpha[feature][idx] * err;
                eval_arr[feature][rev_idxes[feature][idx]] += 2.0 * alpha[feature][rev_idxes[feature][idx]] * err;
            }
        }
    }
}

void gradient_descent(uint64_t tl){
    uint64_t strt = tim();
    int t = 0;
    calc_score();
    std::cerr << std::endl;
    while (tim() - strt < tl){
        std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
        calc_score();
        next_step();
        ++t;
    }
    std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
    calc_score();
    std::cerr << std::endl;
    std::cerr << "fin" << std::endl;
}

void init_arr(){
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j)
            eval_arr[i][j] = 0.0;
    }
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j){
            rev_idxes[i][j] = calc_rev_idx(i, j);
        }
    }
}

void import_eval(string file){
    std::ifstream ifs(file);
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        return;
    }
    string line;
    for (int pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx){
        for (int pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            getline(ifs, line);
            eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
}

void import_test_data(int n_files, char *files[], int use_phase, double beta){
    int t = 0;
    FILE* fp;
    int16_t n_discs, phase, score, player;
    uint16_t raw_features[N_FEATURES];
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j)
            alpha[i][j] = 0.0;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx){
        std::cerr << std::endl << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (t < N_MAX_DATA){
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            phase = (n_discs - 4) / N_PHASE_DISCS;
            fread(&player, 2, 1, fp);
            fread(raw_features, 2, N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            if (phase == use_phase){
                if ((t & 0b1111111111111111) == 0b1111111111111111)
                    cerr << '\r' << t;
                Data data;
                for (int i = 0; i < N_FEATURES; ++i){
                    data.features[i] = raw_features[i];
                    alpha[feature_to_eval_idx[i]][raw_features[i]] += 1.0;
                    feature_idx_to_data[feature_to_eval_idx[i]][raw_features[i]].emplace_back(t);
                }
                data.score = (int)score;
                preds.emplace_back(0);
                test_data.emplace_back(data);
                ++t;
            }
        }
    }
    std::cerr << std::endl;
    std::cerr << t << " data loaded" << std::endl;
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j){
            double n_data_feature = alpha[i][j];
            if (rev_idxes[i][j] != j)
                n_data_feature += alpha[i][rev_idxes[i][j]];
            alpha[i][j] = beta / std::max(50.0, n_data_feature);
        }
    }
}

void output_param(){
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j)
            std::cout << (int)round(eval_arr[i][j]) << std::endl;
    }
    std::cerr << "output data fin" << std::endl;
}

int main(){
    init_arr();
    //import_eval("file.txt");
    char* files[] = {"data5_01.dat", "data5_02.dat", "data5_03.dat"};
    import_test_data(3, files, 20, 0.9);
    gradient_descent(60000);
    output_param();
    return 0;
}