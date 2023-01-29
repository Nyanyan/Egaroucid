#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include "evaluation_definition.hpp"

struct Data{
    uint16_t features[N_FEATURES];
    double score;
};

#define N_MAX_DATA 70000000

double eval_arr[N_EVAL][MAX_EVALUATE_IDX];
double alpha[N_EVAL][MAX_EVALUATE_IDX];
std::vector<Data> test_data;
std::vector<double> preds;
uint16_t rev_idxes[N_EVAL][MAX_EVALUATE_IDX];
std::vector<std::vector<std::vector<int>>> feature_idx_to_data(N_EVAL, std::vector<std::vector<int>>(MAX_EVALUATE_IDX, std::vector<int>(0)));

inline double predict(int problem_idx){
    double res = 0.0;
    for (int i = 0; i < N_FEATURES; ++i)
        res += eval_arr[feature_to_eval_idx[i]][test_data[problem_idx].features[i]];
    if (res > (double)SCORE_MAX * STEP)
        res = (double)SCORE_MAX * STEP;
    else if (res < -(double)SCORE_MAX * STEP)
        res = -(double)SCORE_MAX * STEP;
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
        preds[i] = predict(i) / STEP;
        err = abs(preds[i] - (double)test_data[i].score);
        mse += err / double_data_size * err;
        mae += err / double_data_size;
    }
    std::cerr << " mse " << mse << " mae " << mae << "                             ";
}

void print_result(int phase, int t, uint64_t tl){
    double err;
    double mse = 0.0;
    double mae = 0.0;
    double double_data_size = (double)test_data.size();
    for (int i = 0; i < (int)test_data.size(); ++i){
        preds[i] = predict(i) / STEP;
        err = abs(preds[i] - test_data[i].score);
        mse += err / double_data_size * err;
        mae += err / double_data_size;
    }
    std::cout << phase << " " << t << " " << tl << " " << mse << " " << mae << std::endl;
}

double calc_error(int feature, int idx){
    double res = 0.0;
    double raw_err;
    for (const int &i: feature_idx_to_data[feature][idx]){
        raw_err = test_data[i].score - preds[i];
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

void gradient_descent(uint64_t tl, int phase){
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
    print_result(phase, t, tl);
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
    int alpha_occurance[N_EVAL][MAX_EVALUATE_IDX];
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < eval_sizes[i]; ++j)
            alpha_occurance[i][j] = 0;
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
                    alpha_occurance[feature_to_eval_idx[i]][raw_features[i]] += 1.0;
                    feature_idx_to_data[feature_to_eval_idx[i]][raw_features[i]].emplace_back(t);
                }
                data.score = (double)score;
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
            int n_data_feature = alpha_occurance[i][j];
            if (rev_idxes[i][j] != j)
                n_data_feature += alpha_occurance[i][rev_idxes[i][j]];
            alpha[i][j] = beta / (double)std::max(50, n_data_feature);
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

int main(int argc, char *argv[]){
    if (argc < 7){
        std::cerr << "input [phase] [hour] [minute] [second] [beta] [in_file] [test_data]" << std::endl;
        return 1;
    }
    int phase = atoi(argv[1]);
    uint64_t hour = atoi(argv[2]);
    uint64_t minute = atoi(argv[3]);
    uint64_t second = atoi(argv[4]);
    double beta = atof(argv[5]);
    string in_file = (string)argv[6];
    char* test_files[argc - 6];
    for (int i = 6; i < argc; ++i)
        test_files[i - 6] = argv[i];
    second += minute * 60 + hour * 3600;
    init_arr();
    import_eval(in_file);
    import_test_data(argc - 6, test_files, phase, beta);
    gradient_descent(second * 1000, phase);
    output_param();
    return 0;
}