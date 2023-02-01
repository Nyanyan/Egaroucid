#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include "evaluation_definition.hpp"

struct Adj_Data{
    uint16_t features[ADJ_N_FEATURES];
    double score;
};

#define ADJ_N_MAX_DATA 70000000
#define ADJ_MAX_N_FILES 64

#define abs_error(x) std::abs(x)

double adj_eval_arr[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
int adj_alpha_occurance[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
double adj_alpha[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<Adj_Data> adj_test_data;
std::vector<double> adj_preds;
uint16_t adj_rev_idxes[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<std::vector<std::vector<int>>> adj_feature_idx_to_data(ADJ_N_EVAL, std::vector<std::vector<int>>(ADJ_MAX_EVALUATE_IDX, std::vector<int>(0)));

inline double adj_predict(int problem_idx){
    double res = 0.0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i)
        res += adj_eval_arr[adj_feature_to_eval_idx[i]][adj_test_data[problem_idx].features[i]];
    if (res > (double)SCORE_MAX * ADJ_STEP)
        res = (double)SCORE_MAX * ADJ_STEP;
    else if (res < -(double)SCORE_MAX * ADJ_STEP)
        res = -(double)SCORE_MAX * ADJ_STEP;
    return res;
}

int adj_round_score(double score){
    return (int)round(score / ADJ_STEP);
}

void adj_calc_score(){
    double err;
    double mse = 0.0;
    double mae = 0.0;
    double double_data_size = (double)adj_test_data.size();
    for (int i = 0; i < (int)adj_test_data.size(); ++i){
        adj_preds[i] = adj_predict(i) / ADJ_STEP;
        err = abs_error(adj_preds[i] - (double)adj_test_data[i].score);
        mse += err * err;
        mae += err;
    }
    mse /= double_data_size;
    mae /= double_data_size;
    std::cerr << " mse " << mse << " mae " << mae << "                             ";
}

void adj_print_result(int phase, int t, uint64_t tl){
    double err;
    double mse = 0.0;
    double mae = 0.0;
    double double_data_size = (double)adj_test_data.size();
    for (int i = 0; i < (int)adj_test_data.size(); ++i){
        adj_preds[i] = adj_predict(i) / ADJ_STEP;
        err = abs_error(adj_preds[i] - adj_test_data[i].score);
        mse += err * err;
        mae += err;
    }
    mse /= double_data_size;
    mae /= double_data_size;
    std::cout << phase << " " << t << " " << tl << " " << mse << " " << mae << std::endl;
}

double adj_calc_error(int feature, int idx){
    double res = 0.0;
    double raw_err;
    for (const int &i: adj_feature_idx_to_data[feature][idx]){
        raw_err = adj_test_data[i].score - adj_preds[i];
        res += raw_err;
    }
    return res;
}

void adj_next_step(){
    double err;
    for (int feature = 0; feature < ADJ_N_EVAL; ++feature){
        for (int idx = 0; idx < adj_eval_sizes[feature]; ++idx){
            if (idx == adj_rev_idxes[feature][idx]){
                err = adj_calc_error(feature, idx);
                adj_eval_arr[feature][idx] += 2.0 * adj_alpha[feature][idx] * err * ADJ_STEP;
            } else if (idx < adj_rev_idxes[feature][idx]){
                err = adj_calc_error(feature, idx) + adj_calc_error(feature, adj_rev_idxes[feature][idx]);
                adj_eval_arr[feature][idx] += 2.0 * adj_alpha[feature][idx] * err * ADJ_STEP;
                adj_eval_arr[feature][adj_rev_idxes[feature][idx]] += 2.0 * adj_alpha[feature][adj_rev_idxes[feature][idx]] * err * ADJ_STEP;
            }
        }
    }
}

void adj_gradient_descent(uint64_t tl, int phase){
    uint64_t strt = tim();
    int t = 0;
    adj_calc_score();
    std::cerr << std::endl;
    while (tim() - strt < tl){
        std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
        adj_calc_score();
        adj_next_step();
        ++t;
    }
    std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
    adj_calc_score();
    std::cerr << std::endl;
    adj_print_result(phase, t, tl);
    std::cerr << "fin" << std::endl;
}

void adj_init_arr(){
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            adj_eval_arr[i][j] = 0.0;
    }
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        for (int j = 0; j < adj_eval_sizes[i]; ++j){
            adj_rev_idxes[i][j] = adj_calc_rev_idx(i, j);
        }
    }
}

void adj_import_eval(std::string file){
    std::ifstream ifs(file);
    if (ifs.fail()){
        std::cerr << "evaluation file " << file << " not exist" << std::endl;
        return;
    }
    std::string line;
    for (int pattern_idx = 0; pattern_idx < ADJ_N_EVAL; ++pattern_idx){
        for (int pattern_elem = 0; pattern_elem < adj_eval_sizes[pattern_idx]; ++pattern_elem){
            getline(ifs, line);
            adj_eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
}

void adj_import_test_data(int n_files, char *files[], int use_phase, double beta){
    int t = 0;
    FILE* fp;
    int16_t n_discs, phase, score, player;
    uint16_t raw_features[ADJ_N_FEATURES];
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            adj_alpha_occurance[i][j] = 0;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx){
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (t < ADJ_N_MAX_DATA){
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            phase = (n_discs - 4) / ADJ_N_PHASE_DISCS;
            fread(&player, 2, 1, fp);
            fread(raw_features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            if (phase == use_phase){
                if ((t & 0b1111111111111111) == 0b1111111111111111)
                    std::cerr << '\r' << t;
                Adj_Data data;
                for (int i = 0; i < ADJ_N_FEATURES; ++i){
                    data.features[i] = raw_features[i];
                    adj_alpha_occurance[adj_feature_to_eval_idx[i]][raw_features[i]] += 1.0;
                    adj_feature_idx_to_data[adj_feature_to_eval_idx[i]][raw_features[i]].emplace_back(t);
                }
                data.score = (double)score;
                adj_preds.emplace_back(0);
                adj_test_data.emplace_back(data);
                ++t;
            }
        }
        std::cerr << '\r' << t << std::endl;
    }
    std::cerr << std::endl;
    std::cerr << t << " data loaded" << std::endl;
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        for (int j = 0; j < adj_eval_sizes[i]; ++j){
            int n_data_feature = adj_alpha_occurance[i][j];
            if (adj_rev_idxes[i][j] != j)
                n_data_feature += adj_alpha_occurance[i][adj_rev_idxes[i][j]];
            adj_alpha[i][j] = beta / (double)std::max(50, n_data_feature);
        }
    }
}

void adj_output_param(){
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            std::cout << (int)round(adj_eval_arr[i][j]) << std::endl;
    }
    std::cerr << "output data fin" << std::endl;
}

int main(int argc, char *argv[]){
    if (argc < 7){
        std::cerr << "input [phase] [hour] [minute] [second] [beta] [in_file] [test_data]" << std::endl;
        return 1;
    }
    if (argc - 6 >= ADJ_MAX_N_FILES){
        std::cerr << "too many train files" << std::endl;
        return 1;
    }
    int phase = atoi(argv[1]);
    uint64_t hour = atoi(argv[2]);
    uint64_t minute = atoi(argv[3]);
    uint64_t second = atoi(argv[4]);
    double beta = atof(argv[5]);
    std::string in_file = (std::string)argv[6];
    char* test_files[ADJ_MAX_N_FILES];
    for (int i = 6; i < argc; ++i)
        test_files[i - 6] = argv[i];
    second += minute * 60 + hour * 3600;
    adj_init_arr();
    adj_import_eval(in_file);
    adj_import_test_data(argc - 6, test_files, phase, beta);
    adj_gradient_descent(second * 1000, phase);
    adj_output_param();
    return 0;
}
