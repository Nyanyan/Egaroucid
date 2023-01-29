#include <vector>
#include "evaluation_definition.hpp"

struct Data{
    uint16_t features[N_FEATURES];
    int score;
};

double eval_arr[N_EVAL][MAX_EVALUATE_IDX];
std::vector<Data> test_data;
std::vector<double> preds;
std::vector<std::vector<uint16_t>> used_idxes;
uint16_t rev_idxes[N_EVAL][MAX_EVALUATE_IDX];

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
    int round_pred;
    double double_data_size = (double)test_data.size();
    for (int i = 0; i < (int)test_data.size(); ++i){
        preds[i] = predict(i);
        round_pred = round_score(preds[i]);
        err = (double)abs(round_pred - test_data[i].score);
        mse += err / double_data_size * err;
        mae += err / double_data_size;
    }
    std::cerr << "\rmse " << mse << " mae " << mae << "                             ";
}

void next_step(){
    double err;
    for (int feature = 0; feature < N_FEATURES; ++feature){
        for (const int &idx: used_idxes[feature]){
            if (idx == rev_idxes[feature][idx]){
                err = calc_error(feature, idx);
                eval_arr[feature][idx] += 2.0 * alpha[feature][idx] * err;
            } else if (idx < rev_idxes[feature][idx]){
                err = calc_error(feature, idx) + err = calc_error(feature, rev_idxes[feature][idx]);
                eval_arr[feature][idx] += 2.0 * alpha[feature][idx] * err;
                eval_arr[feature][rev_idx] += 2.0 * alpha[feature][rev_idx] * err;
            }
        }
    }
}

void gradient_descent(uint64_t tl){
    uint64_t strt = tim();
    while (tim() - strt < tl){
        calc_score();
        next_step();
    }
    calc_score();
    std::cerr << std::endl;
}

void init_arr(){
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < MAX_EVALUATE_IDX; ++j)
            eval_arr[i][j] = 0.0;
    }
    for (int i = 0; i < N_EVAL; ++i){
        std::vector<uint16_t> tmp_vec;
        used_idxes.emplace_back(tmp_vec);
    }
    for (int i = 0; i < N_EVAL; ++i){
        for (int j = 0; j < MAX_EVALUATE_IDX; ++j){
            rev_idxes[i][j] = calc_rev_idx(i, j);
        }
    }
}

void import_eval(string file){
    ifstream ifs(file);
    if (ifs.fail()){
        cerr << "evaluation file not exist" << endl;
        return;
    }
    string line;
    int pattern_idx, pattern_elem, dense_idx, canput, sur0, sur1, i, j, k;
    for (pattern_idx = 0; pattern_idx < N_EVAL; ++pattern_idx){
        for (pattern_elem = 0; pattern_elem < eval_sizes[pattern_idx]; ++pattern_elem){
            getline(ifs, line);
            eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
}

void import_test_data(int n_files, char *files[]){

}


int main(){
    init_arr();
    import_eval("file.txt");
    import_test_data();
}