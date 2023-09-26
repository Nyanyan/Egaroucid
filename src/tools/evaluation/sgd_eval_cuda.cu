
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include <numeric>
#include <iterator>
#include <random>
#include <algorithm>
#include <time.h>
#include <chrono>
#define OPTIMIZER_INCLUDE
#include "evaluation_definition.hpp"

#define ADJ_MAX_N_FILES 64

#define ADJ_N_MIN_DATA_FEATURES 200

#define ADJ_EVAL_MAX 4091

#define BATCH_THREAD_SIZE 256
#define BATCH_BLOCK_SIZE 256
#define BATCH_SIZE (BATCH_THREAD_SIZE * BATCH_BLOCK_SIZE)
//#ifndef N_FLOOR_UNIQUE_FEATURES
//    #define N_FLOOR_UNIQUE_FEATURES 16 // floorpow2(ADJ_N_EVAL)
//#endif
#define MAX_BATCH_DO_IDX (BATCH_SIZE / N_FLOOR_UNIQUE_FEATURES)

#define ADJ_PRINT_INTERVAL 16

/*
    @brief timing function

    @return time in milliseconds
*/
inline uint64_t tim(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
};

double adj_eval_arr[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
int adj_alpha_occurance[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
double adj_alpha[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<Adj_Data> adj_test_data;
std::vector<double> adj_preds;
uint16_t adj_rev_idxes[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];

__device__ double adj_predict_device(Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int problem_idx, int* device_strt_idxes) {
    double res = 0.0;
    int eval_idx;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        eval_idx = device_strt_idxes[device_feature_to_eval_idx[i]] + device_test_data[problem_idx].features[i];
        res += device_eval_arr[eval_idx];
    }
    res /= ADJ_STEP;
    if (res > (double)SCORE_MAX)
        res = (double)SCORE_MAX;
    else if (res < -(double)SCORE_MAX)
        res = -(double)SCORE_MAX;
    return res;
}

inline double adj_predict(int problem_idx) {
    double res = 0.0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        res += adj_eval_arr[adj_feature_to_eval_idx[i]][adj_test_data[problem_idx].features[i]];
    }
    res /= ADJ_STEP;
    if (res > (double)SCORE_MAX)
        res = (double)SCORE_MAX;
    else if (res < -(double)SCORE_MAX)
        res = -(double)SCORE_MAX;
    return res;
}

__global__ void adj_device_batch(double additional_learning_rate, bool mae_mse_calc, int* device_batch_random_idx, int* device_n_same_idx_in_feature, int* device_feature_first_idx, uint16_t* device_rev_idxes, double* device_eval_arr, double* device_alpha, int* device_eval_strts, Adj_Data* device_test_data, int n_test_data, int batch_first_idx, int* device_feature_to_eval_idx, double* device_mae_list, double* device_mse_list) {
    int thread_block_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int batch_idx = thread_block_idx + batch_first_idx;
    if (batch_idx >= n_test_data) {
        return;
    }
    int problem_idx = device_batch_random_idx[batch_idx];
    int i, j, feature, eval_idx, global_eval_idx;
    double err = adj_predict_device(device_test_data, device_eval_arr, device_feature_to_eval_idx, problem_idx, device_eval_strts) - device_test_data[problem_idx].score;
    double diff_base = additional_learning_rate * err * ADJ_STEP;
    if (mae_mse_calc) {
        device_mae_list[thread_block_idx] += abs(err);
        device_mse_list[thread_block_idx] += err * err;
    }
    for (int batch_do_idx = 0; batch_do_idx < MAX_BATCH_DO_IDX; ++batch_do_idx) {
        if (batch_do_idx == thread_block_idx / N_FLOOR_UNIQUE_FEATURES) {
            for (i = 0; i < ADJ_N_EVAL; ++i) {
                feature = (thread_block_idx + i) % ADJ_N_EVAL; // to avoid access collision
                for (j = 0; j < device_n_same_idx_in_feature[feature]; ++j) {
                    eval_idx = device_test_data[problem_idx].features[device_feature_first_idx[feature] + j];
                    global_eval_idx = device_eval_strts[feature] + eval_idx;
                    device_eval_arr[global_eval_idx] -= device_alpha[global_eval_idx] * diff_base;
                    if (eval_idx != device_rev_idxes[global_eval_idx]) {
                        device_eval_arr[device_eval_strts[feature] + device_rev_idxes[global_eval_idx]] -= device_alpha[global_eval_idx] * diff_base;
                    }
                }
            }
        }
    }
}

void adj_next_step(double additional_learning_rate, bool mae_mse_calc, int* device_batch_random_idx, int* device_n_same_idx_in_feature, int* device_feature_first_idx, uint16_t* device_rev_idxes, double* device_eval_arr, double* device_alpha, int* device_eval_strts, Adj_Data* device_data, int* device_feature_to_eval_idx, std::vector<int>& batch_random_idx, std::mt19937& engine, double* mae, double* mse) {
    int n_test_data = (int)adj_test_data.size();
    std::shuffle(batch_random_idx.begin(), batch_random_idx.end(), engine);
    int* src = &(batch_random_idx[0]);
    cudaMemcpy(device_batch_random_idx, src, adj_test_data.size() * sizeof(int), cudaMemcpyHostToDevice);
    int n_batch_tasks = ((int)adj_test_data.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    double* mae_list;
    double* mse_list;
    double* device_mae_list = nullptr;
    double* device_mse_list = nullptr;
    if (mae_mse_calc) {
        mae_list = (double*)malloc(BATCH_SIZE * sizeof(double));
        mse_list = (double*)malloc(BATCH_SIZE * sizeof(double));
        for (int i = 0; i < BATCH_SIZE; ++i) {
            mae_list[i] = 0.0;
            mse_list[i] = 0.0;
        }
        cudaMalloc(&device_mae_list, BATCH_SIZE * sizeof(double));
        cudaMalloc(&device_mse_list, BATCH_SIZE * sizeof(double));
        cudaMemcpy(device_mae_list, mae_list, BATCH_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_mse_list, mae_list, BATCH_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < n_batch_tasks; ++i) {
        adj_device_batch << <BATCH_BLOCK_SIZE, BATCH_THREAD_SIZE >> > (additional_learning_rate, mae_mse_calc, device_batch_random_idx, device_n_same_idx_in_feature, device_feature_first_idx, device_rev_idxes, device_eval_arr, device_alpha, device_eval_strts, device_data, (int)adj_test_data.size(), i * BATCH_SIZE, device_feature_to_eval_idx, device_mae_list, device_mse_list);
        cudaDeviceSynchronize();
    }
    if (mae_mse_calc) {
        cudaMemcpy(mae_list, device_mae_list, BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mse_list, device_mse_list, BATCH_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        *mae = 0.0;
        *mse = 0.0;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            *mae += mae_list[i];
            *mse += mse_list[i];
        }
        *mae /= adj_test_data.size();
        *mse /= adj_test_data.size();
        free(mae_list);
        free(mse_list);
        cudaFree(device_mae_list);
        cudaFree(device_mse_list);
    }
}

void adj_copy_train_data(Adj_Data** device_test_data) {
    cudaMalloc(&(*device_test_data), adj_test_data.size() * sizeof(Adj_Data));
    Adj_Data* src = &(adj_test_data[0]);
    cudaMemcpy(*device_test_data, src, adj_test_data.size() * sizeof(Adj_Data), cudaMemcpyHostToDevice);
}

void adj_copy_eval_arr(double** device_eval_arr, int malloc_size) {
    cudaMalloc(&(*device_eval_arr), malloc_size * sizeof(double));
    int strt = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        cudaMemcpy(&(*device_eval_arr)[strt], adj_eval_arr[i], adj_eval_sizes[i] * sizeof(double), cudaMemcpyHostToDevice);
        strt += adj_eval_sizes[i];
    }
}

void adj_get_eval_arr(double* device_eval_arr) {
    int strt = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        cudaMemcpy(adj_eval_arr[i], &device_eval_arr[strt], adj_eval_sizes[i] * sizeof(double), cudaMemcpyDeviceToHost);
        strt += adj_eval_sizes[i];
    }
}

void adj_copy_feature_to_eval_idx(int** device_feature_to_eval_idx) {
    cudaMalloc(&(*device_feature_to_eval_idx), ADJ_N_FEATURES * sizeof(int));
    cudaMemcpy(*device_feature_to_eval_idx, adj_feature_to_eval_idx, ADJ_N_FEATURES * sizeof(int), cudaMemcpyHostToDevice);
}

void adj_copy_rev_idxes(uint16_t** device_rev_idxes, int malloc_size) {
    cudaMalloc(&(*device_rev_idxes), malloc_size * sizeof(uint16_t));
    int strt = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        cudaMemcpy(&(*device_rev_idxes)[strt], adj_rev_idxes[i], adj_eval_sizes[i] * sizeof(uint16_t), cudaMemcpyHostToDevice);
        strt += adj_eval_sizes[i];
    }
}

void adj_copy_alpha(double** device_alpha, int malloc_size) {
    cudaMalloc(&(*device_alpha), malloc_size * sizeof(double));
    int strt = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        cudaMemcpy(&(*device_alpha)[strt], adj_alpha[i], adj_eval_sizes[i] * sizeof(double), cudaMemcpyHostToDevice);
        strt += adj_eval_sizes[i];
    }
}

void adj_copy_eval_strt_idx(int** device_eval_strt_idx) {
    int eval_strt_idx[ADJ_N_EVAL];
    eval_strt_idx[0] = 0;
    for (int i = 0; i < ADJ_N_EVAL - 1; ++i) {
        eval_strt_idx[i + 1] = eval_strt_idx[i] + adj_eval_sizes[i];
    }
    cudaMalloc(&(*device_eval_strt_idx), ADJ_N_EVAL * sizeof(int));
    cudaMemcpy(*device_eval_strt_idx, eval_strt_idx, ADJ_N_EVAL * sizeof(int), cudaMemcpyHostToDevice);
}

void adj_malloc_random_idx(int** batch_random_idx) {
    cudaMalloc(&(*batch_random_idx), adj_test_data.size() * sizeof(int));
}

void adj_copy_n_same_idx_in_feature(int** device_n_same_idx_in_feature) {
    int* lst = (int*)malloc(ADJ_N_EVAL * sizeof(int));
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        int elem = 0;
        for (int j = 0; j < ADJ_N_FEATURES; ++j) {
            if (adj_feature_to_eval_idx[j] == i) {
                ++elem;
            }
        }
        lst[i] = elem;
    }
    cudaMalloc(&(*device_n_same_idx_in_feature), ADJ_N_EVAL * sizeof(int));
    cudaMemcpy(*device_n_same_idx_in_feature, lst, ADJ_N_EVAL * sizeof(int), cudaMemcpyHostToDevice);
    free(lst);
}

void adj_copy_feature_first_idx(int** device_feature_first_idx) {
    int* lst = (int*)malloc(ADJ_N_EVAL * sizeof(int));
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        int elem = 0;
        for (int j = 0; j < ADJ_N_FEATURES; ++j) {
            if (adj_feature_to_eval_idx[j] == i) {
                lst[i] = j;
                break;
            }
        }
    }
    cudaMalloc(&(*device_feature_first_idx), ADJ_N_EVAL * sizeof(int));
    cudaMemcpy(*device_feature_first_idx, lst, ADJ_N_EVAL * sizeof(int), cudaMemcpyHostToDevice);
    free(lst);
}

void calc_mse_mae(double* mse, double* mae) {
    *mae = 0.0;
    *mse = 0.0;
    double err;
    for (int i = 0; i < (int)adj_test_data.size(); ++i) {
        err = std::abs(adj_predict(i) - adj_test_data[i].score);
        *mae += err;
        *mse += err * err;
    }
    *mse /= adj_test_data.size();
    *mae /= adj_test_data.size();
}

void adj_eval_round() {
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            adj_eval_arr[i][j] = round(adj_eval_arr[i][j]);
            if (adj_eval_arr[i][j] >= ADJ_EVAL_MAX) {
                adj_eval_arr[i][j] = ADJ_EVAL_MAX;
            }
            else if (adj_eval_arr[i][j] <= -ADJ_EVAL_MAX) {
                adj_eval_arr[i][j] = -ADJ_EVAL_MAX;
            }
        }
    }
}

void adj_stochastic_gradient_descent(uint64_t tl, int phase, double beta) {
    int n_eval_params = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        n_eval_params += adj_eval_sizes[i];
    }
    Adj_Data* device_test_data = nullptr;
    double* device_eval_arr = nullptr;
    int* device_feature_to_eval_idx = nullptr;
    uint16_t* device_rev_idxes = nullptr;
    double* device_alpha = nullptr;
    int* device_eval_strts = nullptr;
    int* device_batch_random_idx = nullptr;
    int* device_n_same_idx_in_feature = nullptr;
    int* device_feature_first_idx = nullptr;
    adj_copy_train_data(&device_test_data);
    adj_copy_eval_arr(&device_eval_arr, n_eval_params);
    adj_copy_feature_to_eval_idx(&device_feature_to_eval_idx);
    adj_copy_rev_idxes(&device_rev_idxes, n_eval_params);
    adj_copy_alpha(&device_alpha, n_eval_params);
    adj_copy_eval_strt_idx(&device_eval_strts);
    adj_malloc_random_idx(&device_batch_random_idx);
    adj_copy_n_same_idx_in_feature(&device_n_same_idx_in_feature);
    adj_copy_feature_first_idx(&device_feature_first_idx);
    cudaDeviceSynchronize();
    std::cerr << "data copied to GPU" << std::endl;
    std::vector<int> batch_random_idx(adj_test_data.size());
    std::iota(batch_random_idx.begin(), batch_random_idx.end(), 0);
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    double mae, mse;
    uint64_t strt = tim();
    int t = 0;
    calc_mse_mae(&mse, &mae);
    std::cerr << "first mse " << mse << " mae " << mae << std::endl;
    bool mae_mse_calc;
    double min_mse = 10000000.0;
    double min_mae = 10000000.0;
    double additional_learn_rate = 1.0;
    int no_better_mse_count = 0;
    while (tim() - strt < tl) {
        mae_mse_calc = (t & (ADJ_PRINT_INTERVAL - 1)) == 0;
        adj_next_step(additional_learn_rate, mae_mse_calc, device_batch_random_idx, device_n_same_idx_in_feature, device_feature_first_idx, device_rev_idxes, device_eval_arr, device_alpha, device_eval_strts, device_test_data, device_feature_to_eval_idx, batch_random_idx, engine, &mae, &mse);
        if (mae_mse_calc) {
            if (mse < min_mse) {
                min_mse = mse;
                min_mae = mae;
                no_better_mse_count = 0;
            }
            else {
                ++no_better_mse_count;
                if (no_better_mse_count >= 16) {
                    additional_learn_rate *= 0.75;
                    no_better_mse_count = 0;
                }
            }
            std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl << " mse " << mse << " (" << min_mse << ") " << " mae " << mae << " (" << min_mae << ")  no_better " << no_better_mse_count << " a_lr " << additional_learn_rate << "  ";
        }
        ++t;
    }
    std::cerr << std::endl << "done" << std::endl;
    adj_get_eval_arr(device_eval_arr);
    adj_eval_round();
    calc_mse_mae(&mse, &mae);
    std::cout << "phase " << phase << " tl " << tl / 1000 << " data " << adj_test_data.size() << " n " << t << " mse " << mse << " mae " << mae << " beta " << beta << " alr " << additional_learn_rate << std::endl;
    cudaFree(device_test_data);
    cudaFree(device_eval_arr);
    cudaFree(device_feature_to_eval_idx);
    cudaFree(device_rev_idxes);
    cudaFree(device_alpha);
    cudaFree(device_eval_strts);
    cudaFree(device_batch_random_idx);
    cudaFree(device_n_same_idx_in_feature);
    cudaFree(device_feature_first_idx);
    std::cerr << "fin" << std::endl;
}

void adj_init_arr() {
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            adj_eval_arr[i][j] = 0.0;
    }
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            adj_rev_idxes[i][j] = adj_calc_rev_idx(i, j);
        }
    }
}

void adj_import_eval(std::string file) {
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "evaluation file " << file << " not exist" << std::endl;
        return;
    }
    std::cerr << "importing eval params " << file << std::endl;
    std::string line;
    for (int pattern_idx = 0; pattern_idx < ADJ_N_EVAL; ++pattern_idx) {
        for (int pattern_elem = 0; pattern_elem < adj_eval_sizes[pattern_idx]; ++pattern_elem) {
            if (!getline(ifs, line)) {
                std::cerr << "ERROR evaluation file broken" << std::endl;
                return;
            }
            adj_eval_arr[pattern_idx][pattern_elem] = stof(line);
        }
    }
}

void adj_import_test_data(int n_files, char* files[], int use_phase, double beta) {
    int t = 0, i, j;
    FILE* fp;
    int16_t n_discs, score, player;
    Adj_Data data;
    for (i = 0; i < ADJ_N_EVAL; ++i) {
        for (j = 0; j < adj_eval_sizes[i]; ++j)
            adj_alpha_occurance[i][j] = 0;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (true) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            //phase = (n_discs - 4) / ADJ_N_PHASE_DISCS;
            fread(&player, 2, 1, fp);
            fread(&(data.features[0]), 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            //if (phase == use_phase) {
            if ((t & 0b1111111111111111) == 0b1111111111111111)
                std::cerr << '\r' << t;
            for (i = 0; i < ADJ_N_FEATURES; ++i) {
                ++adj_alpha_occurance[adj_feature_to_eval_idx[i]][data.features[i]];
            }
            data.score = (double)score;
            adj_preds.emplace_back(0);
            adj_test_data.emplace_back(data);
            ++t;
            //}
        }
        std::cerr << '\r' << t << std::endl;
    }
    std::cerr << std::endl;
    std::cerr << t << " data loaded" << std::endl;
    for (i = 0; i < ADJ_N_EVAL; ++i) {
        for (j = 0; j < adj_eval_sizes[i]; ++j) {
            int n_data_feature = adj_alpha_occurance[i][j];
            if (adj_rev_idxes[i][j] != j){
                n_data_feature += adj_alpha_occurance[i][adj_rev_idxes[i][j]];
                n_data_feature /= 2.0;
            }
            adj_alpha[i][j] = beta / (double)std::max(ADJ_N_MIN_DATA_FEATURES, n_data_feature);
        }
    }
}

void adj_output_param() {
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            std::cout << (int)round(adj_eval_arr[i][j]) << std::endl;
    }
    std::cerr << "output data fin" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "input [phase] [hour] [minute] [second] [beta] [in_file] [test_data]" << std::endl;
        return 1;
    }
    if (argc - 7 >= ADJ_MAX_N_FILES) {
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
    for (int i = 7; i < argc; ++i)
        test_files[i - 7] = argv[i];
    second += minute * 60 + hour * 3600;
    adj_init_arr();
    adj_import_eval(in_file);
    adj_import_test_data(argc - 7, test_files, phase, beta);
    adj_stochastic_gradient_descent(second * 1000, phase, beta);
    adj_output_param();
    return 0;
}
