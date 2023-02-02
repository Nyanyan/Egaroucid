
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include "evaluation_definition.hpp"

#define ADJ_N_MAX_DATA 70000000
#define ADJ_MAX_N_FILES 64
#define ADJ_MAX_N_DATA_SAME_IDX 100000

#define N_BLOCKS_SCORE 16384
#define N_THREADS_SCORE 512

#define N_BLOCKS_SUM 1
#define N_THREADS_SUM 1

#define N_BLOCKS_STEP 16384
#define N_THREADS_STEP 128

#define ADJ_IDX_UNDEFINED -1

#define abs_error(x) std::abs(x)

struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
};

struct Adj_Feature_to_data {
    int idx;
    Adj_Feature_to_data* next;
};

double adj_eval_arr[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
int adj_alpha_occurance[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
double adj_alpha[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<Adj_Data> adj_test_data;
std::vector<double> adj_preds;
uint16_t adj_rev_idxes[ADJ_N_EVAL][ADJ_MAX_EVALUATE_IDX];
std::vector<std::vector<std::vector<int>>> adj_feature_idx_to_data(ADJ_N_EVAL, std::vector<std::vector<int>>(ADJ_MAX_EVALUATE_IDX, std::vector<int>(0)));

__device__ double adj_predict_device(Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int idx, int* device_strt_idxes) {
    double res = 0.0;
    int all_idx;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        all_idx = device_strt_idxes[device_feature_to_eval_idx[i]] + device_test_data[idx].features[i];
        res += device_eval_arr[all_idx];
    }
    if (res > (double)SCORE_MAX * ADJ_STEP)
        res = (double)SCORE_MAX * ADJ_STEP;
    else if (res < -(double)SCORE_MAX * ADJ_STEP)
        res = -(double)SCORE_MAX * ADJ_STEP;
    return res;
}

__device__ double adj_device_round_score(double score) {
    int res;
    if (score >= 0.0) {
        res = (int)(score + 0.5);
    }
    else {
        res = (int)(score - 0.5);
    }
    return (double)res;
}

__global__ void adj_device_calc_score1(double* mse, double* mae, double* device_preds, Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int* device_strt_idxes, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= data_size) {
        return;
    }
    device_preds[idx] = adj_predict_device(device_test_data, device_eval_arr, device_feature_to_eval_idx, idx, device_strt_idxes) / ADJ_STEP;
    //double err = abs_error(device_preds[idx] - device_test_data[idx].score);
    //mse[idx] = err * err;
    //mae[idx] = err;
}

__device__ int adj_clp2(int xx) {
    unsigned int x = (unsigned int)xx - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return (int)(x + 1);
}

__global__ void adj_device_sum_mse_mae(double* mse, double* mae, int data_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /*
    if (idx >= data_size) {
        return;
    }
    int nidx;
    int log2_size = __clz(adj_clp2(data_size));
    for (int i = 0; i < log2_size; ++i) {
        if (idx % (1 << (i + 1))) {
            break;
        }
        nidx = idx + (1 << i);
        if (nidx < data_size) {
            mse[idx] += mse[nidx];
            mae[idx] += mae[nidx];
        }
    }
    */
    if (idx == 0) {
        double mse_sum = 0.0, mae_sum = 0.0;
        for (int i = 0; i < data_size; ++i) {
            mse_sum += mse[i];
            mae_sum += mae[i];
        }
        mse[0] = mse_sum / data_size;
        mae[0] = mae_sum / data_size;
    }
}

void adj_device_calc_score_p(double* mse, double* mae, double* device_preds, Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int* device_strt_idxes, double* mse_sum, double* mae_sum) {
    int data_size = (int)adj_test_data.size();
    adj_device_calc_score1 << <N_BLOCKS_SCORE, N_THREADS_SCORE >> > (mse, mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_strt_idxes, data_size);
    cudaDeviceSynchronize();
    //adj_device_sum_mse_mae << <N_BLOCKS_SUM, N_THREADS_SUM >> > (mse, mae, data_size);
    //cudaDeviceSynchronize();
    //cudaMemcpy(mse_sum, mse, sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(mae_sum, mae, sizeof(double), cudaMemcpyDeviceToHost);
    double* preds_copy = (double*)malloc(data_size * sizeof(double));
    cudaMemcpy(preds_copy, device_preds, data_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double err;
    *mse_sum = 0.0;
    *mae_sum = 0.0;
    for (int i = 0; i < data_size; ++i) {
        err = abs_error(preds_copy[i] - adj_test_data[i].score);
        *mse_sum += err * err;
        *mae_sum += err;
    }
    *mse_sum /= data_size;
    *mae_sum /= data_size;
    free(preds_copy);
}

void adj_calc_score(double* mse, double* mae, double* device_preds, Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int* device_strt_idxes) {
    double mse_sum, mae_sum;
    adj_device_calc_score_p(mse, mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_strt_idxes, &mse_sum, &mae_sum);
    std::cerr << " mse " << mse_sum << " mae " << mae_sum << "                             ";
}

void adj_print_result(int phase, int t, uint64_t tl, double* mse, double* mae, double* device_preds, Adj_Data* device_test_data, double* device_eval_arr, int* device_feature_to_eval_idx, int* device_strt_idxes) {
    double mse_sum, mae_sum;
    adj_device_calc_score_p(mse, mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_strt_idxes, &mse_sum, &mae_sum);
    std::cout << phase << " " << tl / 1000 << " " << adj_test_data.size() << " " << t << " " << mse_sum << " " << mae_sum << std::endl;
}

__global__ void adj_device_calc_error1(int* device_eval_sizes, int* device_eval_strts, Adj_Feature_to_data* device_feature_idx_to_data, Adj_Data* device_data, double* device_preds, double* device_sum_error, int n_test_data) {
    int all_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int feature = -1, idx = -1;
    for (int i = ADJ_N_EVAL - 1; i >= 0; --i) {
        if (all_idx >= device_eval_strts[i]) {
            feature = i;
            idx = all_idx - device_eval_strts[i];
            break;
        }
    }
    if (feature == -1) {
        return;
    }
    if (idx >= device_eval_sizes[feature]) {
        return;
    }
    double res = 0.0;
    if (device_feature_idx_to_data[all_idx].idx != ADJ_IDX_UNDEFINED) {
        Adj_Feature_to_data* feature_to_data = &device_feature_idx_to_data[all_idx];
        while (feature_to_data != nullptr) {
            res += device_data[feature_to_data->idx].score - device_preds[feature_to_data->idx];
            feature_to_data = feature_to_data->next;
        }
    }
    device_sum_error[all_idx] = res;
}

__global__ void adj_device_next_step1(uint16_t* device_rev_idxes, double* device_eval_arr, double* device_alpha, int* device_eval_sizes, int* device_eval_strts, double* device_sum_error, int n_test_data) {
    int all_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int feature = -1, idx = -1;
    for (int i = ADJ_N_EVAL - 1; i >= 0; --i) {
        if (all_idx >= device_eval_strts[i]) {
            feature = i;
            idx = all_idx - device_eval_strts[i];
            break;
        }
    }
    if (feature == -1) {
        return;
    }
    if (idx >= device_eval_sizes[feature]) {
        return;
    }
    double err = device_sum_error[all_idx];
    if (idx != device_rev_idxes[all_idx]) {
        err += device_sum_error[device_eval_strts[feature] + device_rev_idxes[all_idx]];
    }
    device_eval_arr[all_idx] += 2.0 * device_alpha[all_idx] * err * ADJ_STEP;
}

void adj_next_step(uint16_t* device_rev_idxes, double* device_eval_arr, double* device_alpha, int* device_eval_sizes, int* device_eval_strts, Adj_Feature_to_data* device_feature_idx_to_data, Adj_Data* device_data, double* device_preds, double* device_sum_error) {
    int n_test_data = (int)adj_test_data.size();
    adj_device_calc_error1 << <N_BLOCKS_STEP, N_THREADS_STEP >> > (device_eval_sizes, device_eval_strts, device_feature_idx_to_data, device_data, device_preds, device_sum_error, n_test_data);
    cudaDeviceSynchronize();
    adj_device_next_step1 << <N_BLOCKS_STEP, N_THREADS_STEP >> > (device_rev_idxes, device_eval_arr, device_alpha, device_eval_sizes, device_eval_strts, device_sum_error, n_test_data);
    cudaDeviceSynchronize();
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

void adj_malloc_preds(double** device_preds) {
    cudaMalloc(&(*device_preds), adj_test_data.size() * sizeof(double));
}

void adj_copy_feature_to_eval_idx(int** device_feature_to_eval_idx) {
    cudaMalloc(&(*device_feature_to_eval_idx), ADJ_N_FEATURES * sizeof(int));
    cudaMemcpy(*device_feature_to_eval_idx, adj_feature_to_eval_idx, ADJ_N_FEATURES * sizeof(int), cudaMemcpyHostToDevice);
}

void adj_malloc_mse_mae(double** device_mse_mae) {
    cudaMalloc(&(*device_mse_mae), adj_test_data.size() * sizeof(double));
}

void adj_copy_feature_idx_to_data(Adj_Feature_to_data** device_feature_idx_to_data, Adj_Feature_to_data** device_feature_linked_data, int malloc_size) {
    int n_max_linked_data = (ADJ_N_FEATURES + 1) * adj_test_data.size();
    cudaMalloc(&(*device_feature_idx_to_data), malloc_size * sizeof(Adj_Feature_to_data));
    cudaMalloc(&(*device_feature_linked_data), n_max_linked_data * sizeof(Adj_Feature_to_data));
    Adj_Feature_to_data* feature_idx_to_data = (Adj_Feature_to_data*)malloc(malloc_size * sizeof(Adj_Feature_to_data));
    Adj_Feature_to_data* feature_linked_data = (Adj_Feature_to_data*)malloc(n_max_linked_data * sizeof(Adj_Feature_to_data));
    int all_idx = 0;
    int linked_data_idx = 0;
    Adj_Feature_to_data* device_latest_link = *device_feature_linked_data;
    int n_same_idx;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            n_same_idx = (int)adj_feature_idx_to_data[i][j].size();
            if (n_same_idx == 0) {
                feature_idx_to_data[all_idx].idx = ADJ_IDX_UNDEFINED;
                feature_idx_to_data[all_idx].next = nullptr;
            }
            else if (n_same_idx == 1) {
                feature_idx_to_data[all_idx].idx = adj_feature_idx_to_data[i][j][0];
                feature_idx_to_data[all_idx].next = nullptr;
            }
            else {
                feature_idx_to_data[all_idx].idx = adj_feature_idx_to_data[i][j][0];
                feature_idx_to_data[all_idx].next = device_latest_link;
                for (int k = 1; k < n_same_idx; ++k) {
                    feature_linked_data[linked_data_idx].idx = adj_feature_idx_to_data[i][j][k];
                    feature_linked_data[linked_data_idx].next = nullptr;
                    if (k != 1) {
                        feature_linked_data[linked_data_idx - 1].next = device_latest_link;
                    }
                    ++device_latest_link;
                    ++linked_data_idx;
                    if (linked_data_idx >= n_max_linked_data) {
                        std::cerr << "ERROR too many linked data" << std::endl;
                        exit(1);
                    }
                    
                }
            }
            ++all_idx;
        }
    }
    std::cerr << linked_data_idx << " additional data used, allocated " << n_max_linked_data << std::endl;
    cudaMemcpy(*device_feature_idx_to_data, feature_idx_to_data, malloc_size * sizeof(Adj_Feature_to_data), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_feature_linked_data, feature_linked_data, n_max_linked_data * sizeof(Adj_Feature_to_data), cudaMemcpyHostToDevice);
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

void adj_copy_eval_sizes(int** device_eval_sizes) {
    cudaMalloc(&(*device_eval_sizes), ADJ_N_EVAL * sizeof(int));
    cudaMemcpy(*device_eval_sizes, adj_eval_sizes, ADJ_N_EVAL * sizeof(int), cudaMemcpyHostToDevice);
}

void adj_malloc_sum_error(double** device_sum_error, int malloc_size) {
    cudaMalloc(&(*device_sum_error), malloc_size * sizeof(double));
}

void adj_gradient_descent(uint64_t tl, int phase) {
    int n_eval_params = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        n_eval_params += adj_eval_sizes[i];
    }
    Adj_Data* device_test_data = nullptr;
    double* device_eval_arr = nullptr;
    double* device_preds = nullptr;
    int* device_feature_to_eval_idx = nullptr;
    double* device_mse = nullptr;
    double* device_mae = nullptr;
    Adj_Feature_to_data* device_feature_idx_to_data = nullptr;
    Adj_Feature_to_data* device_feature_linked_data = nullptr;
    uint16_t* device_rev_idxes = nullptr;
    double* device_alpha = nullptr;
    int* device_eval_strts = nullptr;
    int* device_eval_sizes = nullptr;
    double* device_sum_error = nullptr;
    adj_copy_train_data(&device_test_data);
    adj_copy_eval_arr(&device_eval_arr, n_eval_params);
    adj_malloc_preds(&device_preds);
    adj_copy_feature_to_eval_idx(&device_feature_to_eval_idx);
    adj_malloc_mse_mae(&device_mse);
    adj_malloc_mse_mae(&device_mae);
    adj_copy_feature_idx_to_data(&device_feature_idx_to_data, &device_feature_linked_data, n_eval_params);
    adj_copy_rev_idxes(&device_rev_idxes, n_eval_params);
    adj_copy_alpha(&device_alpha, n_eval_params);
    adj_copy_eval_strt_idx(&device_eval_strts);
    adj_copy_eval_sizes(&device_eval_sizes);
    adj_malloc_sum_error(&device_sum_error, n_eval_params);
    cudaDeviceSynchronize();
    std::cerr << "data copied to GPU" << std::endl;

    uint64_t strt = tim();
    int t = 0;
    adj_calc_score(device_mse, device_mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_eval_strts);
    std::cerr << std::endl;
    while (tim() - strt < tl) {
        std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
        adj_calc_score(device_mse, device_mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_eval_strts);
        adj_next_step(device_rev_idxes, device_eval_arr, device_alpha, device_eval_sizes, device_eval_strts, device_feature_idx_to_data, device_test_data, device_preds, device_sum_error);
        ++t;
    }
    std::cerr << '\r' << t << " " << (tim() - strt) * 1000 / tl;
    adj_calc_score(device_mse, device_mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_eval_strts);
    std::cerr << std::endl;
    adj_print_result(phase, t, tl, device_mse, device_mae, device_preds, device_test_data, device_eval_arr, device_feature_to_eval_idx, device_eval_strts);
    adj_get_eval_arr(device_eval_arr);
    std::cerr << "done" << std::endl;

    cudaFree(device_test_data);
    cudaFree(device_eval_arr);
    cudaFree(device_preds);
    cudaFree(device_feature_to_eval_idx);
    cudaFree(device_mse);
    cudaFree(device_mae);
    cudaFree(device_feature_idx_to_data);
    cudaFree(device_feature_linked_data);
    cudaFree(device_rev_idxes);
    cudaFree(device_alpha);
    cudaFree(device_eval_strts);
    cudaFree(device_eval_sizes);
    cudaFree(device_sum_error);
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
    std::string line;
    for (int pattern_idx = 0; pattern_idx < ADJ_N_EVAL; ++pattern_idx) {
        for (int pattern_elem = 0; pattern_elem < adj_eval_sizes[pattern_idx]; ++pattern_elem) {
            getline(ifs, line);
            adj_eval_arr[pattern_idx][pattern_elem] = stoi(line);
        }
    }
}

void adj_import_test_data(int n_files, char* files[], int use_phase, double beta) {
    int t = 0;
    FILE* fp;
    int16_t n_discs, phase, score, player;
    uint16_t raw_features[ADJ_N_FEATURES];
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j)
            adj_alpha_occurance[i][j] = 0;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (t < ADJ_N_MAX_DATA) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            phase = (n_discs - 4) / ADJ_N_PHASE_DISCS;
            fread(&player, 2, 1, fp);
            fread(raw_features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            if (phase == use_phase) {
                if ((t & 0b1111111111111111) == 0b1111111111111111)
                    std::cerr << '\r' << t;
                Adj_Data data;
                for (int i = 0; i < ADJ_N_FEATURES; ++i) {
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
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            int n_data_feature = adj_alpha_occurance[i][j];
            if (adj_rev_idxes[i][j] != j)
                n_data_feature += adj_alpha_occurance[i][adj_rev_idxes[i][j]];
            adj_alpha[i][j] = beta / (double)std::max(50, n_data_feature);
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
    if (argc - 6 >= ADJ_MAX_N_FILES) {
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
