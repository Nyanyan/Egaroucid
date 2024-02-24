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

// train data constant
#define ADJ_MAX_N_FILES 64
#define ADJ_MAX_N_DATA 100000000

// GPU constant
#define N_THREADS_PER_BLOCK_RESIDUAL 1024
#define N_THREADS_PER_BLOCK_NEXT_STEP 1024


// monitor constant
#define N_ERROR_MONITOR 2 // 0 for MSE, 1 for MAE


struct Adj_Data {
    int features[ADJ_N_FEATURES];
    float score;
};



/*
    @brief timing function

    @return time in milliseconds
*/
inline uint64_t tim(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void adj_init_arr(int eval_size, float *host_eval_arr, int *host_rev_idx_arr) {
    for (int i = 0; i < eval_size; ++i) {
        host_eval_arr[i] = 0.0;
    }
    int strt_idx = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            host_rev_idx_arr[strt_idx + j] = strt_idx + adj_calc_rev_idx(i, j);
        }
        strt_idx += adj_eval_sizes[i];
    }
}

void adj_import_eval(std::string file, int eval_size, float *host_eval_arr) {
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "evaluation file " << file << " not exist" << std::endl;
        return;
    }
    std::cerr << "importing eval params " << file << std::endl;
    std::string line;
    for (int i = 0; i < eval_size; ++i){
        if (!getline(ifs, line)) {
            std::cerr << "ERROR evaluation file broken" << std::endl;
            return;
        }
        host_eval_arr[i] = stof(line);
    }
}

int adj_import_train_data(int n_files, char* files[], Adj_Data* host_train_data, int *host_rev_idx_arr, int *host_n_appear_arr) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    uint16_t raw_features[ADJ_N_FEATURES];
    Adj_Data data;
    float score_avg = 0.0;
    int start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        if (i > 0){
            if (adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]){
                start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
            }
        }
        start_idx_arr[i] = start_idx;
    }
    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        std::cerr << files[file_idx] << std::endl;
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        while (n_data < ADJ_MAX_N_DATA) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            fread(&player, 2, 1, fp);
            fread(raw_features, 2, ADJ_N_FEATURES, fp);
            for (int i = 0; i < ADJ_N_FEATURES; ++i){
                host_train_data[n_data].features[i] = start_idx_arr[i] + raw_features[i];
                ++host_n_appear_arr[host_train_data[n_data].features[i]];
                ++host_n_appear_arr[host_rev_idx_arr[host_train_data[n_data].features[i]]];
            }
            fread(&score, 2, 1, fp);
            host_train_data[n_data].score = (float)score * ADJ_STEP;
            if ((n_data & 0xffff) == 0xffff)
                std::cerr << '\r' << n_data;
            score_avg += score;
            ++n_data;
        }
        fclose(fp);
        std::cerr << '\r' << n_data << std::endl;
    }
    score_avg /= n_data;
    std::cerr << std::endl;
    std::cerr << n_data << " data loaded" << std::endl;
    std::cerr << "score avg " << score_avg << std::endl;
    return n_data;
}

__global__ void adj_calculate_residual(const float *device_eval_arr, const int n_data, const Adj_Data *device_train_data, int *device_rev_idx_arr, float *device_residual_arr, float *device_error_monitor_arr){
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_data){
        return;
    }
    if (data_idx == 0){
        for (int i = 0; i < N_ERROR_MONITOR; ++i){
            device_error_monitor_arr[i] = 0.0;
        }
    }
    float predicted_value = 0.0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        predicted_value += device_eval_arr[device_train_data[data_idx].features[i]];
    }
    float residual_error = device_train_data[data_idx].score - predicted_value;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        atomicAdd(&device_residual_arr[device_train_data[data_idx].features[i]], residual_error);
        atomicAdd(&device_residual_arr[device_rev_idx_arr[device_train_data[data_idx].features[i]]], residual_error);
    }
    atomicAdd(&device_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_data);
    atomicAdd(&device_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_data);
}

/*
    @brief GD
*/
__global__ void adj_next_step_gd(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    device_eval_arr[eval_idx] += lr * grad;
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Momentum
*/
__global__ void adj_next_step_momentum(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_m_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    constexpr float beta1 = 0.9;
    device_m_arr[eval_idx] = beta1 * device_m_arr[eval_idx] + lr * grad;
    device_eval_arr[eval_idx] += device_m_arr[eval_idx];
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief AdaGrad
*/
__global__ void adj_next_step_adagrad(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    constexpr float beta2 = 0.999;
    constexpr float epsilon = 1e-7;
    device_v_arr[eval_idx] += grad * grad;
    device_eval_arr[eval_idx] += lr * grad / (sqrt(device_v_arr[eval_idx]) + epsilon);
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Adam
*/
__global__ void adj_next_step_adam(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_m_arr, float *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    constexpr float beta1 = 0.9;
    constexpr float beta2 = 0.999;
    constexpr float epsilon = 1e-7;
    float lrt = lr * sqrt(1.0 - pow(beta2, n_loop)) / (1.0 - pow(beta1, n_loop));
    device_m_arr[eval_idx] += (1.0 - beta1) * (grad - device_m_arr[eval_idx]);
    device_v_arr[eval_idx] += (1.0 - beta2) * (grad * grad - device_v_arr[eval_idx]);
    device_eval_arr[eval_idx] += lrt * device_m_arr[eval_idx] / (sqrt(device_v_arr[eval_idx]) + epsilon);
    device_residual_arr[eval_idx] = 0.0;
}

void adj_output_param(int eval_size, float *host_eval_arr) {
    for (int i = 0; i < eval_size; ++i) {
        std::cout << (int)round(host_eval_arr[i]) << std::endl;
    }
    std::cerr << "output data fin" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 7) {
        std::cerr << "input [phase] [hour] [minute] [second] [alpha] [n_patience] [in_file] [test_data]" << std::endl;
        return 1;
    }
    if (argc - 8 >= ADJ_MAX_N_FILES) {
        std::cerr << "too many train files" << std::endl;
        return 1;
    }
    int phase = atoi(argv[1]);
    uint64_t hour = atoi(argv[2]);
    uint64_t minute = atoi(argv[3]);
    uint64_t second = atoi(argv[4]);
    float alpha = atof(argv[5]);
    int n_patience = atoi(argv[6]);
    std::string in_file = (std::string)argv[7];
    char* test_files[ADJ_MAX_N_FILES];
    for (int i = 8; i < argc; ++i)
        test_files[i - 8] = argv[i];
    second += minute * 60 + hour * 3600;
    uint64_t msecond = second * 1000;

    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        eval_size += adj_eval_sizes[i];
    }
    std::cerr << "eval_size " << eval_size << std::endl;
    float *host_eval_arr = (float*)malloc(sizeof(float) * eval_size); // eval array
    int *host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size); // reversed index
    Adj_Data* host_train_data = (Adj_Data*)malloc(sizeof(Adj_Data) * ADJ_MAX_N_DATA); // train data
    int *host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    float *host_error_monitor_arr = (float*)malloc(sizeof(float) * N_ERROR_MONITOR);
    if (host_eval_arr == nullptr || host_rev_idx_arr == nullptr || host_train_data == nullptr){
        std::cerr << "cannot allocate memory" << std::endl;
        return 1;
    }
    adj_init_arr(eval_size, host_eval_arr, host_rev_idx_arr);
    adj_import_eval(in_file, eval_size, host_eval_arr);
    int n_data = adj_import_train_data(argc - 8, test_files, host_train_data, host_rev_idx_arr, host_n_appear_arr);
    for (int i = 0; i < eval_size; ++i){
        host_n_appear_arr[i] = std::min(100, host_n_appear_arr[i]);
    }

    float *device_eval_arr; // device eval array
    int *device_rev_idx_arr; // device reversed index
    Adj_Data *device_train_data; // device train data
    int *device_n_appear_arr;
    float *device_residual_arr;
    float *device_error_monitor_arr;
    cudaMalloc(&device_eval_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size);
    cudaMalloc(&device_train_data, sizeof(Adj_Data) * n_data);
    cudaMalloc(&device_n_appear_arr, sizeof(int) * eval_size);
    cudaMalloc(&device_residual_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR);
    cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(float) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_train_data, host_train_data, sizeof(Adj_Data) * n_data, cudaMemcpyHostToDevice);
    cudaMemcpy(device_n_appear_arr, host_n_appear_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice);
    cudaMemset(device_residual_arr, 0, sizeof(float) * eval_size);

    // for adam optimizer
    float *device_m_arr;
    float *device_v_arr;
    cudaMalloc(&device_m_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_v_arr, sizeof(float) * eval_size);
    cudaMemset(device_m_arr, 0, sizeof(float) * eval_size);
    cudaMemset(device_v_arr, 0, sizeof(float) * eval_size);
    
    const int n_blocks_residual = (n_data + N_THREADS_PER_BLOCK_RESIDUAL - 1) / N_THREADS_PER_BLOCK_RESIDUAL;
    const int n_blocks_next_step = (eval_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;
    std::cerr << "n_blocks_residual " << n_blocks_residual << " n_blocks_next_step " << n_blocks_next_step << std::endl;
    float alpha_stab = alpha; // / n_data;
    adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_data, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
    cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);
    std::cerr << "before MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << std::endl;
    uint64_t strt = tim();
    int n_loop = 0;
    while (tim() - strt < msecond){
        ++n_loop;
        adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_data, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
        cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);
        std::cerr << "\rn_loop " << n_loop << " progress " << (tim() - strt) * 100 / msecond << "% MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << "               ";
        
        // adj_next_step_gd <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab);
        // adj_next_step_momentum <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, n_loop);
        // adj_next_step_adagrad <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_v_arr, n_loop);
        adj_next_step_adam <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, device_v_arr, n_loop);
    }
    std::cerr << std::endl;

    cudaMemcpy(host_eval_arr, device_eval_arr, sizeof(float) * eval_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < eval_size; ++i){
        host_eval_arr[i] = round(host_eval_arr[i]);
    }

    adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_data, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
    cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);
    std::cout << "phase " << phase << " time " << (tim() - strt) << " ms data " << n_data << " n_loop " << n_loop << " MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << " (with int) alpha " << alpha << std::endl;

    adj_output_param(eval_size, host_eval_arr);
    return 0;
}
