/*
    Egaroucid Project

    @file eval_optimizer_cuda.cu
        Evaluation Function Optimizer in CUDA
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

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
#if ADJ_CELL_WEIGHT
    #define ADJ_MAX_N_DATA 1000000
#else
    #define ADJ_MAX_N_DATA 100000000
#endif
#define ADJ_MAX_N_TEST_DATA 100000

// GPU constant
#define N_THREADS_PER_BLOCK_RESIDUAL 1024
#define N_THREADS_PER_BLOCK_NEXT_STEP 1024


// monitor constant
#define N_ERROR_MONITOR 2 // 0 for MSE, 1 for MAE


struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    float score;
};



/*
    @brief timing function

    @return time in milliseconds
*/
inline uint64_t tim(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

/*
    @brief initialize some arrays
*/
void adj_init_arr(int eval_size, float *host_eval_arr, int *host_rev_idx_arr, int *host_n_appear_arr) {
    for (int i = 0; i < eval_size; ++i) {
        host_eval_arr[i] = 0.0;
        host_n_appear_arr[i] = 0;
    }
    int strt_idx = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            host_rev_idx_arr[strt_idx + j] = strt_idx + adj_calc_rev_idx(i, j);
        }
        strt_idx += adj_eval_sizes[i];
    }
}

/*
    @brief import pre-calculated evaluation function
*/
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

/*
    @brief import train data
*/
int adj_import_data(int n_files, char* files[], Adj_Data* host_train_data, int *host_rev_idx_arr, int *host_n_appear_arr) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
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
            fread(host_train_data[n_data].features, 2, ADJ_N_FEATURES, fp);
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

/*
    @brief calculate residual error
*/
__global__ void adj_calculate_residual(const float *device_eval_arr, const int n_data, const int *device_start_idx_arr, const Adj_Data *device_train_data, int *device_rev_idx_arr, float *device_residual_arr, float *device_error_monitor_arr){
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
        #if ADJ_CELL_WEIGHT
            if (device_train_data[data_idx].features[i] < 10){
                predicted_value += device_eval_arr[device_train_data[data_idx].features[i]];
            } else if (device_train_data[data_idx].features[i] < 20){
                predicted_value -= device_eval_arr[device_train_data[data_idx].features[i] - 10];
            }
        #else
            predicted_value += device_eval_arr[device_start_idx_arr[i] + (int)device_train_data[data_idx].features[i]];
        #endif
    }
    float residual_error = device_train_data[data_idx].score - predicted_value;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        #if ADJ_CELL_WEIGHT
            if (device_train_data[data_idx].features[i] < 10){
                atomicAdd(&device_residual_arr[device_train_data[data_idx].features[i]], residual_error);
                atomicAdd(&device_residual_arr[device_rev_idx_arr[device_train_data[data_idx].features[i]]], residual_error);
            } else if (device_train_data[data_idx].features[i] < 20){
                atomicAdd(&device_residual_arr[device_train_data[data_idx].features[i] - 10], -residual_error);
                atomicAdd(&device_residual_arr[device_rev_idx_arr[device_train_data[data_idx].features[i] - 10]], -residual_error);
            }
        #else
            atomicAdd(&device_residual_arr[device_start_idx_arr[i] + (int)device_train_data[data_idx].features[i]], residual_error);
            int rev_idx = device_rev_idx_arr[device_start_idx_arr[i] + (int)device_train_data[data_idx].features[i]];
            //if (rev_idx != device_start_idx_arr[i] + (int)device_train_data[data_idx].features[i])
            atomicAdd(&device_residual_arr[rev_idx], residual_error);
        #endif
    }
    atomicAdd(&device_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_data);
    atomicAdd(&device_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_data);
}

/*
    @brief Gradient Descent Optimizer
*/
__global__ void gradient_descent(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        device_eval_arr[eval_idx] += lr * grad;
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Momentum Optimizer
*/
__global__ void momentum(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_m_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        constexpr float beta1 = 0.9;
        device_m_arr[eval_idx] = beta1 * device_m_arr[eval_idx] + lr * grad;
        device_eval_arr[eval_idx] += device_m_arr[eval_idx];
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief AdaGrad Optimizer
*/
__global__ void adagrad(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        constexpr float beta2 = 0.999;
        constexpr float epsilon = 1e-7;
        device_v_arr[eval_idx] += grad * grad;
        device_eval_arr[eval_idx] += lr * grad / (sqrt(device_v_arr[eval_idx]) + epsilon);
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Adam Optimizer
*/
__global__ void adam(const int eval_size, float *device_eval_arr, int *device_n_appear_arr, float *device_residual_arr, float alpha_stab, float *device_m_arr, float *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    float lr = alpha_stab / device_n_appear_arr[eval_idx];
    float grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        constexpr float beta1 = 0.9;
        constexpr float beta2 = 0.999;
        constexpr float epsilon = 1e-7;
        float lrt = lr * sqrt(1.0 - pow(beta2, n_loop)) / (1.0 - pow(beta1, n_loop));
        device_m_arr[eval_idx] += (1.0 - beta1) * (grad - device_m_arr[eval_idx]);
        device_v_arr[eval_idx] += (1.0 - beta2) * (grad * grad - device_v_arr[eval_idx]);
        device_eval_arr[eval_idx] += lrt * device_m_arr[eval_idx] / (sqrt(device_v_arr[eval_idx]) + epsilon);
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Output Parameters as integer
*/
void adj_output_param(int eval_size, float *host_eval_arr) {
    for (int i = 0; i < eval_size; ++i) {
        std::cout << (int)round(host_eval_arr[i]) << std::endl;
    }
    std::cerr << "output data fin" << std::endl;
}

/*
    @brief calculate test loss with CPU
*/
void test_loss(float *eval_arr, int *host_start_idx_arr, int n_data, Adj_Data *data, float *mse, float *mae){
    *mse = 0.0;
    *mae = 0.0;
    for (int i = 0; i < n_data; ++i){
        int score = 0;
        for (int j = 0; j < ADJ_N_FEATURES; ++j){
            score += eval_arr[host_start_idx_arr[j] + data[i].features[j]];
        }
        /*
        score += score >= 0 ? ADJ_STEP_2 : -ADJ_STEP_2;
        score /= ADJ_STEP;
        if (score < -SCORE_MAX)
            score = -SCORE_MAX;
        if (score > SCORE_MAX)
            score = SCORE_MAX;
        float abs_error = fabs(data[i].score - score);
        */
        float abs_error = fabs(data[i].score * ADJ_STEP - score) / ADJ_STEP;
        *mse += abs_error * abs_error;
        *mae += abs_error;
    }
    *mse /= n_data;
    *mae /= n_data;
}

int main(int argc, char* argv[]) {
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 8) {
        std::cerr << "input [phase] [hour] [minute] [second] [alpha] [n_patience] [in_file] [train_data...]" << std::endl;
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
    char* train_files[ADJ_MAX_N_FILES];
    int n_train_data_file = argc - 8;
    for (int i = 0; i < n_train_data_file; ++i)
        train_files[i] = argv[i + 8];
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
    adj_init_arr(eval_size, host_eval_arr, host_rev_idx_arr, host_n_appear_arr);
    adj_import_eval(in_file, eval_size, host_eval_arr);
    int n_all_data = adj_import_data(n_train_data_file, train_files, host_train_data, host_rev_idx_arr, host_n_appear_arr);
    // shuffle data
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::shuffle(host_train_data, host_train_data + n_all_data, engine);
    // divide data
    int n_test_data = n_all_data * 0.05; // use 5% as test data
    int n_train_data = n_all_data - n_test_data;
    Adj_Data* host_test_data = host_train_data + n_train_data;
    std::cerr << "n_train_data " << n_train_data << " n_test_data " << n_test_data << std::endl;
    // calculate n_appear of train data
    int host_start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        if (i > 0){
            if (adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]){
                start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
            }
        }
        host_start_idx_arr[i] = start_idx;
    }
    for (int data_idx = 0; data_idx < n_train_data; ++data_idx){
        for (int i = 0; i < ADJ_N_FEATURES; ++i){
            #if ADJ_CELL_WEIGHT
                if (host_train_data[data_idx].features[i] < 10){
                    ++host_n_appear_arr[host_train_data[data_idx].features[i]];
                    ++host_n_appear_arr[host_rev_idx_arr[host_train_data[data_idx].features[i]]];
                } else if (host_train_data[data_idx].features[i] < 20){
                    ++host_n_appear_arr[host_train_data[data_idx].features[i] - 10];
                    ++host_n_appear_arr[host_rev_idx_arr[host_train_data[data_idx].features[i] - 10]];
                }
            #else
                ++host_n_appear_arr[host_start_idx_arr[i] + (int)host_train_data[data_idx].features[i]];
                int rev_idx = host_rev_idx_arr[host_start_idx_arr[i] + (int)host_train_data[data_idx].features[i]];
                //if (rev_idx != start_idx_arr[i] + (int)host_train_data[data_idx].features[i])
                ++host_n_appear_arr[rev_idx];
            #endif
        }
    }
    for (int i = 0; i < eval_size; ++i){
        host_n_appear_arr[i] = std::min(100, host_n_appear_arr[i]);
    }

    float *device_eval_arr; // device eval array
    int *device_rev_idx_arr; // device reversed index
    Adj_Data *device_train_data; // device train data
    int *device_n_appear_arr;
    float *device_residual_arr;
    float *device_error_monitor_arr;
    int *device_start_idx_arr;
    cudaMalloc(&device_eval_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size);
    cudaMalloc(&device_train_data, sizeof(Adj_Data) * n_train_data);
    cudaMalloc(&device_n_appear_arr, sizeof(int) * eval_size);
    cudaMalloc(&device_residual_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR);
    cudaMalloc(&device_start_idx_arr, sizeof(int) * ADJ_N_FEATURES);
    cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(float) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_train_data, host_train_data, sizeof(Adj_Data) * n_train_data, cudaMemcpyHostToDevice);
    cudaMemcpy(device_n_appear_arr, host_n_appear_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice);
    cudaMemset(device_residual_arr, 0, sizeof(float) * eval_size);
    cudaMemcpy(device_start_idx_arr, host_start_idx_arr, sizeof(int) * ADJ_N_FEATURES, cudaMemcpyHostToDevice);

    // for adam optimizer
    float *device_m_arr;
    float *device_v_arr;
    cudaMalloc(&device_m_arr, sizeof(float) * eval_size);
    cudaMalloc(&device_v_arr, sizeof(float) * eval_size);
    cudaMemset(device_m_arr, 0, sizeof(float) * eval_size);
    cudaMemset(device_v_arr, 0, sizeof(float) * eval_size);
    
    const int n_blocks_residual = (n_train_data + N_THREADS_PER_BLOCK_RESIDUAL - 1) / N_THREADS_PER_BLOCK_RESIDUAL;
    const int n_blocks_next_step = (eval_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;
    std::cerr << "n_blocks_residual " << n_blocks_residual << " n_blocks_next_step " << n_blocks_next_step << std::endl;
    std::cerr << "phase " << phase << std::endl;
    float alpha_stab = alpha; // / n_data;
    adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_train_data, device_start_idx_arr, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
    cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);
    std::cerr << "before MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << std::endl;
    uint64_t strt = tim();
    int n_loop = 0;
    float test_mse, test_mae, min_test_mse, min_test_mae;
    int n_test_loss_increase = 0;
    test_loss(host_eval_arr, host_start_idx_arr, n_test_data, host_test_data, &min_test_mse, &min_test_mae);
    while (tim() - strt < msecond){
        ++n_loop;
        cudaMemcpy(host_eval_arr, device_eval_arr, sizeof(float) * eval_size, cudaMemcpyDeviceToHost);
        test_loss(host_eval_arr, host_start_idx_arr, n_test_data, host_test_data, &test_mse, &test_mae);
        if (test_mse <= min_test_mse){
            min_test_mse = test_mse;
            n_test_loss_increase = 0;
        } else{
            ++n_test_loss_increase;
            if (n_test_loss_increase > n_patience){
                break;
            }
        }

        adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_train_data, device_start_idx_arr, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
        cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);

        std::cerr << "\rn_loop " << n_loop << " progress " << (tim() - strt) * 100 / msecond << "% MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << "  test_MSE " << test_mse << " test_MAE " << test_mae << " loss_inc " << n_test_loss_increase << "                    ";
        
        // gradient_descent <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab);
        // momentum <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, n_loop);
        // adagrad <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_v_arr, n_loop);
        adam <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, device_v_arr, n_loop);
    }
    std::cerr << std::endl;

    // round eval
    cudaMemcpy(host_eval_arr, device_eval_arr, sizeof(float) * eval_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < eval_size; ++i){
        host_eval_arr[i] = round(host_eval_arr[i]);
    }
    cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(float) * eval_size, cudaMemcpyHostToDevice);

    // calculate final loss
    adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_train_data, device_start_idx_arr, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
    cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(float) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost);
    test_loss(host_eval_arr, host_start_idx_arr, n_test_data, host_test_data, &test_mse, &test_mae);
    std::cout << "phase " << phase << " time " << (tim() - strt) << " ms n_train_data " << n_train_data << " n_test_data " << n_test_data << " n_loop " << n_loop << " MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << " test_MSE " << test_mse << " test_MAE " << test_mae << " (with int) alpha " << alpha << " n_patience " << n_patience << std::endl;

    adj_output_param(eval_size, host_eval_arr);
    return 0;
}
