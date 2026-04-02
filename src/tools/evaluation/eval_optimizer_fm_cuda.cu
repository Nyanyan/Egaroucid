/*
    Egaroucid Project

    @file eval_optimizer_cuda_fm.cu
        Evaluation Function Optimizer with Factorization Machine in CUDA
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err__) << std::endl; \
        return 1; \
    } \
} while (0)

#define CUDA_CHECK_LAUNCH() do { \
    cudaError_t err__ = cudaGetLastError(); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[CUDA KERNEL ERROR] " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(err__) << std::endl; \
        return 1; \
    } \
} while (0)

// settings
#define RESIDUAL_USE_CLIP false
#define USE_WARMUP false

#define OPTIMIZER_INCLUDE
#include "evaluation_definition.hpp"

// train data constant
#define ADJ_MAX_N_FILES 200
#if ADJ_CELL_WEIGHT
    #define ADJ_MAX_N_DATA 1000000
#else
    #define ADJ_MAX_N_DATA 200000000
#endif

// FM constants
#define ADJ_FM_DIM 4
#define ADJ_FM_INIT_STDDEV 0.01

// training constant
#define ADJ_IGNORE_N_APPEAR 0

// GPU constant
#define N_THREADS_PER_BLOCK_TEST 256
#define N_THREADS_PER_BLOCK_RESIDUAL 256
#define N_THREADS_PER_BLOCK_NEXT_STEP 1024

// monitor constant
#define N_ERROR_MONITOR 2
#define N_TEST_ERROR_MONITOR 2


struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
};


inline uint64_t tim() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

__host__ __device__ inline bool get_feature_index_and_value(
    const Adj_Data& data,
    const int* start_idx_arr,
    int feature_pos,
    int& idx,
    double& x_val) {
#if ADJ_CELL_WEIGHT
    uint16_t feature = data.features[feature_pos];
    if (feature < 10) {
        idx = (int)feature;
        x_val = 1.0;
        return true;
    }
    if (feature < 20) {
        idx = (int)feature - 10;
        x_val = -1.0;
        return true;
    }
    return false;
#else
    idx = start_idx_arr[feature_pos] + (int)data.features[feature_pos];
    x_val = 1.0;
    return true;
#endif
}

void adj_init_arr(
    int eval_size,
    double* host_factor_arr,
    int* host_rev_idx_arr,
    int* host_n_appear_arr) {
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::normal_distribution<double> normal_dist(0.0, ADJ_FM_INIT_STDDEV);

    for (int i = 0; i < eval_size; ++i) {
        host_n_appear_arr[i] = 0;
    }
    for (int i = 0; i < eval_size * ADJ_FM_DIM; ++i) {
        host_factor_arr[i] = normal_dist(engine);
    }

    int strt_idx = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        for (int j = 0; j < adj_eval_sizes[i]; ++j) {
            host_rev_idx_arr[strt_idx + j] = strt_idx + adj_calc_rev_idx(i, j);
        }
        strt_idx += adj_eval_sizes[i];
    }
}

void adj_import_eval_fm(
    const std::string& file,
    int eval_size,
    double* host_factor_arr) {
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "evaluation file " << file << " not exist, start from initialized params" << std::endl;
        return;
    }
    std::cerr << "importing fm params " << file << std::endl;

    std::vector<double> raw;
    raw.reserve((size_t)eval_size * (size_t)(ADJ_FM_DIM + 1));
    double value;
    while (ifs >> value) {
        raw.emplace_back(value);
    }
    const int fm_size = eval_size * ADJ_FM_DIM;
    const size_t expected_full = (size_t)eval_size + (size_t)fm_size;
    const size_t expected_factor_only = (size_t)fm_size;

    if (raw.size() == expected_full) {
        for (int i = 0; i < fm_size; ++i) {
            host_factor_arr[i] = raw[(size_t)eval_size + (size_t)i];
        }
        std::cerr << "imported legacy linear+factor format, factor " << fm_size << " / " << fm_size << std::endl;
        return;
    }
    if (raw.size() == expected_factor_only) {
        for (int i = 0; i < fm_size; ++i) {
            host_factor_arr[i] = raw[(size_t)i];
        }
        std::cerr << "imported factor-only format, factor " << fm_size << " / " << fm_size << std::endl;
        return;
    }

    std::cerr << "[ERROR] invalid fm param length found=" << raw.size()
              << " expected=" << expected_full
              << " or " << expected_factor_only << std::endl;
}

int adj_import_data(int n_files, char* files[], Adj_Data* host_train_data, double* score_avg_out) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    double score_avg = 0.0;

    for (int file_idx = 0; file_idx < n_files; ++file_idx) {
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        int n_data_before = n_data;
        while (n_data < ADJ_MAX_N_DATA) {
            if (fread(&n_discs, 2, 1, fp) < 1) {
                break;
            }
            fread(&player, 2, 1, fp);
            fread(host_train_data[n_data].features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            host_train_data[n_data].score = (double)score * ADJ_STEP;
            score_avg += score;
            ++n_data;
        }
        fclose(fp);
        if (n_data_before < n_data) {
            std::cerr << files[file_idx] << " " << n_data << std::endl;
        }
    }
    if (n_data > 0) {
        score_avg /= n_data;
    }
    if (score_avg_out != nullptr) {
        *score_avg_out = score_avg;
    }
    std::cerr << "score avg " << score_avg << std::endl;
    return n_data;
}

__global__ void adj_calculate_residual_fm(
    const double* device_factor_arr,
    const int n_data,
    const int* device_start_idx_arr,
    const Adj_Data* device_train_data,
    const int* device_rev_idx_arr,
    double* device_residual_factor_arr,
    double* device_error_monitor_arr) {
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_data) {
        return;
    }

    int idx_arr[ADJ_N_FEATURES];
    double x_arr[ADJ_N_FEATURES];
    int used_n = 0;
    double predicted_value = 0.0;

    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        int idx;
        double x_val;
        if (get_feature_index_and_value(device_train_data[data_idx], device_start_idx_arr, i, idx, x_val)) {
            idx_arr[used_n] = idx;
            x_arr[used_n] = x_val;
            ++used_n;
        }
    }

    double sum_vx[ADJ_FM_DIM];
    double sum_vx2[ADJ_FM_DIM];
    for (int f = 0; f < ADJ_FM_DIM; ++f) {
        sum_vx[f] = 0.0;
        sum_vx2[f] = 0.0;
    }
    for (int i = 0; i < used_n; ++i) {
        const int idx = idx_arr[i];
        const double x_val = x_arr[i];
        for (int f = 0; f < ADJ_FM_DIM; ++f) {
            const double vif = device_factor_arr[idx * ADJ_FM_DIM + f];
            const double vx = vif * x_val;
            sum_vx[f] += vx;
            sum_vx2[f] += vx * vx;
        }
    }
    for (int f = 0; f < ADJ_FM_DIM; ++f) {
        predicted_value += 0.5 * (sum_vx[f] * sum_vx[f] - sum_vx2[f]);
    }

#if RESIDUAL_USE_CLIP
    if (predicted_value > HW2 * ADJ_STEP) {
        predicted_value = HW2 * ADJ_STEP;
    } else if (predicted_value < -HW2 * ADJ_STEP) {
        predicted_value = -HW2 * ADJ_STEP;
    }
#endif

    const double residual_error = device_train_data[data_idx].score - predicted_value;

    for (int i = 0; i < used_n; ++i) {
        const int idx = idx_arr[i];
        const int rev_idx = device_rev_idx_arr[idx];
        const double x_val = x_arr[i];

        for (int f = 0; f < ADJ_FM_DIM; ++f) {
            const double vif = device_factor_arr[idx * ADJ_FM_DIM + f];
            const double grad_coef = residual_error * x_val * (sum_vx[f] - vif * x_val);
            atomicAdd(&device_residual_factor_arr[idx * ADJ_FM_DIM + f], grad_coef);
            atomicAdd(&device_residual_factor_arr[rev_idx * ADJ_FM_DIM + f], grad_coef);
        }
    }

    atomicAdd(&device_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_data);
    atomicAdd(&device_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_data);
}

__global__ void adj_calculate_val_loss_fm(
    const double* device_factor_arr,
    const int n_val_data,
    const int* device_start_idx_arr,
    const Adj_Data* device_val_data,
    double* device_val_error_monitor_arr) {
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_val_data) {
        return;
    }

    int idx_arr[ADJ_N_FEATURES];
    double x_arr[ADJ_N_FEATURES];
    int used_n = 0;
    double predicted_value = 0.0;

    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        int idx;
        double x_val;
        if (get_feature_index_and_value(device_val_data[data_idx], device_start_idx_arr, i, idx, x_val)) {
            idx_arr[used_n] = idx;
            x_arr[used_n] = x_val;
            ++used_n;
        }
    }

    for (int f = 0; f < ADJ_FM_DIM; ++f) {
        double sum_vx = 0.0;
        double sum_vx2 = 0.0;
        for (int i = 0; i < used_n; ++i) {
            const double vx = device_factor_arr[idx_arr[i] * ADJ_FM_DIM + f] * x_arr[i];
            sum_vx += vx;
            sum_vx2 += vx * vx;
        }
        predicted_value += 0.5 * (sum_vx * sum_vx - sum_vx2);
    }

#if RESIDUAL_USE_CLIP
    if (predicted_value > HW2 * ADJ_STEP) {
        predicted_value = HW2 * ADJ_STEP;
    } else if (predicted_value < -HW2 * ADJ_STEP) {
        predicted_value = -HW2 * ADJ_STEP;
    }
#endif

    const double residual_error = device_val_data[data_idx].score - predicted_value;
    atomicAdd(&device_val_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_val_data);
    atomicAdd(&device_val_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_val_data);
}

__global__ void adam_factor(
    const int phase,
    const int eval_size,
    double* device_factor_arr,
    const int* device_n_appear_arr,
    double* device_residual_factor_arr,
    const double alpha_stab,
    double* device_m_factor_arr,
    double* device_v_factor_arr,
    const int n_loop) {
    const int factor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int fm_size = eval_size * ADJ_FM_DIM;
    if (factor_idx >= fm_size) {
        return;
    }

    const int eval_idx = factor_idx / ADJ_FM_DIM;
    if (device_n_appear_arr[eval_idx] > ADJ_IGNORE_N_APPEAR || (phase <= 11 && device_n_appear_arr[eval_idx] > 0)) {
        const double lr = alpha_stab / (double)device_n_appear_arr[eval_idx];
        const double grad = 2.0 * device_residual_factor_arr[factor_idx];
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-7;
        const double lrt = lr * sqrt(1.0 - pow(beta2, n_loop)) / (1.0 - pow(beta1, n_loop));
        device_m_factor_arr[factor_idx] += (1.0 - beta1) * (grad - device_m_factor_arr[factor_idx]);
        device_v_factor_arr[factor_idx] += (1.0 - beta2) * (grad * grad - device_v_factor_arr[factor_idx]);
        device_factor_arr[factor_idx] += lrt * device_m_factor_arr[factor_idx] / (sqrt(device_v_factor_arr[factor_idx]) + epsilon);
    }
    device_residual_factor_arr[factor_idx] = 0.0;
}

void adj_output_param_fm(
    int phase,
    int eval_size,
    const double* host_factor_arr) {
    const std::string filename = std::string("trained/") + std::to_string(phase) + "_fm.txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << std::endl;
        return;
    }
    ofs.setf(std::ios::fixed);
    ofs.precision(10);
    for (int i = 0; i < eval_size * ADJ_FM_DIM; ++i) {
        ofs << host_factor_arr[i] << '\n';
    }
    ofs.close();
    std::cerr << "FM params output to " << filename << std::endl;
}

void adj_output_weight(int phase, int eval_size, const int* weight_arr) {
    const std::string filename = std::string("trained/weight_") + std::to_string(phase) + "_fm.txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << std::endl;
        return;
    }
    for (int i = 0; i < eval_size; ++i) {
        ofs << weight_arr[i] << '\n';
    }
    ofs.close();
    std::cerr << "weight output to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    std::cerr << "FM dim " << ADJ_FM_DIM << std::endl;

    if (argc < 10) {
        std::cerr << "input [phase] [hour] [minute] [second] [alpha] [n_patience] [reduce_lr_patience] [reduce_lr_ratio] [in_file] [train_data...]" << std::endl;
        return 1;
    }
    if (argc - 10 >= ADJ_MAX_N_FILES) {
        std::cerr << "too many train files" << std::endl;
        return 1;
    }

    const int phase = atoi(argv[1]);
    uint64_t hour = atoi(argv[2]);
    uint64_t minute = atoi(argv[3]);
    uint64_t second = atoi(argv[4]);
    double alpha = atof(argv[5]);
    const double alpha_in = alpha;
    const int n_patience = atoi(argv[6]);
    const int reduce_lr_patience = atoi(argv[7]);
    const double reduce_lr_ratio = atof(argv[8]);
    const std::string in_file = (std::string)argv[9];

    char* train_files[ADJ_MAX_N_FILES];
    const int n_train_data_file = argc - 10;
    for (int i = 0; i < n_train_data_file; ++i) {
        train_files[i] = argv[i + 10];
    }

    second += minute * 60 + hour * 3600;
    const uint64_t msecond = second * 1000;

    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        eval_size += adj_eval_sizes[i];
    }
    const int fm_size = eval_size * ADJ_FM_DIM;
    std::cerr << "eval_size " << eval_size << " fm_size " << fm_size << std::endl;

    double* host_factor_arr = (double*)malloc(sizeof(double) * fm_size);
    int* host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size);
    Adj_Data* host_train_data = (Adj_Data*)malloc(sizeof(Adj_Data) * ADJ_MAX_N_DATA);
    int* host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    int* weight_arr = (int*)malloc(sizeof(int) * eval_size);
    double* host_error_monitor_arr = (double*)malloc(sizeof(double) * N_ERROR_MONITOR);
    double* host_val_error_monitor_arr = (double*)malloc(sizeof(double) * N_TEST_ERROR_MONITOR);
    if (host_factor_arr == nullptr || host_rev_idx_arr == nullptr ||
        host_train_data == nullptr || host_n_appear_arr == nullptr || host_error_monitor_arr == nullptr ||
        host_val_error_monitor_arr == nullptr) {
        std::cerr << "cannot allocate memory" << std::endl;
        return 1;
    }

    adj_init_arr(eval_size, host_factor_arr, host_rev_idx_arr, host_n_appear_arr);
    adj_import_eval_fm(in_file, eval_size, host_factor_arr);

    double score_avg = 0.0;
    const int n_all_data = adj_import_data(n_train_data_file, train_files, host_train_data, &score_avg);
    if (n_all_data <= 0) {
        std::cerr << "no training data" << std::endl;
        return 1;
    }
    std::cerr << n_all_data << " data loaded" << std::endl;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::shuffle(host_train_data, host_train_data + n_all_data, engine);
    std::cerr << "data shuffled" << std::endl;

    int n_val_data;
    int n_train_data;
    Adj_Data* host_val_data;
    if (phase > 11) {
        n_val_data = (int)(n_all_data * 0.05);
        if (n_val_data <= 0) {
            n_val_data = 1;
        }
        n_train_data = n_all_data - n_val_data;
        host_val_data = host_train_data + n_train_data;
    } else {
        n_val_data = n_all_data;
        n_train_data = n_all_data;
        host_val_data = host_train_data;
    }
    std::cerr << "n_train_data " << n_train_data << " n_val_data " << n_val_data << std::endl;

    int host_start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        if (i > 0 && adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
        host_start_idx_arr[i] = start_idx;
    }

    for (int data_idx = 0; data_idx < n_train_data; ++data_idx) {
        for (int i = 0; i < ADJ_N_FEATURES; ++i) {
            int idx;
            double x_val;
            if (!get_feature_index_and_value(host_train_data[data_idx], host_start_idx_arr, i, idx, x_val)) {
                continue;
            }
            ++host_n_appear_arr[idx];
            ++host_n_appear_arr[host_rev_idx_arr[idx]];
        }
    }
    for (int i = 0; i < eval_size; ++i) {
        weight_arr[i] = host_n_appear_arr[i];
        host_n_appear_arr[i] = std::min(50, host_n_appear_arr[i]);
    }
    std::cerr << "train data appearance calculated" << std::endl;

    double* device_factor_arr;
    int* device_rev_idx_arr;
    Adj_Data* device_train_data;
    Adj_Data* device_val_data;
    int* device_n_appear_arr;
    double* device_residual_factor_arr;
    double* device_error_monitor_arr;
    double* device_val_error_monitor_arr;
    int* device_start_idx_arr;

    CUDA_CHECK(cudaMalloc(&device_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_train_data, sizeof(Adj_Data) * n_train_data));
    CUDA_CHECK(cudaMalloc(&device_val_data, sizeof(Adj_Data) * n_val_data));
    CUDA_CHECK(cudaMalloc(&device_n_appear_arr, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_residual_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR));
    CUDA_CHECK(cudaMalloc(&device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR));
    CUDA_CHECK(cudaMalloc(&device_start_idx_arr, sizeof(int) * ADJ_N_FEATURES));

    CUDA_CHECK(cudaMemcpy(device_factor_arr, host_factor_arr, sizeof(double) * fm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_train_data, host_train_data, sizeof(Adj_Data) * n_train_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_val_data, host_val_data, sizeof(Adj_Data) * n_val_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_n_appear_arr, host_n_appear_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_residual_factor_arr, 0, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemcpy(device_start_idx_arr, host_start_idx_arr, sizeof(int) * ADJ_N_FEATURES, cudaMemcpyHostToDevice));

    double* device_m_factor_arr;
    double* device_v_factor_arr;
    CUDA_CHECK(cudaMalloc(&device_m_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_v_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemset(device_m_factor_arr, 0, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemset(device_v_factor_arr, 0, sizeof(double) * fm_size));

    const int n_blocks_val = (n_val_data + N_THREADS_PER_BLOCK_TEST - 1) / N_THREADS_PER_BLOCK_TEST;
    const int n_blocks_residual = (n_train_data + N_THREADS_PER_BLOCK_RESIDUAL - 1) / N_THREADS_PER_BLOCK_RESIDUAL;
    const int n_blocks_next_factor = (fm_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;

    std::cerr << "n_blocks_val " << n_blocks_val
              << " n_blocks_residual " << n_blocks_residual
              << " n_blocks_next_factor " << n_blocks_next_factor << std::endl;

    std::cerr << "phase " << phase << std::endl;
    CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
    adj_calculate_residual_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
        device_factor_arr,
        n_train_data,
        device_start_idx_arr,
        device_train_data,
        device_rev_idx_arr,
        device_residual_factor_arr,
        device_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));
    std::cerr << "before MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << std::endl;

    const uint64_t strt = tim();
    int n_loop = 0;
    double min_val_mse = 100000000.0;
    int n_val_loss_increase = 0;
    int n_val_loss_increase_reduce_lr = 0;
#if USE_WARMUP
    double alpha_stab = alpha / 5.0;
#else
    double alpha_stab = alpha;
#endif

    while (tim() - strt < msecond) {
        ++n_loop;

        CUDA_CHECK(cudaMemset(device_val_error_monitor_arr, 0, sizeof(double) * N_TEST_ERROR_MONITOR));
        adj_calculate_val_loss_fm<<<n_blocks_val, N_THREADS_PER_BLOCK_TEST>>>(
            device_factor_arr,
            n_val_data,
            device_start_idx_arr,
            device_val_data,
            device_val_error_monitor_arr);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(host_val_error_monitor_arr, device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR, cudaMemcpyDeviceToHost));

        if (host_val_error_monitor_arr[0] <= min_val_mse) {
            min_val_mse = host_val_error_monitor_arr[0];
            n_val_loss_increase = 0;
            n_val_loss_increase_reduce_lr = 0;
        } else {
            ++n_val_loss_increase;
            ++n_val_loss_increase_reduce_lr;
            if (n_val_loss_increase > n_patience) {
                break;
            }
        }

        CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
        adj_calculate_residual_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
            device_factor_arr,
            n_train_data,
            device_start_idx_arr,
            device_train_data,
            device_rev_idx_arr,
            device_residual_factor_arr,
            device_error_monitor_arr);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));

        std::cerr << "\rn_loop " << n_loop
              << " progress " << (tim() - strt) * 100 / msecond << "%"
              << " MSE " << host_error_monitor_arr[0]
              << " MAE " << host_error_monitor_arr[1]
              << "  val_MSE " << host_val_error_monitor_arr[0]
              << " val_MAE " << host_val_error_monitor_arr[1]
              << " val_loss_inc " << n_val_loss_increase
              << " alpha " << alpha_stab
              << "                    ";

        adam_factor<<<n_blocks_next_factor, N_THREADS_PER_BLOCK_NEXT_STEP>>>(
            phase,
            eval_size,
            device_factor_arr,
            device_n_appear_arr,
            device_residual_factor_arr,
            alpha_stab,
            device_m_factor_arr,
            device_v_factor_arr,
            n_loop);
        CUDA_CHECK_LAUNCH();

#if USE_WARMUP
        if (alpha_stab < alpha) {
            alpha_stab += alpha / 50.0;
        }
#endif
        if (n_val_loss_increase_reduce_lr >= reduce_lr_patience) {
            alpha *= reduce_lr_ratio;
            n_val_loss_increase_reduce_lr = 0;
        }
#if USE_WARMUP
        if (alpha_stab > alpha) {
            alpha_stab = alpha;
        }
#else
        alpha_stab = alpha;
#endif
    }
    std::cerr << std::endl;

    CUDA_CHECK(cudaMemcpy(host_factor_arr, device_factor_arr, sizeof(double) * fm_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
    adj_calculate_residual_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
        device_factor_arr,
        n_train_data,
        device_start_idx_arr,
        device_train_data,
        device_rev_idx_arr,
        device_residual_factor_arr,
        device_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(device_val_error_monitor_arr, 0, sizeof(double) * N_TEST_ERROR_MONITOR));
    adj_calculate_val_loss_fm<<<n_blocks_val, N_THREADS_PER_BLOCK_TEST>>>(
        device_factor_arr,
        n_val_data,
        device_start_idx_arr,
        device_val_data,
        device_val_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_val_error_monitor_arr, device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR, cudaMemcpyDeviceToHost));

    adj_output_param_fm(phase, eval_size, host_factor_arr);
    adj_output_weight(phase, eval_size, weight_arr);

    std::cerr << "phase " << phase
              << " time " << (tim() - strt) << " ms"
              << " n_train_data " << n_train_data
              << " n_val_data " << n_val_data
              << " score_avg " << score_avg
              << " n_loop " << n_loop
              << " MSE " << host_error_monitor_arr[0]
              << " MAE " << host_error_monitor_arr[1]
              << " val_MSE " << host_val_error_monitor_arr[0]
              << " val_MAE " << host_val_error_monitor_arr[1]
              << " (FM float) alpha " << alpha_in
              << " n_patience " << n_patience
              << " reduce_lr_patience " << reduce_lr_patience
              << " reduce_lr_ratio " << reduce_lr_ratio
              << std::endl;
    std::cout << "phase " << phase
              << " time " << (tim() - strt) << " ms"
              << " n_train_data " << n_train_data
              << " n_val_data " << n_val_data
              << " score_avg " << score_avg
              << " n_loop " << n_loop
              << " MSE " << host_error_monitor_arr[0]
              << " MAE " << host_error_monitor_arr[1]
              << " val_MSE " << host_val_error_monitor_arr[0]
              << " val_MAE " << host_val_error_monitor_arr[1]
              << " (FM float) alpha " << alpha_in
              << " n_patience " << n_patience
              << " reduce_lr_patience " << reduce_lr_patience
              << " reduce_lr_ratio " << reduce_lr_ratio
              << std::endl;

    return 0;
}
