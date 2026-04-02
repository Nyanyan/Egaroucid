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
#define ROUND_USE_CLIP true
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
#define ADJ_FM_DIM 6
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

__device__ inline void atomic_add_fp64(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        const double next = __longlong_as_double(assumed) + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(next));
    } while (assumed != old);
}

void adj_init_arr(
    int eval_size,
    double* host_linear_arr,
    double* host_factor_arr,
    int* host_rev_idx_arr,
    int* host_n_appear_arr) {
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::normal_distribution<double> normal_dist(0.0, ADJ_FM_INIT_STDDEV);

    for (int i = 0; i < eval_size; ++i) {
        host_linear_arr[i] = 0.0;
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
    double* host_linear_arr,
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
        for (int i = 0; i < eval_size; ++i) {
            host_linear_arr[i] = raw[(size_t)i];
        }
        for (int i = 0; i < fm_size; ++i) {
            host_factor_arr[i] = raw[(size_t)eval_size + (size_t)i];
        }
        std::cerr << "imported linear+factor format, linear " << eval_size << " factor " << fm_size << std::endl;
        return;
    }
    if (raw.size() == expected_factor_only) {
        for (int i = 0; i < eval_size; ++i) {
            host_linear_arr[i] = 0.0;
        }
        for (int i = 0; i < fm_size; ++i) {
            host_factor_arr[i] = raw[(size_t)i];
        }
        std::cerr << "imported factor-only format, linear initialized to 0, factor " << fm_size << std::endl;
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
    const double* device_linear_arr,
    const double* device_factor_arr,
    const int n_data,
    const int* device_start_idx_arr,
    const Adj_Data* device_train_data,
    const int* device_rev_idx_arr,
    double* device_residual_linear_arr,
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
            predicted_value += device_linear_arr[idx] * x_val;
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

        const double grad_linear = residual_error * x_val;
        atomic_add_fp64(&device_residual_linear_arr[idx], grad_linear);
        atomic_add_fp64(&device_residual_linear_arr[rev_idx], grad_linear);

        for (int f = 0; f < ADJ_FM_DIM; ++f) {
            const double vif = device_factor_arr[idx * ADJ_FM_DIM + f];
            const double grad_coef = residual_error * x_val * (sum_vx[f] - vif * x_val);
            double* dst = device_residual_factor_arr + idx * ADJ_FM_DIM + f;
            double* rev_dst = device_residual_factor_arr + rev_idx * ADJ_FM_DIM + f;
            atomic_add_fp64(dst, grad_coef);
            atomic_add_fp64(rev_dst, grad_coef);
        }
    }

    atomic_add_fp64(&device_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_data);
    atomic_add_fp64(&device_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_data);
}

__global__ void adj_calculate_val_loss_fm(
    const double* device_linear_arr,
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
            predicted_value += device_linear_arr[idx] * x_val;
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
    atomic_add_fp64(&device_val_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_val_data);
    atomic_add_fp64(&device_val_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_val_data);
}

__global__ void adj_calculate_loss_round_linear_fm(
    const int change_idx,
    const int rev_change_idx,
    const int* device_linear_arr_roundup,
    const int* device_linear_arr_rounddown,
    const bool* device_round_arr,
    const double* device_factor_arr,
    const int n_data,
    const int* device_start_idx_arr,
    const Adj_Data* device_train_data,
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
        if (!get_feature_index_and_value(device_train_data[data_idx], device_start_idx_arr, i, idx, x_val)) {
            continue;
        }
        idx_arr[used_n] = idx;
        x_arr[used_n] = x_val;
        ++used_n;

        bool round_down = device_round_arr[idx];
        if (idx == change_idx || idx == rev_change_idx) {
            round_down = !round_down;
        }
        const int linear_i = round_down ? device_linear_arr_rounddown[idx] : device_linear_arr_roundup[idx];
        predicted_value += (double)linear_i * x_val;
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

    predicted_value += predicted_value >= 0.0 ? ADJ_STEP_2 : -ADJ_STEP_2;
    predicted_value /= ADJ_STEP;
#if ROUND_USE_CLIP
    if (predicted_value > HW2) {
        predicted_value = HW2;
    } else if (predicted_value < -HW2) {
        predicted_value = -HW2;
    }
#endif

    const double residual_error = device_train_data[data_idx].score / ADJ_STEP - predicted_value;
    atomic_add_fp64(&device_error_monitor_arr[0], residual_error * residual_error / n_data);
    atomic_add_fp64(&device_error_monitor_arr[1], fabs(residual_error) / n_data);
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

__global__ void adam_linear(
    const int phase,
    const int eval_size,
    double* device_linear_arr,
    const int* device_n_appear_arr,
    double* device_residual_linear_arr,
    const double alpha_stab,
    double* device_m_linear_arr,
    double* device_v_linear_arr,
    const int n_loop) {
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size) {
        return;
    }

    if (device_n_appear_arr[eval_idx] > ADJ_IGNORE_N_APPEAR || (phase <= 11 && device_n_appear_arr[eval_idx] > 0)) {
        const double lr = alpha_stab / (double)device_n_appear_arr[eval_idx];
        const double grad = 2.0 * device_residual_linear_arr[eval_idx];
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-7;
        const double lrt = lr * sqrt(1.0 - pow(beta2, n_loop)) / (1.0 - pow(beta1, n_loop));
        device_m_linear_arr[eval_idx] += (1.0 - beta1) * (grad - device_m_linear_arr[eval_idx]);
        device_v_linear_arr[eval_idx] += (1.0 - beta2) * (grad * grad - device_v_linear_arr[eval_idx]);
        device_linear_arr[eval_idx] += lrt * device_m_linear_arr[eval_idx] / (sqrt(device_v_linear_arr[eval_idx]) + epsilon);
    }
    device_residual_linear_arr[eval_idx] = 0.0;
}

void adj_output_param_fm(
    int phase,
    int eval_size,
    const double* host_linear_arr,
    const double* host_factor_arr) {
    const std::string filename = std::string("trained/") + std::to_string(phase) + "_fm.txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << std::endl;
        return;
    }
    ofs.setf(std::ios::fixed);
    ofs.precision(10);
    for (int i = 0; i < eval_size; ++i) {
        int linear_i = (int)std::llround(host_linear_arr[i]);
        if (linear_i > 32767) {
            linear_i = 32767;
        } else if (linear_i < -32767) {
            linear_i = -32767;
        }
        ofs << linear_i << '\n';
    }
    for (int i = 0; i < eval_size * ADJ_FM_DIM; ++i) {
        ofs << host_factor_arr[i] << '\n';
    }
    ofs.close();
    std::cerr << "linear+FM params output to " << filename << std::endl;
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

    double* host_linear_arr = (double*)malloc(sizeof(double) * eval_size);
    double* host_factor_arr = (double*)malloc(sizeof(double) * fm_size);
    int* host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size);
    Adj_Data* host_train_data = (Adj_Data*)malloc(sizeof(Adj_Data) * ADJ_MAX_N_DATA);
    int* host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    int* weight_arr = (int*)malloc(sizeof(int) * eval_size);
    double* host_error_monitor_arr = (double*)malloc(sizeof(double) * N_ERROR_MONITOR);
    double* host_val_error_monitor_arr = (double*)malloc(sizeof(double) * N_TEST_ERROR_MONITOR);
    if (host_linear_arr == nullptr || host_factor_arr == nullptr || host_rev_idx_arr == nullptr ||
        host_train_data == nullptr || host_n_appear_arr == nullptr || host_error_monitor_arr == nullptr ||
        host_val_error_monitor_arr == nullptr) {
        std::cerr << "cannot allocate memory" << std::endl;
        return 1;
    }

    adj_init_arr(eval_size, host_linear_arr, host_factor_arr, host_rev_idx_arr, host_n_appear_arr);
    adj_import_eval_fm(in_file, eval_size, host_linear_arr, host_factor_arr);

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

    double* device_linear_arr;
    double* device_factor_arr;
    int* device_rev_idx_arr;
    Adj_Data* device_train_data;
    Adj_Data* device_val_data;
    int* device_n_appear_arr;
    double* device_residual_linear_arr;
    double* device_residual_factor_arr;
    double* device_error_monitor_arr;
    double* device_val_error_monitor_arr;
    int* device_start_idx_arr;

    CUDA_CHECK(cudaMalloc(&device_linear_arr, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_train_data, sizeof(Adj_Data) * n_train_data));
    CUDA_CHECK(cudaMalloc(&device_val_data, sizeof(Adj_Data) * n_val_data));
    CUDA_CHECK(cudaMalloc(&device_n_appear_arr, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_residual_linear_arr, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_residual_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR));
    CUDA_CHECK(cudaMalloc(&device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR));
    CUDA_CHECK(cudaMalloc(&device_start_idx_arr, sizeof(int) * ADJ_N_FEATURES));

    CUDA_CHECK(cudaMemcpy(device_linear_arr, host_linear_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_factor_arr, host_factor_arr, sizeof(double) * fm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_train_data, host_train_data, sizeof(Adj_Data) * n_train_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_val_data, host_val_data, sizeof(Adj_Data) * n_val_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_n_appear_arr, host_n_appear_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_residual_linear_arr, 0, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMemset(device_residual_factor_arr, 0, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemcpy(device_start_idx_arr, host_start_idx_arr, sizeof(int) * ADJ_N_FEATURES, cudaMemcpyHostToDevice));

    double* device_m_linear_arr;
    double* device_v_linear_arr;
    double* device_m_factor_arr;
    double* device_v_factor_arr;
    CUDA_CHECK(cudaMalloc(&device_m_linear_arr, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_v_linear_arr, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_m_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMalloc(&device_v_factor_arr, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemset(device_m_linear_arr, 0, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMemset(device_v_linear_arr, 0, sizeof(double) * eval_size));
    CUDA_CHECK(cudaMemset(device_m_factor_arr, 0, sizeof(double) * fm_size));
    CUDA_CHECK(cudaMemset(device_v_factor_arr, 0, sizeof(double) * fm_size));

    const int n_blocks_val = (n_val_data + N_THREADS_PER_BLOCK_TEST - 1) / N_THREADS_PER_BLOCK_TEST;
    const int n_blocks_residual = (n_train_data + N_THREADS_PER_BLOCK_RESIDUAL - 1) / N_THREADS_PER_BLOCK_RESIDUAL;
    const int n_blocks_next_linear = (eval_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;
    const int n_blocks_next_factor = (fm_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;

    std::cerr << "n_blocks_val " << n_blocks_val
              << " n_blocks_residual " << n_blocks_residual
              << " n_blocks_next_linear " << n_blocks_next_linear
              << " n_blocks_next_factor " << n_blocks_next_factor << std::endl;

    std::cerr << "phase " << phase << std::endl;
    CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
    adj_calculate_residual_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
        device_linear_arr,
        device_factor_arr,
        n_train_data,
        device_start_idx_arr,
        device_train_data,
        device_rev_idx_arr,
        device_residual_linear_arr,
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
            device_linear_arr,
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
            device_linear_arr,
            device_factor_arr,
            n_train_data,
            device_start_idx_arr,
            device_train_data,
            device_rev_idx_arr,
            device_residual_linear_arr,
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

        adam_linear<<<n_blocks_next_linear, N_THREADS_PER_BLOCK_NEXT_STEP>>>(
            phase,
            eval_size,
            device_linear_arr,
            device_n_appear_arr,
            device_residual_linear_arr,
            alpha_stab,
            device_m_linear_arr,
            device_v_linear_arr,
            n_loop);
        CUDA_CHECK_LAUNCH();

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

    CUDA_CHECK(cudaMemcpy(host_linear_arr, device_linear_arr, sizeof(double) * eval_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_factor_arr, device_factor_arr, sizeof(double) * fm_size, cudaMemcpyDeviceToHost));

    int* host_linear_arr_roundup = (int*)malloc(sizeof(int) * eval_size);
    int* host_linear_arr_rounddown = (int*)malloc(sizeof(int) * eval_size);
    bool* host_round_arr = (bool*)malloc(sizeof(bool) * eval_size);
    if (host_linear_arr_roundup == nullptr || host_linear_arr_rounddown == nullptr || host_round_arr == nullptr) {
        std::cerr << "cannot allocate memory for rounding" << std::endl;
        return 1;
    }

    int n_roundup = 0;
    int n_rounddown = 0;
    for (int i = 0; i < eval_size; ++i) {
        int roundup = (int)std::ceil(host_linear_arr[i]);
        int rounddown = (int)std::floor(host_linear_arr[i]);
        if (roundup > 32767) {
            roundup = 32767;
        }
        if (roundup < -32767) {
            roundup = -32767;
        }
        if (rounddown > 32767) {
            rounddown = 32767;
        }
        if (rounddown < -32767) {
            rounddown = -32767;
        }
        host_linear_arr_roundup[i] = roundup;
        host_linear_arr_rounddown[i] = rounddown;
        host_round_arr[i] = ((int)std::llround(host_linear_arr[i]) == rounddown); // false: round-up, true: round-down
        if (roundup != rounddown) {
            if (!host_round_arr[i]) {
                ++n_roundup;
            } else {
                ++n_rounddown;
            }
        }
    }
    std::cerr << "linear init round-up " << n_roundup << " round-down " << n_rounddown << std::endl;

    int* device_linear_arr_roundup;
    int* device_linear_arr_rounddown;
    bool* device_round_arr;
    CUDA_CHECK(cudaMalloc(&device_linear_arr_roundup, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_linear_arr_rounddown, sizeof(int) * eval_size));
    CUDA_CHECK(cudaMalloc(&device_round_arr, sizeof(bool) * eval_size));
    CUDA_CHECK(cudaMemcpy(device_linear_arr_roundup, host_linear_arr_roundup, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_linear_arr_rounddown, host_linear_arr_rounddown, sizeof(int) * eval_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_round_arr, host_round_arr, sizeof(bool) * eval_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
    adj_calculate_loss_round_linear_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
        -1,
        -1,
        device_linear_arr_roundup,
        device_linear_arr_rounddown,
        device_round_arr,
        device_factor_arr,
        n_train_data,
        device_start_idx_arr,
        device_train_data,
        device_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));

    double min_mse = host_error_monitor_arr[0];
    double min_mae = host_error_monitor_arr[1];
    std::cerr << "before linear rounding MSE " << min_mse << " MAE " << min_mae << std::endl;

    std::uniform_int_distribution<int> randint_eval(0, eval_size - 1);
    uint64_t round_n_loop = 0;
    uint64_t round_n_updated = 0;
    uint64_t round_n_improve = 0;
    const uint64_t round_strt = tim();
    const uint64_t round_tl = 60000;
    while (tim() - round_strt < round_tl && ((round_n_loop < 100) || ((double)round_n_improve * 100.0 / (double)round_n_loop > 0.01))) {
        const int change_idx = randint_eval(engine);
        if (host_linear_arr_roundup[change_idx] == host_linear_arr_rounddown[change_idx]) {
            continue;
        }
        const int rev_change_idx = host_rev_idx_arr[change_idx];
        CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
        adj_calculate_loss_round_linear_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
            change_idx,
            rev_change_idx,
            device_linear_arr_roundup,
            device_linear_arr_rounddown,
            device_round_arr,
            device_factor_arr,
            n_train_data,
            device_start_idx_arr,
            device_train_data,
            device_error_monitor_arr);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));
        if (host_error_monitor_arr[0] <= min_mse) {
            ++round_n_updated;
            if (host_error_monitor_arr[0] < min_mse) {
                ++round_n_improve;
            }
            min_mse = host_error_monitor_arr[0];
            min_mae = host_error_monitor_arr[1];
            host_round_arr[change_idx] = !host_round_arr[change_idx];
            if (change_idx != rev_change_idx) {
                host_round_arr[rev_change_idx] = !host_round_arr[rev_change_idx];
            }
            CUDA_CHECK(cudaMemcpy(device_round_arr, host_round_arr, sizeof(bool) * eval_size, cudaMemcpyHostToDevice));
        }
        ++round_n_loop;
        const int percent = (int)((tim() - round_strt) * 100 / round_tl);
        std::cerr << "\rlinear rounding " << percent << "%"
                  << " n_loop " << round_n_loop
                  << " n_updated " << round_n_updated
                  << " n_improve " << round_n_improve
                  << " MSE " << min_mse
                  << " MAE " << min_mae
                  << "                        ";
    }
    std::cerr << std::endl;

    n_roundup = 0;
    n_rounddown = 0;
    for (int i = 0; i < eval_size; ++i) {
        host_linear_arr[i] = host_round_arr[i] ? (double)host_linear_arr_rounddown[i] : (double)host_linear_arr_roundup[i];
        if (host_linear_arr_roundup[i] != host_linear_arr_rounddown[i]) {
            if (!host_round_arr[i]) {
                ++n_roundup;
            } else {
                ++n_rounddown;
            }
        }
    }
    std::cerr << "linear final round-up " << n_roundup << " round-down " << n_rounddown << std::endl;

    CUDA_CHECK(cudaMemcpy(device_linear_arr, host_linear_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR));
    adj_calculate_residual_fm<<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>>(
        device_linear_arr,
        device_factor_arr,
        n_train_data,
        device_start_idx_arr,
        device_train_data,
        device_rev_idx_arr,
        device_residual_linear_arr,
        device_residual_factor_arr,
        device_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(device_val_error_monitor_arr, 0, sizeof(double) * N_TEST_ERROR_MONITOR));
    adj_calculate_val_loss_fm<<<n_blocks_val, N_THREADS_PER_BLOCK_TEST>>>(
        device_linear_arr,
        device_factor_arr,
        n_val_data,
        device_start_idx_arr,
        device_val_data,
        device_val_error_monitor_arr);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(host_val_error_monitor_arr, device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR, cudaMemcpyDeviceToHost));

    adj_output_param_fm(phase, eval_size, host_linear_arr, host_factor_arr);
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
              << " (Linear+FM float) alpha " << alpha_in
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
              << " (Linear+FM float) alpha " << alpha_in
              << " n_patience " << n_patience
              << " reduce_lr_patience " << reduce_lr_patience
              << " reduce_lr_ratio " << reduce_lr_ratio
              << std::endl;

    return 0;
}
