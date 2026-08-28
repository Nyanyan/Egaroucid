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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#define OPTIMIZER_INCLUDE
#include "evaluation_definition.hpp"

// train data constant
#define ADJ_MAX_N_FILES 200
#if ADJ_CELL_WEIGHT
    #define ADJ_MAX_N_DATA 1000000
#else
    #define ADJ_MAX_N_DATA 200000000
#endif
#define ADJ_MAX_N_TEST_DATA 100000


// settings
#define RESIDUAL_USE_CLIP false
#define ROUND_USE_CLIP false
#define USE_PARAM_LIMIT false
#define USE_WARMUP false


// training constant
#define ADJ_IGNORE_N_APPEAR 0 // ignore features that appeared only 3 or less times

// GPU constant
#define N_THREADS_PER_BLOCK_TEST 1024
#define N_THREADS_PER_BLOCK_RESIDUAL 1024
#define N_THREADS_PER_BLOCK_NEXT_STEP 1024


// monitor constant
#define N_ERROR_MONITOR 2 // 0 for MSE, 1 for MAE
#define N_TEST_ERROR_MONITOR 2 // 0 for MSE, 1 for MAE
#define N_REGRESSION_MONITOR 7

constexpr uint32_t DEFAULT_VALIDATION_SEED = 20260828U;
constexpr int DEFAULT_METRICS_INTERVAL = 25;

// train param
#define N_APPEAR_MIN_VAL 100


struct Adj_Data {
    uint16_t features[ADJ_N_FEATURES];
    double score;
};

struct Adj_Regression_Summary {
    double mse;
    double mae;
    double bias;
    double teacher_mean;
    double predicted_mean;
    double predicted_on_teacher_slope;
    double predicted_on_teacher_intercept;
    double teacher_on_predicted_slope;
    double teacher_on_predicted_intercept;
    double correlation;
    double ols_r_squared;
    double prediction_r_squared;
    double predicted_residual_covariance;
};

uint32_t adj_get_env_uint32(const char *name, uint32_t fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    try {
        return static_cast<uint32_t>(std::stoull(value));
    } catch (...) {
        std::cerr << "invalid " << name << "='" << value << "', use " << fallback << std::endl;
        return fallback;
    }
}

int adj_get_env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        std::cerr << "invalid " << name << "='" << value << "', use " << fallback << std::endl;
        return fallback;
    }
}

std::string adj_get_env_string(const char *name, const std::string &fallback) {
    const char *value = std::getenv(name);
    return value == nullptr || *value == '\0' ? fallback : std::string(value);
}

void adj_cuda_check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA failure in " << operation << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(1);
    }
}

class Adj_SplitMix64 {
private:
    uint64_t state;

public:
    explicit Adj_SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    uint64_t uniform_bounded(uint64_t bound) {
        // Reject the short prefix so that modulo reduction is exactly uniform.
        // Unsigned overflow is defined by C++, making this independent of the
        // standard library's uniform_int_distribution implementation.
        const uint64_t threshold = (0ULL - bound) % bound;
        while (true) {
            const uint64_t value = next();
            if (value >= threshold) {
                return value % bound;
            }
        }
    }
};

void adj_deterministic_shuffle(Adj_Data *data, int n_data, uint64_t seed) {
    Adj_SplitMix64 random_engine(seed);
    for (uint64_t remaining = static_cast<uint64_t>(n_data); remaining > 1; --remaining) {
        const uint64_t swap_idx = random_engine.uniform_bounded(remaining);
        std::swap(data[remaining - 1], data[swap_idx]);
    }
}

uint64_t adj_validation_fingerprint(const Adj_Data *data, int n_data, int *n_hashed) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;
    const uint64_t count = static_cast<uint64_t>(n_data);
    for (int shift = 0; shift < 8; ++shift) {
        hash ^= static_cast<uint8_t>((count >> (shift * 8)) & 0xffU);
        hash *= FNV_PRIME;
    }
    for (int data_idx = 0; data_idx < n_data; ++data_idx) {
        for (int feature_idx = 0; feature_idx < ADJ_N_FEATURES; ++feature_idx) {
            uint16_t value = data[data_idx].features[feature_idx];
            hash ^= static_cast<uint8_t>(value & 0xffU);
            hash *= FNV_PRIME;
            hash ^= static_cast<uint8_t>(value >> 8);
            hash *= FNV_PRIME;
        }
        int score = static_cast<int>(std::llround(data[data_idx].score / ADJ_STEP));
        for (int shift = 0; shift < 4; ++shift) {
            hash ^= static_cast<uint8_t>((static_cast<uint32_t>(score) >> (shift * 8)) & 0xffU);
            hash *= FNV_PRIME;
        }
    }
    *n_hashed = n_data;
    return hash;
}



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
void adj_init_arr(int eval_size, double *host_eval_arr, int *host_rev_idx_arr, int *host_n_appear_arr) {
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
bool adj_import_eval(std::string file, int eval_size, double *host_eval_arr) {
    auto initialize_zero = [&]() {
        for (int i = 0; i < eval_size; ++i) {
            host_eval_arr[i] = 0.0;
        }
    };
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "evaluation file " << file << " not exist, initialize with 0" << std::endl;
        initialize_zero();
        return false;
    }
    std::cerr << "importing eval params " << file << std::endl;
    std::string line;
    for (int i = 0; i < eval_size; ++i){
        if (!getline(ifs, line)) {
            std::cerr << "ERROR evaluation file broken" << std::endl;
            initialize_zero();
            return false;
        }
        try {
            host_eval_arr[i] = std::stod(line);
        } catch (...) {
            std::cerr << "ERROR invalid evaluation parameter at line " << (i + 1) << std::endl;
            initialize_zero();
            return false;
        }
    }
    return true;
}

/*
    @brief import train data
*/
std::pair<int, double> adj_import_data(int n_files, char* files[], Adj_Data* host_train_data, int *host_rev_idx_arr, int *host_n_appear_arr) {
    int n_data = 0;
    FILE* fp;
    int16_t n_discs, score, player;
    Adj_Data data;
    double score_avg = 0.0;
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
        if (fopen_s(&fp, files[file_idx], "rb") != 0) {
            std::cerr << "can't open " << files[file_idx] << std::endl;
            continue;
        }
        int n_data_before = n_data;
        while (n_data < ADJ_MAX_N_DATA) {
            if (fread(&n_discs, 2, 1, fp) < 1)
                break;
            fread(&player, 2, 1, fp);
            fread(host_train_data[n_data].features, 2, ADJ_N_FEATURES, fp);
            fread(&score, 2, 1, fp);
            host_train_data[n_data].score = (double)score * ADJ_STEP;
            //if ((n_data & 0xffff) == 0xffff)
            //    std::cerr << '\r' << n_data;
            score_avg += score;
            ++n_data;
        }
        fclose(fp);
        if (n_data_before < n_data) {
            std::cerr << files[file_idx] << " " << n_data << std::endl;
        }
    }
    score_avg /= n_data;
    //std::cerr << std::endl;
    //std::cerr << n_data << " data loaded" << std::endl;
    std::cerr << "score avg " << score_avg << std::endl;
    return std::make_pair(n_data, score_avg);
}

/*
    @brief calculate residual error
*/
__global__ void adj_calculate_residual(const double *device_eval_arr, const int n_data, const int *device_start_idx_arr, const Adj_Data *device_train_data, int *device_rev_idx_arr, double *device_residual_arr, double *device_error_monitor_arr){
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_data){
        return;
    }
    double predicted_value = 0.0;
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
#if RESIDUAL_USE_CLIP
    if (predicted_value > HW2 * ADJ_STEP) {
        predicted_value = HW2 * ADJ_STEP;
    } else if (predicted_value < -HW2 * ADJ_STEP) {
        predicted_value = -HW2 * ADJ_STEP;
    }
#endif
    double residual_error = device_train_data[data_idx].score - predicted_value;
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
    @brief calculate val loss
*/
__global__ void adj_calculate_val_loss(const double *device_eval_arr, const int n_val_data, const int *device_start_idx_arr, const Adj_Data *device_val_data, double *device_val_error_monitor_arr){
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_val_data){
        return;
    }
    double predicted_value = 0.0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        #if ADJ_CELL_WEIGHT
            if (device_val_data[data_idx].features[i] < 10){
                predicted_value += device_eval_arr[device_val_data[data_idx].features[i]];
            } else if (device_val_data[data_idx].features[i] < 20){
                predicted_value -= device_eval_arr[device_val_data[data_idx].features[i] - 10];
            }
        #else
            predicted_value += device_eval_arr[device_start_idx_arr[i] + (int)device_val_data[data_idx].features[i]];
        #endif
    }
#if RESIDUAL_USE_CLIP
    if (predicted_value > HW2 * ADJ_STEP) {
        predicted_value = HW2 * ADJ_STEP;
    } else if (predicted_value < -HW2 * ADJ_STEP) {
        predicted_value = -HW2 * ADJ_STEP;
    }
#endif
    double residual_error = device_val_data[data_idx].score - predicted_value;
    atomicAdd(&device_val_error_monitor_arr[0], (residual_error / ADJ_STEP) * (residual_error / ADJ_STEP) / n_val_data);
    atomicAdd(&device_val_error_monitor_arr[1], fabs(residual_error / ADJ_STEP) / n_val_data);
}

__device__ double adj_warp_reduce_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ void adj_reduce_regression_values(double values[N_REGRESSION_MONITOR], double *device_monitor) {
    constexpr int MAX_WARPS = 32;
    __shared__ double warp_sums[N_REGRESSION_MONITOR][MAX_WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;
    for (int metric_idx = 0; metric_idx < N_REGRESSION_MONITOR; ++metric_idx) {
        values[metric_idx] = adj_warp_reduce_sum(values[metric_idx]);
        if (lane == 0) {
            warp_sums[metric_idx][warp] = values[metric_idx];
        }
    }
    __syncthreads();
    if (warp == 0) {
        for (int metric_idx = 0; metric_idx < N_REGRESSION_MONITOR; ++metric_idx) {
            double value = lane < n_warps ? warp_sums[metric_idx][lane] : 0.0;
            value = adj_warp_reduce_sum(value);
            if (lane == 0) {
                atomicAdd(&device_monitor[metric_idx], value);
            }
        }
    }
}

__global__ void adj_calculate_regression_metrics(
    const double *device_eval_arr,
    const int n_data,
    const int *device_start_idx_arr,
    const Adj_Data *device_data,
    double *device_monitor
) {
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double predicted_value = 0.0;
    double teacher_value = 0.0;
    if (data_idx < n_data) {
        for (int i = 0; i < ADJ_N_FEATURES; ++i) {
#if ADJ_CELL_WEIGHT
            if (device_data[data_idx].features[i] < 10) {
                predicted_value += device_eval_arr[device_data[data_idx].features[i]];
            } else if (device_data[data_idx].features[i] < 20) {
                predicted_value -= device_eval_arr[device_data[data_idx].features[i] - 10];
            }
#else
            predicted_value += device_eval_arr[device_start_idx_arr[i] + static_cast<int>(device_data[data_idx].features[i])];
#endif
        }
#if RESIDUAL_USE_CLIP
        if (predicted_value > HW2 * ADJ_STEP) {
            predicted_value = HW2 * ADJ_STEP;
        } else if (predicted_value < -HW2 * ADJ_STEP) {
            predicted_value = -HW2 * ADJ_STEP;
        }
#endif
        predicted_value /= ADJ_STEP;
        teacher_value = device_data[data_idx].score / ADJ_STEP;
    }
    const double residual = teacher_value - predicted_value;
    double values[N_REGRESSION_MONITOR] = {
        predicted_value,
        teacher_value,
        predicted_value * teacher_value,
        teacher_value * teacher_value,
        predicted_value * predicted_value,
        residual * residual,
        fabs(residual)
    };
    adj_reduce_regression_values(values, device_monitor);
}

__global__ void adj_calculate_regression_metrics_round(
    const int *device_eval_arr_roundup,
    const int *device_eval_arr_rounddown,
    const bool *device_round_arr,
    const int n_data,
    const int *device_start_idx_arr,
    const Adj_Data *device_data,
    double *device_monitor
) {
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double predicted_value = 0.0;
    double teacher_value = 0.0;
    if (data_idx < n_data) {
        int predicted_raw = 0;
        for (int i = 0; i < ADJ_N_FEATURES; ++i) {
            const int idx = device_start_idx_arr[i] + static_cast<int>(device_data[data_idx].features[i]);
            predicted_raw += device_round_arr[idx] ? device_eval_arr_rounddown[idx] : device_eval_arr_roundup[idx];
        }
        predicted_raw += predicted_raw >= 0 ? ADJ_STEP_2 : -ADJ_STEP_2;
        predicted_raw /= ADJ_STEP;
#if ROUND_USE_CLIP
        if (predicted_raw > HW2) {
            predicted_raw = HW2;
        } else if (predicted_raw < -HW2) {
            predicted_raw = -HW2;
        }
#endif
        predicted_value = predicted_raw;
        teacher_value = device_data[data_idx].score / ADJ_STEP;
    }
    const double residual = teacher_value - predicted_value;
    double values[N_REGRESSION_MONITOR] = {
        predicted_value,
        teacher_value,
        predicted_value * teacher_value,
        teacher_value * teacher_value,
        predicted_value * predicted_value,
        residual * residual,
        fabs(residual)
    };
    adj_reduce_regression_values(values, device_monitor);
}

Adj_Regression_Summary adj_summarize_regression(const double *sums, int n_data) {
    const double n = static_cast<double>(n_data);
    const double predicted_mean = sums[0] / n;
    const double teacher_mean = sums[1] / n;
    const double predicted_teacher_mean = sums[2] / n;
    const double teacher_squared_mean = sums[3] / n;
    const double predicted_squared_mean = sums[4] / n;
    const double covariance = predicted_teacher_mean - predicted_mean * teacher_mean;
    const double teacher_variance = std::max(0.0, teacher_squared_mean - teacher_mean * teacher_mean);
    const double predicted_variance = std::max(0.0, predicted_squared_mean - predicted_mean * predicted_mean);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double predicted_on_teacher_slope = teacher_variance > 0.0 ? covariance / teacher_variance : nan;
    const double teacher_on_predicted_slope = predicted_variance > 0.0 ? covariance / predicted_variance : nan;
    const double correlation = teacher_variance > 0.0 && predicted_variance > 0.0
        ? covariance / std::sqrt(teacher_variance * predicted_variance)
        : nan;
    Adj_Regression_Summary result;
    result.mse = sums[5] / n;
    result.mae = sums[6] / n;
    result.bias = predicted_mean - teacher_mean;
    result.teacher_mean = teacher_mean;
    result.predicted_mean = predicted_mean;
    result.predicted_on_teacher_slope = predicted_on_teacher_slope;
    result.predicted_on_teacher_intercept = predicted_mean - predicted_on_teacher_slope * teacher_mean;
    result.teacher_on_predicted_slope = teacher_on_predicted_slope;
    result.teacher_on_predicted_intercept = teacher_mean - teacher_on_predicted_slope * predicted_mean;
    result.correlation = correlation;
    result.ols_r_squared = correlation * correlation;
    result.prediction_r_squared = teacher_variance > 0.0 ? 1.0 - result.mse / teacher_variance : nan;
    result.predicted_residual_covariance = covariance - predicted_variance;
    return result;
}

Adj_Regression_Summary adj_collect_regression_metrics(
    const double *device_eval_arr,
    int n_data,
    const int *device_start_idx_arr,
    const Adj_Data *device_data,
    double *device_monitor,
    double *host_monitor
) {
    adj_cuda_check(cudaMemset(device_monitor, 0, sizeof(double) * N_REGRESSION_MONITOR), "initialize float regression monitor");
    const int n_blocks = (n_data + N_THREADS_PER_BLOCK_TEST - 1) / N_THREADS_PER_BLOCK_TEST;
    adj_calculate_regression_metrics<<<n_blocks, N_THREADS_PER_BLOCK_TEST>>>(
        device_eval_arr, n_data, device_start_idx_arr, device_data, device_monitor
    );
    adj_cuda_check(cudaGetLastError(), "launch float regression metrics");
    adj_cuda_check(cudaMemcpy(host_monitor, device_monitor, sizeof(double) * N_REGRESSION_MONITOR, cudaMemcpyDeviceToHost), "copy float regression metrics");
    return adj_summarize_regression(host_monitor, n_data);
}

Adj_Regression_Summary adj_collect_regression_metrics_round(
    const int *device_eval_arr_roundup,
    const int *device_eval_arr_rounddown,
    const bool *device_round_arr,
    int n_data,
    const int *device_start_idx_arr,
    const Adj_Data *device_data,
    double *device_monitor,
    double *host_monitor
) {
    adj_cuda_check(cudaMemset(device_monitor, 0, sizeof(double) * N_REGRESSION_MONITOR), "initialize integer regression monitor");
    const int n_blocks = (n_data + N_THREADS_PER_BLOCK_TEST - 1) / N_THREADS_PER_BLOCK_TEST;
    adj_calculate_regression_metrics_round<<<n_blocks, N_THREADS_PER_BLOCK_TEST>>>(
        device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr,
        n_data, device_start_idx_arr, device_data, device_monitor
    );
    adj_cuda_check(cudaGetLastError(), "launch integer regression metrics");
    adj_cuda_check(cudaMemcpy(host_monitor, device_monitor, sizeof(double) * N_REGRESSION_MONITOR, cudaMemcpyDeviceToHost), "copy integer regression metrics");
    return adj_summarize_regression(host_monitor, n_data);
}

void adj_write_metrics_row(
    std::ofstream &metrics,
    int phase,
    const std::string &stage,
    int loop,
    uint64_t elapsed_ms,
    const std::string &split,
    int n_data,
    const std::string &representation,
    double alpha,
    int validation_loss_increase,
    const Adj_Regression_Summary &summary
) {
    metrics << phase << ',' << stage << ',' << loop << ',' << elapsed_ms << ','
            << split << ',' << n_data << ',' << representation << ','
            << std::setprecision(17) << alpha << ',' << validation_loss_increase << ','
            << summary.mse << ',' << summary.mae << ',' << summary.bias << ','
            << summary.teacher_mean << ',' << summary.predicted_mean << ','
            << summary.predicted_on_teacher_slope << ',' << summary.predicted_on_teacher_intercept << ','
            << summary.teacher_on_predicted_slope << ',' << summary.teacher_on_predicted_intercept << ','
            << summary.correlation << ',' << summary.ols_r_squared << ',' << summary.prediction_r_squared << ','
            << summary.predicted_residual_covariance << '\n';
    metrics.flush();
}

/*
    @brief calculate loss for hillclimb
*/
__global__ void adj_calculate_loss_round(const int change_idx, const int rev_change_idx, const int *device_eval_arr_roundup, const int *device_eval_arr_rounddown, const bool *device_round_arr, const int n_train_data, const int *device_start_idx_arr, const Adj_Data *device_train_data, double *device_error_monitor_arr){
    const int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= n_train_data){
        return;
    }
    int predicted_value = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i){
        #if ADJ_CELL_WEIGHT /******************************************** UNDER CONSTRUCTION ********************************************/
            if (device_val_data[data_idx].features[i] < 10){
                predicted_value += device_eval_arr[device_val_data[data_idx].features[i]];
            } else if (device_val_data[data_idx].features[i] < 20){
                predicted_value -= device_eval_arr[device_val_data[data_idx].features[i] - 10];
            }
        #else
            int idx = device_start_idx_arr[i] + (int)device_train_data[data_idx].features[i];
            if (idx == change_idx || idx == rev_change_idx){
                if (device_round_arr[idx]){ // changed to round-up
                    predicted_value += device_eval_arr_roundup[idx];
                } else{ // changed to round-down
                    predicted_value += device_eval_arr_rounddown[idx];
                }
            } else{
                if (!device_round_arr[idx]){ // round-up
                    predicted_value += device_eval_arr_roundup[idx];
                } else{ // round-down
                    predicted_value += device_eval_arr_rounddown[idx];
                }
            }
        #endif
    }
    predicted_value += predicted_value >= 0 ? ADJ_STEP_2 : -ADJ_STEP_2;
    predicted_value /= ADJ_STEP;
#if ROUND_USE_CLIP
    if (predicted_value > HW2) {
        predicted_value = HW2;
    } else if (predicted_value < -HW2) {
        predicted_value = -HW2;
    }
#endif
    double residual_error = device_train_data[data_idx].score / ADJ_STEP - predicted_value;
    atomicAdd(&device_error_monitor_arr[0], residual_error * residual_error / n_train_data);
    atomicAdd(&device_error_monitor_arr[1], fabs(residual_error) / n_train_data);
}

/*
    @brief Gradient Descent Optimizer
*/
__global__ void gradient_descent(const int eval_size, double *device_eval_arr, int *device_n_appear_arr, double *device_residual_arr, double alpha_stab){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    double lr = alpha_stab / device_n_appear_arr[eval_idx];
    double grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        device_eval_arr[eval_idx] += lr * grad;
#if USE_PARAM_LIMIT
        if (device_eval_arr[eval_idx] > ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = ADJ_EVAL_PARAM_MAX;
        }
        if (device_eval_arr[eval_idx] < -ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = -ADJ_EVAL_PARAM_MAX;
        }
#endif
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Momentum Optimizer
*/
__global__ void momentum(const int eval_size, double *device_eval_arr, int *device_n_appear_arr, double *device_residual_arr, double alpha_stab, double *device_m_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    double lr = alpha_stab / device_n_appear_arr[eval_idx];
    double grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        constexpr double beta1 = 0.9;
        device_m_arr[eval_idx] = beta1 * device_m_arr[eval_idx] + lr * grad;
        device_eval_arr[eval_idx] += device_m_arr[eval_idx];
#if USE_PARAM_LIMIT
        if (device_eval_arr[eval_idx] > ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = ADJ_EVAL_PARAM_MAX;
        }
        if (device_eval_arr[eval_idx] < -ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = -ADJ_EVAL_PARAM_MAX;
        }
#endif
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief AdaGrad Optimizer
*/
__global__ void adagrad(const int eval_size, double *device_eval_arr, int *device_n_appear_arr, double *device_residual_arr, double alpha_stab, double *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    double lr = alpha_stab / device_n_appear_arr[eval_idx];
    double grad = 2.0 * device_residual_arr[eval_idx];
    if (grad != 0.0){
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-7;
        device_v_arr[eval_idx] += grad * grad;
        device_eval_arr[eval_idx] += lr * grad / (sqrt(device_v_arr[eval_idx]) + epsilon);
#if USE_PARAM_LIMIT
        if (device_eval_arr[eval_idx] > ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = ADJ_EVAL_PARAM_MAX;
        }
        if (device_eval_arr[eval_idx] < -ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = -ADJ_EVAL_PARAM_MAX;
        }
#endif
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Adam Optimizer
*/
__global__ void adam(const int phase, const int eval_size, double *device_eval_arr, int *device_n_appear_arr, double *device_residual_arr, double alpha_stab, double *device_m_arr, double *device_v_arr, const int n_loop){
    const int eval_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eval_idx >= eval_size){
        return;
    }
    if (device_n_appear_arr[eval_idx] > ADJ_IGNORE_N_APPEAR || (phase <= 11 && device_n_appear_arr[eval_idx] > 0)) {
        double lr = alpha_stab / (double)device_n_appear_arr[eval_idx];
        double grad = 2.0 * device_residual_arr[eval_idx];
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-7;
        double lrt = lr * sqrt(1.0 - pow(beta2, n_loop)) / (1.0 - pow(beta1, n_loop));
        device_m_arr[eval_idx] += (1.0 - beta1) * (grad - device_m_arr[eval_idx]);
        device_v_arr[eval_idx] += (1.0 - beta2) * (grad * grad - device_v_arr[eval_idx]);
        device_eval_arr[eval_idx] += lrt * device_m_arr[eval_idx] / (sqrt(device_v_arr[eval_idx]) + epsilon);
#if USE_PARAM_LIMIT
        if (device_eval_arr[eval_idx] > ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = ADJ_EVAL_PARAM_MAX;
        }
        if (device_eval_arr[eval_idx] < -ADJ_EVAL_PARAM_MAX) {
            device_eval_arr[eval_idx] = -ADJ_EVAL_PARAM_MAX;
        }
#endif
    }
    device_residual_arr[eval_idx] = 0.0;
}

/*
    @brief Output Parameters as integer
*/
void adj_output_param(int phase, int eval_size, double *host_eval_arr, const std::string &output_dir) {
    std::string filename = output_dir + "/" + std::to_string(phase) + ".txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << ", output to stdout" << std::endl;
    } else {
        for (int i = 0; i < eval_size; ++i) {
            ofs << (int)round(host_eval_arr[i]) << '\n';
        }
        ofs.close();
        std::cerr << "data output to " << filename << std::endl;
    }
}

void adj_output_float_param(int phase, int eval_size, const double *host_eval_arr, const std::string &output_dir) {
    std::string filename = output_dir + "/float_" + std::to_string(phase) + ".txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << std::endl;
    } else {
        ofs << std::setprecision(17);
        for (int i = 0; i < eval_size; ++i) {
            ofs << host_eval_arr[i] << '\n';
        }
        ofs.close();
        std::cerr << "floating-point parameters output to " << filename << std::endl;
    }
}

void adj_output_weight(int phase, int eval_size, int *weight_arr, const std::string &output_dir) {
    std::string filename = output_dir + "/weight_" + std::to_string(phase) + ".txt";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "cannot open " << filename << ", output to stdout" << std::endl;
    } else {
        for (int i = 0; i < eval_size; ++i) {
            ofs << weight_arr[i] << '\n';
        }
        ofs.close();
        std::cerr << "weight output to " << filename << std::endl;
    }
}

bool adj_close(double actual, double expected, double tolerance = 1.0e-10) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

int adj_monitor_self_test() {
    constexpr int N_DATA = 4099;
    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        eval_size += adj_eval_sizes[i];
    }
    double *host_eval_arr = (double*)malloc(sizeof(double) * eval_size);
    int *host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size);
    int *host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    Adj_Data *host_data = (Adj_Data*)malloc(sizeof(Adj_Data) * N_DATA);
    if (host_eval_arr == nullptr || host_rev_idx_arr == nullptr || host_n_appear_arr == nullptr || host_data == nullptr) {
        std::cerr << "monitor self-test host allocation failed" << std::endl;
        return 1;
    }
    adj_init_arr(eval_size, host_eval_arr, host_rev_idx_arr, host_n_appear_arr);
    int host_start_idx_arr[ADJ_N_FEATURES];
    int start_idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        if (i > 0 && adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            start_idx += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
        host_start_idx_arr[i] = start_idx;
    }
    for (int data_idx = 0; data_idx < N_DATA; ++data_idx) {
        for (int feature_idx = 0; feature_idx < ADJ_N_FEATURES; ++feature_idx) {
            host_data[data_idx].features[feature_idx] = 0;
        }
        host_data[data_idx].score = ADJ_STEP;
    }

    double *device_eval_arr = nullptr;
    int *device_rev_idx_arr = nullptr;
    int *device_start_idx_arr = nullptr;
    Adj_Data *device_data = nullptr;
    double *device_residual_arr = nullptr;
    double *device_error_monitor_arr = nullptr;
    double *device_regression_monitor_arr = nullptr;
    int *device_roundup = nullptr;
    int *device_rounddown = nullptr;
    bool *device_round = nullptr;
    cudaMalloc(&device_eval_arr, sizeof(double) * eval_size);
    cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size);
    cudaMalloc(&device_start_idx_arr, sizeof(int) * ADJ_N_FEATURES);
    cudaMalloc(&device_data, sizeof(Adj_Data) * N_DATA);
    cudaMalloc(&device_residual_arr, sizeof(double) * eval_size);
    cudaMalloc(&device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR);
    cudaMalloc(&device_regression_monitor_arr, sizeof(double) * N_REGRESSION_MONITOR);
    cudaMalloc(&device_roundup, sizeof(int) * eval_size);
    cudaMalloc(&device_rounddown, sizeof(int) * eval_size);
    cudaMalloc(&device_round, sizeof(bool) * eval_size);
    cudaError_t allocation_error = cudaGetLastError();
    if (allocation_error != cudaSuccess) {
        std::cerr << "monitor self-test device allocation failed: " << cudaGetErrorString(allocation_error) << std::endl;
        return 1;
    }
    cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_start_idx_arr, host_start_idx_arr, sizeof(int) * ADJ_N_FEATURES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data, host_data, sizeof(Adj_Data) * N_DATA, cudaMemcpyHostToDevice);
    cudaMemset(device_roundup, 0, sizeof(int) * eval_size);
    cudaMemset(device_rounddown, 0, sizeof(int) * eval_size);
    cudaMemset(device_round, 0, sizeof(bool) * eval_size);

    constexpr int N_THREADS = 1024;
    const int n_blocks = (N_DATA + N_THREADS - 1) / N_THREADS;
    double host_error_monitor[N_ERROR_MONITOR];
    for (int repetition = 0; repetition < 20; ++repetition) {
        cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR);
        adj_calculate_val_loss<<<n_blocks, N_THREADS>>>(
            device_eval_arr, N_DATA, device_start_idx_arr, device_data, device_error_monitor_arr
        );
        cudaMemcpy(host_error_monitor, device_error_monitor_arr, sizeof(host_error_monitor), cudaMemcpyDeviceToHost);
        if (!adj_close(host_error_monitor[0], 1.0) || !adj_close(host_error_monitor[1], 1.0)) {
            std::cerr << "validation monitor self-test failed at repetition " << repetition
                      << " MSE " << host_error_monitor[0] << " MAE " << host_error_monitor[1] << std::endl;
            return 1;
        }

        cudaMemset(device_residual_arr, 0, sizeof(double) * eval_size);
        cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR);
        adj_calculate_residual<<<n_blocks, N_THREADS>>>(
            device_eval_arr, N_DATA, device_start_idx_arr, device_data,
            device_rev_idx_arr, device_residual_arr, device_error_monitor_arr
        );
        cudaMemcpy(host_error_monitor, device_error_monitor_arr, sizeof(host_error_monitor), cudaMemcpyDeviceToHost);
        if (!adj_close(host_error_monitor[0], 1.0) || !adj_close(host_error_monitor[1], 1.0)) {
            std::cerr << "training monitor self-test failed at repetition " << repetition
                      << " MSE " << host_error_monitor[0] << " MAE " << host_error_monitor[1] << std::endl;
            return 1;
        }

        cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR);
        adj_calculate_loss_round<<<n_blocks, N_THREADS>>>(
            -1, -1, device_roundup, device_rounddown, device_round, N_DATA,
            device_start_idx_arr, device_data, device_error_monitor_arr
        );
        cudaMemcpy(host_error_monitor, device_error_monitor_arr, sizeof(host_error_monitor), cudaMemcpyDeviceToHost);
        if (!adj_close(host_error_monitor[0], 1.0) || !adj_close(host_error_monitor[1], 1.0)) {
            std::cerr << "round monitor self-test failed at repetition " << repetition
                      << " MSE " << host_error_monitor[0] << " MAE " << host_error_monitor[1] << std::endl;
            return 1;
        }
    }

    double host_regression_monitor[N_REGRESSION_MONITOR];
    Adj_Regression_Summary regression = adj_collect_regression_metrics(
        device_eval_arr, N_DATA, device_start_idx_arr, device_data,
        device_regression_monitor_arr, host_regression_monitor
    );
    if (!adj_close(regression.mse, 1.0) || !adj_close(regression.mae, 1.0) ||
        !adj_close(regression.bias, -1.0) || !adj_close(regression.teacher_mean, 1.0) ||
        !adj_close(regression.predicted_mean, 0.0)) {
        std::cerr << "regression monitor self-test failed" << std::endl;
        return 1;
    }
    cudaError_t kernel_error = cudaDeviceSynchronize();
    if (kernel_error != cudaSuccess) {
        std::cerr << "monitor self-test CUDA error: " << cudaGetErrorString(kernel_error) << std::endl;
        return 1;
    }
    std::cout << "monitor self-test passed: repeated multi-block MSE=1 MAE=1" << std::endl;
    return 0;
}


int main(int argc, char* argv[]) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test-monitor") == 0) {
        return adj_monitor_self_test();
    }
    std::cerr << EVAL_DEFINITION_NAME << std::endl;
    std::cerr << EVAL_DEFINITION_DESCRIPTION << std::endl;
    if (argc < 10) {
        std::cerr << "input [phase] [hour] [minute] [second] [alpha] [n_patience] [reduce_lr_patience] [reduce_lr_ratio] [in_file] [train_data...]" << std::endl;
        return 1;
    }
    if (argc - 10 >= ADJ_MAX_N_FILES) {
        std::cerr << "too many train files" << std::endl;
        return 1;
    }
    int phase = atoi(argv[1]);
    uint64_t hour = atoi(argv[2]);
    uint64_t minute = atoi(argv[3]);
    uint64_t second = atoi(argv[4]);
    double alpha = atof(argv[5]);
    double alpha_in = alpha;
    int n_patience = atoi(argv[6]);
    int reduce_lr_patience = atoi(argv[7]);
    double reduce_lr_ratio = atof(argv[8]);
    std::string in_file = (std::string)argv[9];
    const uint32_t validation_seed = adj_get_env_uint32("EGAROUCID_EVAL_VALIDATION_SEED", DEFAULT_VALIDATION_SEED);
    const int metrics_interval = std::max(1, adj_get_env_int("EGAROUCID_EVAL_METRICS_INTERVAL", DEFAULT_METRICS_INTERVAL));
    const int round_seconds = std::max(0, adj_get_env_int("EGAROUCID_EVAL_ROUND_SECONDS", 60));
    const std::string output_dir = adj_get_env_string("EGAROUCID_EVAL_OUTPUT_DIR", "trained");
    const std::string reference_file = adj_get_env_string("EGAROUCID_EVAL_REFERENCE_FILE", "");
    const char *round_seed_env = std::getenv("EGAROUCID_EVAL_ROUND_SEED");
    std::random_device round_seed_generator;
    const uint32_t round_seed = round_seed_env == nullptr || *round_seed_env == '\0'
        ? round_seed_generator()
        : adj_get_env_uint32("EGAROUCID_EVAL_ROUND_SEED", 0U);
    std::mt19937 round_engine(round_seed);
    char* train_files[ADJ_MAX_N_FILES];
    int n_train_data_file = argc - 10;
    for (int i = 0; i < n_train_data_file; ++i)
        train_files[i] = argv[i + 10];
    second += minute * 60 + hour * 3600;
    uint64_t msecond = second * 1000;

    int eval_size = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i){
        eval_size += adj_eval_sizes[i];
    }
    std::cerr << "eval_size " << eval_size << " sizeof_Adj_Data " << sizeof(Adj_Data) << std::endl;
    double *host_eval_arr = (double*)malloc(sizeof(double) * eval_size); // eval array
    int *host_rev_idx_arr = (int*)malloc(sizeof(int) * eval_size); // reversed index
    Adj_Data* host_train_data = (Adj_Data*)malloc(sizeof(Adj_Data) * ADJ_MAX_N_DATA); // train data
    int *host_n_appear_arr = (int*)malloc(sizeof(int) * eval_size);
    int *weight_arr = (int*)malloc(sizeof(int) * eval_size);
    double *host_error_monitor_arr = (double*)malloc(sizeof(double) * N_ERROR_MONITOR);
    double *host_val_error_monitor_arr = (double*)malloc(sizeof(double) * N_TEST_ERROR_MONITOR);
    double *host_regression_monitor_arr = (double*)malloc(sizeof(double) * N_REGRESSION_MONITOR);
    if (host_eval_arr == nullptr || host_rev_idx_arr == nullptr || host_train_data == nullptr ||
        host_n_appear_arr == nullptr || weight_arr == nullptr || host_error_monitor_arr == nullptr ||
        host_val_error_monitor_arr == nullptr || host_regression_monitor_arr == nullptr){
        std::cerr << "cannot allocate memory" << std::endl;
        return 1;
    }
    adj_init_arr(eval_size, host_eval_arr, host_rev_idx_arr, host_n_appear_arr);
    adj_import_eval(in_file, eval_size, host_eval_arr);
    std::pair<int, double> import_result = adj_import_data(n_train_data_file, train_files, host_train_data, host_rev_idx_arr, host_n_appear_arr);
    int n_all_data = import_result.first;
    double score_avg = import_result.second;
    std::cerr << n_all_data << " data loaded" << std::endl;
    // Fix validation membership while keeping the rounding RNG independent.
    // std::shuffle is deliberately avoided: its permutation may differ between
    // standard-library implementations even when the engine and seed match.
    adj_deterministic_shuffle(host_train_data, n_all_data, validation_seed);
    std::cerr << "data shuffled with toolchain-independent validation_seed " << validation_seed << std::endl;
    // divide data
    int n_val_data, n_train_data;
    Adj_Data* host_val_data;
    if (phase > 11){ // to phase 11, the all data available
        n_val_data = n_all_data / 20; // use 5% as validation data
        if (n_val_data <= 0){
            n_val_data = 1;
        }
        n_train_data  = n_all_data - n_val_data;
        host_val_data = host_train_data + n_train_data;
    } else {
        n_val_data = n_all_data; // use 100% train data
        n_train_data  = n_all_data; // use 100% val data
        host_val_data = host_train_data;
    }
    int validation_fingerprint_records = 0;
    const uint64_t validation_fingerprint = adj_validation_fingerprint(
        host_val_data, n_val_data, &validation_fingerprint_records
    );
    const std::string validation_mode = phase > 11 ? "fixed_shuffled_suffix_5pct" : "shared_all_not_holdout";
    std::cerr << "n_train_data " << n_train_data << " n_val_data " << n_val_data
              << " validation_mode " << validation_mode
              << " validation_seed " << validation_seed
              << " validation_fingerprint 0x" << std::hex << validation_fingerprint << std::dec
              << " validation_fingerprint_records " << validation_fingerprint_records
              << " round_seed " << round_seed << std::endl;
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
    for (int i = 0; i < eval_size; ++i) {
        weight_arr[i] = host_n_appear_arr[i];
    }
    for (int i = 0; i < eval_size; ++i) {
        host_n_appear_arr[i] = std::min(50, host_n_appear_arr[i]);
        // host_n_appear_arr[i] = std::max(N_APPEAR_MIN_VAL, host_n_appear_arr[i]);
    }
    std::cerr << "train data appearance calculated" << std::endl;

    double *device_eval_arr; // device eval array
    int *device_rev_idx_arr; // device reversed index
    Adj_Data *device_train_data;
    Adj_Data *device_val_data;
    int *device_n_appear_arr;
    double *device_residual_arr;
    double *device_error_monitor_arr;
    double *device_val_error_monitor_arr;
    double *device_regression_monitor_arr;
    int *device_start_idx_arr;
    adj_cuda_check(cudaMalloc(&device_eval_arr, sizeof(double) * eval_size), "allocate evaluation parameters");
    adj_cuda_check(cudaMalloc(&device_rev_idx_arr, sizeof(int) * eval_size), "allocate reverse indices");
    adj_cuda_check(cudaMalloc(&device_train_data, sizeof(Adj_Data) * n_train_data), "allocate training data");
    if (phase > 11) {
        adj_cuda_check(cudaMalloc(&device_val_data, sizeof(Adj_Data) * n_val_data), "allocate validation data");
    } else {
        device_val_data = device_train_data;
    }
    adj_cuda_check(cudaMalloc(&device_n_appear_arr, sizeof(int) * eval_size), "allocate appearance counts");
    adj_cuda_check(cudaMalloc(&device_residual_arr, sizeof(double) * eval_size), "allocate residuals");
    adj_cuda_check(cudaMalloc(&device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR), "allocate training monitor");
    adj_cuda_check(cudaMalloc(&device_val_error_monitor_arr, sizeof(double) * N_TEST_ERROR_MONITOR), "allocate validation monitor");
    adj_cuda_check(cudaMalloc(&device_regression_monitor_arr, sizeof(double) * N_REGRESSION_MONITOR), "allocate regression monitor");
    adj_cuda_check(cudaMalloc(&device_start_idx_arr, sizeof(int) * ADJ_N_FEATURES), "allocate feature offsets");
    adj_cuda_check(cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice), "copy evaluation parameters");
    adj_cuda_check(cudaMemcpy(device_rev_idx_arr, host_rev_idx_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice), "copy reverse indices");
    adj_cuda_check(cudaMemcpy(device_train_data, host_train_data, sizeof(Adj_Data) * n_train_data, cudaMemcpyHostToDevice), "copy training data");
    if (phase > 11) {
        adj_cuda_check(cudaMemcpy(device_val_data, host_val_data, sizeof(Adj_Data) * n_val_data, cudaMemcpyHostToDevice), "copy validation data");
    }
    adj_cuda_check(cudaMemcpy(device_n_appear_arr, host_n_appear_arr, sizeof(int) * eval_size, cudaMemcpyHostToDevice), "copy appearance counts");
    adj_cuda_check(cudaMemset(device_residual_arr, 0, sizeof(double) * eval_size), "initialize residuals");
    adj_cuda_check(cudaMemcpy(device_start_idx_arr, host_start_idx_arr, sizeof(int) * ADJ_N_FEATURES, cudaMemcpyHostToDevice), "copy feature offsets");

    // for adam optimizer
    double *device_m_arr;
    double *device_v_arr;
    adj_cuda_check(cudaMalloc(&device_m_arr, sizeof(double) * eval_size), "allocate Adam first moment");
    adj_cuda_check(cudaMalloc(&device_v_arr, sizeof(double) * eval_size), "allocate Adam second moment");
    adj_cuda_check(cudaMemset(device_m_arr, 0, sizeof(double) * eval_size), "initialize Adam first moment");
    adj_cuda_check(cudaMemset(device_v_arr, 0, sizeof(double) * eval_size), "initialize Adam second moment");
    
    const int n_blocks_val = (n_val_data + N_THREADS_PER_BLOCK_TEST - 1) / N_THREADS_PER_BLOCK_TEST;
    const int n_blocks_residual = (n_train_data + N_THREADS_PER_BLOCK_RESIDUAL - 1) / N_THREADS_PER_BLOCK_RESIDUAL;
    const int n_blocks_next_step = (eval_size + N_THREADS_PER_BLOCK_NEXT_STEP - 1) / N_THREADS_PER_BLOCK_NEXT_STEP;
    std::cerr << "n_blocks_val " << n_blocks_val << " n_blocks_residual " << n_blocks_residual << " n_blocks_next_step " << n_blocks_next_step << std::endl;
    std::cerr << "phase " << phase << std::endl;

    const std::string metrics_filename = output_dir + "/metrics_phase_" + std::to_string(phase) + ".csv";
    std::ofstream metrics_output(metrics_filename);
    if (!metrics_output.is_open()) {
        std::cerr << "cannot open metrics output " << metrics_filename << std::endl;
        return 1;
    }
    metrics_output
        << "phase,stage,loop,elapsed_ms,split,n,representation,alpha,validation_loss_increase,"
        << "mse,mae,bias_e_minus_z,teacher_mean,predicted_mean,"
        << "slope_e_on_z,intercept_e_on_z,slope_z_on_e,intercept_z_on_e,"
        << "correlation,ols_r_squared,prediction_r_squared,cov_e_with_z_minus_e\n";

    auto record_float_metrics = [&](const std::string &stage, int loop, uint64_t elapsed_ms,
                                    double current_alpha, int validation_loss_increase) {
        Adj_Regression_Summary train_summary = adj_collect_regression_metrics(
            device_eval_arr, n_train_data, device_start_idx_arr, device_train_data,
            device_regression_monitor_arr, host_regression_monitor_arr
        );
        adj_write_metrics_row(
            metrics_output, phase, stage, loop, elapsed_ms, "train", n_train_data,
            "float", current_alpha, validation_loss_increase, train_summary
        );
        if (phase <= 11) {
            adj_write_metrics_row(
                metrics_output, phase, stage, loop, elapsed_ms, "validation_shared_train", n_val_data,
                "float", current_alpha, validation_loss_increase, train_summary
            );
        } else {
            Adj_Regression_Summary validation_summary = adj_collect_regression_metrics(
                device_eval_arr, n_val_data, device_start_idx_arr, device_val_data,
                device_regression_monitor_arr, host_regression_monitor_arr
            );
            adj_write_metrics_row(
                metrics_output, phase, stage, loop, elapsed_ms, "validation", n_val_data,
                "float", current_alpha, validation_loss_increase, validation_summary
            );
        }
        return train_summary;
    };

    if (!reference_file.empty()) {
        double *host_reference_eval_arr = (double*)malloc(sizeof(double) * eval_size);
        if (host_reference_eval_arr == nullptr) {
            std::cerr << "cannot allocate reference parameters" << std::endl;
            return 1;
        }
        if (!adj_import_eval(reference_file, eval_size, host_reference_eval_arr)) {
            std::cerr << "cannot use reference parameters " << reference_file << std::endl;
            return 1;
        }
        adj_cuda_check(cudaMemcpy(device_eval_arr, host_reference_eval_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice), "copy reference parameters");
        record_float_metrics("deployed_reference", -1, 0, alpha, 0);
        adj_cuda_check(cudaMemcpy(device_eval_arr, host_eval_arr, sizeof(double) * eval_size, cudaMemcpyHostToDevice), "restore initial parameters");
        free(host_reference_eval_arr);
    }

    Adj_Regression_Summary initial_summary = record_float_metrics("initial", 0, 0, alpha, 0);
    std::cerr << "before MSE " << initial_summary.mse << " MAE " << initial_summary.mae
              << " slope_e_on_z " << initial_summary.predicted_on_teacher_slope
              << " slope_z_on_e " << initial_summary.teacher_on_predicted_slope << std::endl;
    uint64_t strt = tim();
    uint64_t metrics_time = 0;
    int n_loop = 0;
    double min_val_mse = 100000000.0;
    int n_val_loss_increase = 0;
    int n_val_loss_increase_reduce_lr = 0;
#if USE_WARMUP
    double alpha_stab = alpha / 5.0; // warming up for Adam
#else
    double alpha_stab = alpha;
#endif
    while (tim() - strt - metrics_time < msecond) {
        ++n_loop;

        // val loss
        adj_cuda_check(cudaMemset(device_val_error_monitor_arr, 0, sizeof(double) * N_TEST_ERROR_MONITOR), "initialize validation monitor");
        adj_calculate_val_loss <<<n_blocks_val, N_THREADS_PER_BLOCK_TEST>>> (device_eval_arr, n_val_data, device_start_idx_arr, device_val_data, device_val_error_monitor_arr);
        adj_cuda_check(cudaGetLastError(), "launch validation loss");
        adj_cuda_check(cudaMemcpy(host_val_error_monitor_arr, device_val_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy validation monitor");
        if (host_val_error_monitor_arr[0] <= min_val_mse){
            min_val_mse = host_val_error_monitor_arr[0];
            n_val_loss_increase = 0;
            n_val_loss_increase_reduce_lr = 0;
        } else{
            ++n_val_loss_increase;
            ++n_val_loss_increase_reduce_lr;
            if (n_val_loss_increase > n_patience){
                break;
            }
        }

        // train loss & residual
        adj_cuda_check(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR), "initialize training monitor");
        adj_calculate_residual <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (device_eval_arr, n_train_data, device_start_idx_arr, device_train_data, device_rev_idx_arr, device_residual_arr, device_error_monitor_arr);
        adj_cuda_check(cudaGetLastError(), "launch training residual");
        adj_cuda_check(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy training monitor");

        const uint64_t effective_elapsed = tim() - strt - metrics_time;
        std::cerr << "\rn_loop " << n_loop << " progress " << effective_elapsed * 100 / msecond << "% MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << "  val_MSE " << host_val_error_monitor_arr[0] << " val_MAE " << host_val_error_monitor_arr[1] << " val_loss_inc " << n_val_loss_increase << " alpha " << alpha_stab << "                    ";

        if (n_loop == 1 || n_loop % metrics_interval == 0) {
            const uint64_t metrics_start = tim();
            record_float_metrics("adam_pre_update", n_loop, effective_elapsed, alpha_stab, n_val_loss_increase);
            metrics_time += tim() - metrics_start;
        }
        
        // next step
        // gradient_descent <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab);
        // momentum <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, n_loop);
        // adagrad <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_v_arr, n_loop);
        adam <<<n_blocks_next_step, N_THREADS_PER_BLOCK_NEXT_STEP>>> (phase, eval_size, device_eval_arr, device_n_appear_arr, device_residual_arr, alpha_stab, device_m_arr, device_v_arr, n_loop);
        adj_cuda_check(cudaGetLastError(), "launch Adam update");
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

    const uint64_t adam_elapsed = tim() - strt - metrics_time;
    Adj_Regression_Summary final_float_summary = record_float_metrics(
        "adam_final_float", n_loop, adam_elapsed, alpha_stab, n_val_loss_increase
    );
    std::cerr << "Adam final float MSE " << final_float_summary.mse
              << " MAE " << final_float_summary.mae
              << " slope_e_on_z " << final_float_summary.predicted_on_teacher_slope
              << " slope_z_on_e " << final_float_summary.teacher_on_predicted_slope
              << " cov_e_with_z_minus_e " << final_float_summary.predicted_residual_covariance
              << std::endl;

    // init round eval with hillclimb
    adj_cuda_check(cudaMemcpy(host_eval_arr, device_eval_arr, sizeof(double) * eval_size, cudaMemcpyDeviceToHost), "copy final float parameters");
    adj_output_float_param(phase, eval_size, host_eval_arr, output_dir);
    int *host_eval_arr_roundup = (int*)malloc(sizeof(int) * eval_size);
    int *host_eval_arr_rounddown = (int*)malloc(sizeof(int) * eval_size);
    bool *host_round_arr = (bool*)malloc(sizeof(bool) * eval_size);
    int n_roundup = 0, n_rounddown = 0;
    for (int i = 0; i < eval_size; ++i){
        host_eval_arr_roundup[i] = static_cast<int>(std::ceil(host_eval_arr[i]));
        host_eval_arr_rounddown[i] = static_cast<int>(std::floor(host_eval_arr[i]));
        host_round_arr[i] = (round(host_eval_arr[i]) == host_eval_arr_rounddown[i]); // 0 for round-up, 1 for round-down
        if (host_eval_arr_roundup[i] != host_eval_arr_rounddown[i]){
            if (!host_round_arr[i]){
                ++n_roundup;
            } else{
                ++n_rounddown;
            }
        }
    }
    std::cerr << "n_roundup " << n_roundup << " n_rounddown " << n_rounddown << std::endl;
    int *device_eval_arr_roundup;
    int *device_eval_arr_rounddown;
    bool *device_round_arr;
    adj_cuda_check(cudaMalloc(&device_eval_arr_roundup, sizeof(int) * eval_size), "allocate rounded-up parameters");
    adj_cuda_check(cudaMalloc(&device_eval_arr_rounddown, sizeof(int) * eval_size), "allocate rounded-down parameters");
    adj_cuda_check(cudaMalloc(&device_round_arr, sizeof(bool) * eval_size), "allocate rounding choices");
    adj_cuda_check(cudaMemcpy(device_eval_arr_roundup, host_eval_arr_roundup, sizeof(int) * eval_size, cudaMemcpyHostToDevice), "copy rounded-up parameters");
    adj_cuda_check(cudaMemcpy(device_eval_arr_rounddown, host_eval_arr_rounddown, sizeof(int) * eval_size, cudaMemcpyHostToDevice), "copy rounded-down parameters");
    adj_cuda_check(cudaMemcpy(device_round_arr, host_round_arr, sizeof(bool) * eval_size, cudaMemcpyHostToDevice), "copy rounding choices");

    auto record_round_metrics = [&](const std::string &stage, int loop, uint64_t elapsed_ms,
                                    double current_alpha, int validation_loss_increase) {
        Adj_Regression_Summary train_summary = adj_collect_regression_metrics_round(
            device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr,
            n_train_data, device_start_idx_arr, device_train_data,
            device_regression_monitor_arr, host_regression_monitor_arr
        );
        adj_write_metrics_row(
            metrics_output, phase, stage, loop, elapsed_ms, "train", n_train_data,
            "integer", current_alpha, validation_loss_increase, train_summary
        );
        if (phase <= 11) {
            adj_write_metrics_row(
                metrics_output, phase, stage, loop, elapsed_ms, "validation_shared_train", n_val_data,
                "integer", current_alpha, validation_loss_increase, train_summary
            );
        } else {
            Adj_Regression_Summary validation_summary = adj_collect_regression_metrics_round(
                device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr,
                n_val_data, device_start_idx_arr, device_val_data,
                device_regression_monitor_arr, host_regression_monitor_arr
            );
            adj_write_metrics_row(
                metrics_output, phase, stage, loop, elapsed_ms, "validation", n_val_data,
                "integer", current_alpha, validation_loss_increase, validation_summary
            );
        }
        return train_summary;
    };

    Adj_Regression_Summary nearest_round_summary = record_round_metrics(
        "rounded_nearest", n_loop, adam_elapsed, alpha_stab, n_val_loss_increase
    );
    std::cerr << "nearest rounding MSE " << nearest_round_summary.mse
              << " MAE " << nearest_round_summary.mae
              << " slope_e_on_z " << nearest_round_summary.predicted_on_teacher_slope
              << " slope_z_on_e " << nearest_round_summary.teacher_on_predicted_slope << std::endl;

    // round eval with hillclimb
    adj_cuda_check(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR), "initialize nearest-rounding monitor");
    adj_calculate_loss_round <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (-1, -1, device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr, n_train_data, device_start_idx_arr, device_train_data, device_error_monitor_arr);
    adj_cuda_check(cudaGetLastError(), "launch nearest-rounding loss");
    adj_cuda_check(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy nearest-rounding monitor");
    double min_mse = host_error_monitor_arr[0], min_mae = host_error_monitor_arr[1];
    std::cerr << "before rounding MSE " << min_mse << " MAE " << min_mae << std::endl;
    std::uniform_int_distribution<int> randint_eval(0, eval_size - 1); // [0, eval_size - 1] (include last)
    uint64_t round_n_loop = 0, round_n_updated = 0, round_n_improve = 0;
    uint64_t round_strt = tim();
    uint64_t round_tl = static_cast<uint64_t>(round_seconds) * 1000;
    while (tim() - round_strt < round_tl && ((double)round_n_improve * 100.0 / round_n_loop > 0.01 || round_n_loop < 100)){ // improve percentage > 1% or loop_count < 100
        int change_idx = randint_eval(round_engine);
        if (host_eval_arr_roundup[change_idx] != host_eval_arr_rounddown[change_idx]){
            int rev_change_idx = host_rev_idx_arr[change_idx];
            adj_cuda_check(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR), "initialize rounding candidate monitor");
            adj_calculate_loss_round <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (change_idx, rev_change_idx, device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr, n_train_data, device_start_idx_arr, device_train_data, device_error_monitor_arr);
            adj_cuda_check(cudaGetLastError(), "launch rounding candidate loss");
            adj_cuda_check(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy rounding candidate monitor");
            if (host_error_monitor_arr[0] <= min_mse){
                ++round_n_updated;
                if (host_error_monitor_arr[0] < min_mse){
                    ++round_n_improve;
                }
                min_mse = host_error_monitor_arr[0];
                min_mae = host_error_monitor_arr[1];
                host_round_arr[change_idx] ^= 1; // change round-up/down
                if (change_idx != rev_change_idx){
                    host_round_arr[rev_change_idx] ^= 1; // change round-up/down
                }
                adj_cuda_check(cudaMemcpy(device_round_arr, host_round_arr, sizeof(bool) * eval_size, cudaMemcpyHostToDevice), "update rounding choices"); // copy less
            }
            ++round_n_loop;
            uint64_t percent = (tim() - round_strt) * 100 / round_tl;
            std::cerr << '\r' << "rounding " << percent << "%" << " n_loop " << round_n_loop << " n_updated " << round_n_updated << " n_improve " << round_n_improve << " MSE " << min_mse << " MAE " << min_mae << "                         ";
        }
    }
    std::cerr << std::endl;

    // round eval arr with hillclimb result
    n_roundup = 0;
    n_rounddown = 0;
    for (int i = 0; i < eval_size; ++i){
        if (!host_round_arr[i]){ // round-up
            host_eval_arr[i] = host_eval_arr_roundup[i];
        } else{ // round_down
            host_eval_arr[i] = host_eval_arr_rounddown[i];
        }
        if (host_eval_arr_roundup[i] != host_eval_arr_rounddown[i]){
            if (!host_round_arr[i]){
                ++n_roundup;
            } else{
                ++n_rounddown;
            }
        }
    }
    std::cerr << "n_roundup " << n_roundup << " n_rounddown " << n_rounddown << std::endl;

    Adj_Regression_Summary final_round_summary = record_round_metrics(
        "rounded_hillclimb_final", n_loop, adam_elapsed + tim() - round_strt,
        alpha_stab, n_val_loss_increase
    );
    std::cerr << "hillclimb final MSE " << final_round_summary.mse
              << " MAE " << final_round_summary.mae
              << " slope_e_on_z " << final_round_summary.predicted_on_teacher_slope
              << " slope_z_on_e " << final_round_summary.teacher_on_predicted_slope
              << " cov_e_with_z_minus_e " << final_round_summary.predicted_residual_covariance
              << std::endl;

    // calculate final loss
    adj_cuda_check(cudaMemset(device_error_monitor_arr, 0, sizeof(double) * N_ERROR_MONITOR), "initialize final training monitor");
    adj_calculate_loss_round <<<n_blocks_residual, N_THREADS_PER_BLOCK_RESIDUAL>>> (-1, -1, device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr, n_train_data, device_start_idx_arr, device_train_data, device_error_monitor_arr);
    adj_cuda_check(cudaGetLastError(), "launch final training loss");
    adj_cuda_check(cudaMemcpy(host_error_monitor_arr, device_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy final training monitor");
    adj_cuda_check(cudaMemset(device_val_error_monitor_arr, 0, sizeof(double) * N_TEST_ERROR_MONITOR), "initialize final validation monitor");
    adj_calculate_loss_round <<<n_blocks_val, N_THREADS_PER_BLOCK_TEST>>> (-1, -1, device_eval_arr_roundup, device_eval_arr_rounddown, device_round_arr, n_val_data, device_start_idx_arr, device_val_data, device_val_error_monitor_arr);
    adj_cuda_check(cudaGetLastError(), "launch final validation loss");
    adj_cuda_check(cudaMemcpy(host_val_error_monitor_arr, device_val_error_monitor_arr, sizeof(double) * N_ERROR_MONITOR, cudaMemcpyDeviceToHost), "copy final validation monitor");

    // output param
    adj_output_param(phase, eval_size, host_eval_arr, output_dir);
    adj_output_weight(phase, eval_size, weight_arr, output_dir);

    std::cerr << "phase " << phase << " time " << (tim() - strt) << " ms n_train_data " << n_train_data << " n_val_data " << n_val_data << " score_avg " << score_avg << " n_loop " << n_loop << " MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << " val_MSE " << host_val_error_monitor_arr[0] << " val_MAE " << host_val_error_monitor_arr[1] << " (with int) alpha " << alpha_in << " n_patience " << n_patience << " reduce_lr_patience " << reduce_lr_patience << " reduce_lr_ratio " << reduce_lr_ratio << std::endl;
    std::cout << "phase " << phase << " time " << (tim() - strt) << " ms n_train_data " << n_train_data << " n_val_data " << n_val_data << " score_avg " << score_avg << " n_loop " << n_loop << " MSE " << host_error_monitor_arr[0] << " MAE " << host_error_monitor_arr[1] << " val_MSE " << host_val_error_monitor_arr[0] << " val_MAE " << host_val_error_monitor_arr[1] << " (with int) alpha " << alpha_in << " n_patience " << n_patience << " reduce_lr_patience " << reduce_lr_patience << " reduce_lr_ratio " << reduce_lr_ratio << std::endl;

    return 0;
}
