/*
    Egaroucid Project

    @file eval_optimizer_fm_joint_cuda.cu
        CUDA joint optimizer for a linear + shared Factorization Machine evaluation
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <atomic>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#define OPTIMIZER_INCLUDE
#include "evaluation_definition.hpp"

constexpr char FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr uint32_t FM_FILE_VERSION = 1;
constexpr int N_ZEROS_PLUS_LOCAL = 1 << 12;
constexpr int FM_N_PATTERN_FEATURES = ADJ_N_FEATURES - 1;
constexpr uint64_t INDEXED_PHASE_RECORD_BYTES =
    sizeof(int16_t) + sizeof(int16_t) + sizeof(uint16_t) * ADJ_N_FEATURES + sizeof(int16_t);
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr uint64_t SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15ULL;
constexpr double TWO_PI = 6.283185307179586476925286766559;

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } \
} while (0)

#define CUDA_CHECK_MAIN(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return 1; \
    } \
} while (0)

struct Options {
    std::string base_eval;
    std::string data_root;
    std::string out_file;
    int dim = 2;
    int epochs = 1;
    uint64_t train_samples = 0;
    uint64_t val_samples = 0;
    uint64_t batch_size = 1000000;
    double linear_lr = 0.1;
    double fm_lr = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double adam_eps = 1.0e-8;
    double linear_l2 = 0.0;
    double fm_l2 = 0.0;
    double grad_clip_raw = 0.0;
    double linear_param_clip = ADJ_EVAL_PARAM_MAX;
    double fm_vector_clip = 127.0;
    double init_std = 0.01;
    int scale = 128;
    uint64_t seed = 20260724ULL;
    int record_start = 223;
    int record_end = -1;
    int phase_start = 0;
    int phase_end = ADJ_N_PHASES - 1;
    int early_stop_patience = 100;
    double max_memory_gib = 100.0;
    uint64_t train_metric_limit = 1000000;
    uint64_t val_metric_limit = 0;
    int progress_interval_sec = 30;
    std::string read_mode = "bulk";
    int read_threads = 4;
    bool dry_run = false;
};

struct IndexedDatum {
    int16_t n_discs = 0;
    int16_t player = 0;
    uint16_t features[ADJ_N_FEATURES] = {};
    int16_t score = 0;
};

struct Sample {
    uint16_t features[ADJ_N_FEATURES];
    int16_t score;
    uint16_t phase;
};

struct FileEntry {
    std::filesystem::path path;
    int phase = 0;
    int record = 0;
    uint64_t records = 0;
    uint64_t begin = 0;
};

struct SampleRequest {
    uint64_t global_index = 0;
    bool validation = false;
};

struct Loss {
    double mse = 0.0;
    double mae = 0.0;
    uint64_t n = 0;
};

struct SampleStats {
    uint64_t bad_feature_count = 0;
    uint64_t phase_mismatch_count = 0;
    std::array<uint64_t, ADJ_N_PHASES> train_phase_counts = {};
    std::array<uint64_t, ADJ_N_PHASES> val_phase_counts = {};
};

__constant__ int c_linear_starts[ADJ_N_FEATURES];
__constant__ int c_fm_offsets[FM_N_PATTERN_FEATURES];

uint64_t tim_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

template<typename T>
bool read_scalar(FILE *fp, T *v) {
    return fread(v, sizeof(T), 1, fp) == 1;
}

template<typename T>
void write_scalar(std::ofstream &out, const T &v) {
    out.write((const char*)&v, sizeof(T));
}

void usage() {
    std::cerr
        << "usage: eval_optimizer_fm_joint_cuda"
        << " --base-eval bin/resources/eval.egev2"
        << " --data-root E:/egaroucid_data/train_data/bin_data/20241125_1"
        << " --out-file model/.../eval.egevfm"
        << " --dim 2 --epochs 10 --train-samples N --val-samples N"
        << " [--batch-size 1000000] [--linear-lr 0.1] [--fm-lr 0.01]"
        << " [--record-start 223] [--record-end -1] [--scale 128]"
        << " [--read-mode bulk] [--read-threads 4] [--progress-interval-sec 30]"
        << " [--max-memory-gib 100] [--dry-run 0]\n";
}

bool parse_bool(const std::string &s) {
    return s == "1" || s == "true" || s == "True" || s == "yes" || s == "on";
}

bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] missing value for " << key << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else if (key == "--base-eval") {
            const char *v = need_value(); if (!v) return false; opt->base_eval = v;
        } else if (key == "--data-root") {
            const char *v = need_value(); if (!v) return false; opt->data_root = v;
        } else if (key == "--out-file") {
            const char *v = need_value(); if (!v) return false; opt->out_file = v;
        } else if (key == "--dim") {
            const char *v = need_value(); if (!v) return false; opt->dim = std::atoi(v);
        } else if (key == "--epochs") {
            const char *v = need_value(); if (!v) return false; opt->epochs = std::atoi(v);
        } else if (key == "--train-samples") {
            const char *v = need_value(); if (!v) return false; opt->train_samples = std::strtoull(v, nullptr, 10);
        } else if (key == "--val-samples") {
            const char *v = need_value(); if (!v) return false; opt->val_samples = std::strtoull(v, nullptr, 10);
        } else if (key == "--batch-size") {
            const char *v = need_value(); if (!v) return false; opt->batch_size = std::strtoull(v, nullptr, 10);
        } else if (key == "--linear-lr") {
            const char *v = need_value(); if (!v) return false; opt->linear_lr = std::atof(v);
        } else if (key == "--fm-lr") {
            const char *v = need_value(); if (!v) return false; opt->fm_lr = std::atof(v);
        } else if (key == "--beta1") {
            const char *v = need_value(); if (!v) return false; opt->beta1 = std::atof(v);
        } else if (key == "--beta2") {
            const char *v = need_value(); if (!v) return false; opt->beta2 = std::atof(v);
        } else if (key == "--adam-eps") {
            const char *v = need_value(); if (!v) return false; opt->adam_eps = std::atof(v);
        } else if (key == "--linear-l2") {
            const char *v = need_value(); if (!v) return false; opt->linear_l2 = std::atof(v);
        } else if (key == "--fm-l2") {
            const char *v = need_value(); if (!v) return false; opt->fm_l2 = std::atof(v);
        } else if (key == "--grad-clip-raw") {
            const char *v = need_value(); if (!v) return false; opt->grad_clip_raw = std::atof(v);
        } else if (key == "--linear-param-clip") {
            const char *v = need_value(); if (!v) return false; opt->linear_param_clip = std::atof(v);
        } else if (key == "--fm-vector-clip") {
            const char *v = need_value(); if (!v) return false; opt->fm_vector_clip = std::atof(v);
        } else if (key == "--init-std") {
            const char *v = need_value(); if (!v) return false; opt->init_std = std::atof(v);
        } else if (key == "--scale") {
            const char *v = need_value(); if (!v) return false; opt->scale = std::atoi(v);
        } else if (key == "--seed") {
            const char *v = need_value(); if (!v) return false; opt->seed = std::strtoull(v, nullptr, 10);
        } else if (key == "--record-start") {
            const char *v = need_value(); if (!v) return false; opt->record_start = std::atoi(v);
        } else if (key == "--record-end") {
            const char *v = need_value(); if (!v) return false; opt->record_end = std::atoi(v);
        } else if (key == "--phase-start") {
            const char *v = need_value(); if (!v) return false; opt->phase_start = std::atoi(v);
        } else if (key == "--phase-end") {
            const char *v = need_value(); if (!v) return false; opt->phase_end = std::atoi(v);
        } else if (key == "--early-stop-patience") {
            const char *v = need_value(); if (!v) return false; opt->early_stop_patience = std::atoi(v);
        } else if (key == "--max-memory-gib") {
            const char *v = need_value(); if (!v) return false; opt->max_memory_gib = std::atof(v);
        } else if (key == "--train-metric-limit") {
            const char *v = need_value(); if (!v) return false; opt->train_metric_limit = std::strtoull(v, nullptr, 10);
        } else if (key == "--val-metric-limit") {
            const char *v = need_value(); if (!v) return false; opt->val_metric_limit = std::strtoull(v, nullptr, 10);
        } else if (key == "--progress-interval-sec") {
            const char *v = need_value(); if (!v) return false; opt->progress_interval_sec = std::atoi(v);
        } else if (key == "--read-mode") {
            const char *v = need_value(); if (!v) return false; opt->read_mode = v;
        } else if (key == "--read-threads") {
            const char *v = need_value(); if (!v) return false; opt->read_threads = std::atoi(v);
        } else if (key == "--dry-run") {
            const char *v = need_value(); if (!v) return false; opt->dry_run = parse_bool(v);
        } else {
            std::cerr << "[ERROR] unknown option " << key << "\n";
            return false;
        }
    }
    return true;
}

bool validate_options(const Options &opt) {
    if (opt.base_eval.empty() || opt.data_root.empty() || opt.out_file.empty()) {
        std::cerr << "[ERROR] --base-eval, --data-root and --out-file are required\n";
        return false;
    }
    if (opt.dim <= 0 || opt.dim > 64 || opt.epochs < 0 || opt.train_samples == 0 ||
        opt.val_samples == 0 || opt.batch_size == 0 || opt.scale <= 0) {
        std::cerr << "[ERROR] invalid dim/epochs/sample/batch/scale option\n";
        return false;
    }
    if (opt.phase_start < 0 || opt.phase_end < opt.phase_start || opt.phase_end >= ADJ_N_PHASES ||
        opt.record_start < 0 || (opt.record_end >= 0 && opt.record_end < opt.record_start)) {
        std::cerr << "[ERROR] invalid phase or record range\n";
        return false;
    }
    if (opt.linear_lr <= 0.0 || opt.fm_lr <= 0.0 || opt.beta1 < 0.0 || opt.beta1 >= 1.0 ||
        opt.beta2 < 0.0 || opt.beta2 >= 1.0 || opt.adam_eps <= 0.0 || opt.init_std <= 0.0 ||
        opt.linear_l2 < 0.0 || opt.fm_l2 < 0.0 || opt.grad_clip_raw < 0.0 ||
        opt.linear_param_clip <= 0.0 || opt.fm_vector_clip <= 0.0 || opt.max_memory_gib <= 0.0 ||
        opt.early_stop_patience < 0 || opt.progress_interval_sec < 0 || opt.read_threads <= 0) {
        std::cerr << "[ERROR] invalid optimizer option\n";
        return false;
    }
    if (opt.read_mode != "bulk" && opt.read_mode != "scan" && opt.read_mode != "seek") {
        std::cerr << "[ERROR] read-mode must be bulk, scan or seek\n";
        return false;
    }
    return true;
}

std::vector<int16_t> load_unzip_egev2_local(const std::string &file) {
    FILE *fp = std::fopen(file.c_str(), "rb");
    if (fp == nullptr) {
        std::cerr << "[ERROR] can't open base eval " << file << "\n";
        return {};
    }
    int32_t n_zipped = 0;
    if (!read_scalar(fp, &n_zipped) || n_zipped <= 0) {
        std::cerr << "[ERROR] base eval header broken " << file << "\n";
        std::fclose(fp);
        return {};
    }
    std::vector<int16_t> zipped((size_t)n_zipped);
    if (fread(zipped.data(), sizeof(int16_t), zipped.size(), fp) < zipped.size()) {
        std::cerr << "[ERROR] base eval payload broken " << file << "\n";
        std::fclose(fp);
        return {};
    }
    std::fclose(fp);

    std::vector<int16_t> res;
    res.reserve((size_t)ADJ_N_PHASES * 700000);
    for (const int16_t elem: zipped) {
        if (elem >= N_ZEROS_PLUS_LOCAL) {
            res.insert(res.end(), elem - N_ZEROS_PLUS_LOCAL, 0);
        } else {
            res.emplace_back(elem);
        }
    }
    return res;
}

std::array<int, ADJ_N_FEATURES> make_linear_starts() {
    std::array<int, ADJ_N_FEATURES> starts = {};
    int start = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        if (i > 0 && adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            start += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
        starts[(size_t)i] = start;
    }
    return starts;
}

std::array<int, FM_N_PATTERN_FEATURES> make_fm_feature_offsets(uint64_t *total_vectors) {
    std::array<int, FM_N_PATTERN_FEATURES> offsets = {};
    int offset = 0;
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        offsets[(size_t)i] = offset;
        offset += adj_eval_sizes[adj_feature_to_eval_idx[i]];
    }
    *total_vectors = (uint64_t)offset;
    return offsets;
}

int linear_params_per_phase() {
    int res = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        res += adj_eval_sizes[i];
    }
    return res;
}

bool parse_record_number(const std::filesystem::path &path, int *record) {
    if (path.extension() != ".dat") {
        return false;
    }
    const std::string stem = path.stem().string();
    if (stem.empty()) {
        return false;
    }
    for (const char c: stem) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    *record = std::atoi(stem.c_str());
    return true;
}

std::vector<FileEntry> build_manifest(const Options &opt, std::array<uint64_t, ADJ_N_PHASES> *phase_counts) {
    std::vector<FileEntry> entries;
    uint64_t total = 0;
    uint64_t bad_size_files = 0;
    for (int phase = opt.phase_start; phase <= opt.phase_end; ++phase) {
        const std::filesystem::path phase_dir = std::filesystem::path(opt.data_root) / std::to_string(phase);
        if (!std::filesystem::exists(phase_dir)) {
            std::cerr << "[WARN] missing phase directory " << phase_dir.string() << "\n";
            continue;
        }
        std::vector<std::filesystem::path> files;
        for (const auto &it: std::filesystem::directory_iterator(phase_dir)) {
            if (!it.is_regular_file()) {
                continue;
            }
            int record = 0;
            if (!parse_record_number(it.path(), &record)) {
                continue;
            }
            if (record < opt.record_start || (opt.record_end >= 0 && record > opt.record_end)) {
                continue;
            }
            files.emplace_back(it.path());
        }
        std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) {
            int ra = 0, rb = 0;
            parse_record_number(a, &ra);
            parse_record_number(b, &rb);
            return ra < rb;
        });
        for (const auto &file: files) {
            const uint64_t bytes = (uint64_t)std::filesystem::file_size(file);
            if (bytes == 0) {
                continue;
            }
            if (bytes % INDEXED_PHASE_RECORD_BYTES != 0) {
                ++bad_size_files;
                std::cerr << "[ERROR] bad indexed data size " << file.string()
                          << " bytes " << bytes
                          << " record_bytes " << INDEXED_PHASE_RECORD_BYTES << "\n";
                continue;
            }
            int record = 0;
            parse_record_number(file, &record);
            FileEntry entry;
            entry.path = file;
            entry.phase = phase;
            entry.record = record;
            entry.records = bytes / INDEXED_PHASE_RECORD_BYTES;
            entry.begin = total;
            entries.emplace_back(entry);
            total += entry.records;
            (*phase_counts)[(size_t)phase] += entry.records;
        }
    }
    if (bad_size_files != 0) {
        entries.clear();
    }
    return entries;
}

uint64_t manifest_total_count(const std::vector<FileEntry> &entries) {
    if (entries.empty()) {
        return 0;
    }
    const FileEntry &last = entries.back();
    return last.begin + last.records;
}

double bytes_to_gib(const long double bytes) {
    return (double)(bytes / (1024.0L * 1024.0L * 1024.0L));
}

uint64_t mix_u64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t next_splitmix64(uint64_t *state) {
    *state += SPLITMIX64_INCREMENT;
    return mix_u64(*state);
}

uint64_t bounded_random_u64(uint64_t *state, const uint64_t bound) {
    if (bound <= 1) {
        return 0;
    }
    const uint64_t threshold = (0ULL - bound) % bound;
    while (true) {
        const uint64_t x = next_splitmix64(state);
        if (x >= threshold) {
            return x % bound;
        }
    }
}

double unit_from_u64(const uint64_t x) {
    return ((double)(x >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

float deterministic_normal(const uint64_t seed, const uint64_t index, const double stddev) {
    if (stddev <= 0.0) {
        return 0.0f;
    }
    const uint64_t x0 = mix_u64(seed ^ (index * 0xD6E8FEB86659FD93ULL));
    const uint64_t x1 = mix_u64((seed + 0xA5A5A5A5A5A5A5A5ULL) ^ (index * 0x9E3779B97F4A7C15ULL));
    const double u0 = std::max(unit_from_u64(x0), std::numeric_limits<double>::min());
    const double u1 = unit_from_u64(x1);
    const double z0 = std::sqrt(-2.0 * std::log(u0)) * std::cos(TWO_PI * u1);
    return (float)(z0 * stddev);
}

uint64_t sample_split_key(const uint64_t position, const uint64_t seed) {
    return mix_u64(position ^ (seed + 0xBADC0FFEE0DDF00DULL));
}

template <typename T>
void deterministic_shuffle(std::vector<T> *values, const uint64_t seed) {
    uint64_t state = seed;
    for (uint64_t i = (uint64_t)values->size(); i > 1; --i) {
        const uint64_t j = bounded_random_u64(&state, i);
        std::swap((*values)[(size_t)(i - 1)], (*values)[(size_t)j]);
    }
}

struct HostMemoryEstimate {
    uint64_t request_generation_bytes = 0;
    uint64_t sample_loading_bytes = 0;
    uint64_t training_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t bulk_buffer_bytes = 0;
};

uint64_t manifest_max_file_records(const std::vector<FileEntry> &entries) {
    uint64_t res = 0;
    for (const FileEntry &entry: entries) {
        res = std::max<uint64_t>(res, entry.records);
    }
    return res;
}

HostMemoryEstimate estimate_host_memory_bytes(
    const Options &opt,
    const uint64_t linear_count,
    const uint64_t fm_count,
    const std::vector<FileEntry> &manifest
) {
    const uint64_t sample_count = opt.train_samples + opt.val_samples;
    const uint64_t max_file_records = manifest_max_file_records(manifest);
    HostMemoryEstimate res;
    res.bulk_buffer_bytes = opt.read_mode == "bulk"
        ? max_file_records * INDEXED_PHASE_RECORD_BYTES * (uint64_t)std::max(1, opt.read_threads)
        : 0;
    const long double base_eval_bytes = (long double)linear_count * sizeof(int16_t);
    const long double request_bytes = (long double)sample_count * sizeof(SampleRequest);
    const long double sample_bytes = (long double)sample_count * sizeof(Sample);
    const long double params_bytes = (long double)(linear_count + fm_count) * sizeof(float);
    const long double overhead_bytes = 512.0L * 1024.0L * 1024.0L;
    const long double request_generation_work_bytes = std::max(
        (long double)sample_count * (40.0L + (long double)sizeof(uint64_t)),
        (long double)sample_count * (sizeof(std::pair<uint64_t, uint64_t>) + sizeof(SampleRequest))
    );
    const long double max_group_duplicate_bytes = (long double)max_file_records * sizeof(Sample);

    res.request_generation_bytes = (uint64_t)(base_eval_bytes + request_generation_work_bytes + overhead_bytes);
    res.sample_loading_bytes = (uint64_t)(
        base_eval_bytes + request_bytes + sample_bytes +
        (long double)res.bulk_buffer_bytes + max_group_duplicate_bytes + overhead_bytes
    );
    res.training_bytes = (uint64_t)(base_eval_bytes + sample_bytes + params_bytes + overhead_bytes);
    res.peak_bytes = std::max(res.request_generation_bytes, std::max(res.sample_loading_bytes, res.training_bytes));
    return res;
}

uint64_t estimate_device_memory_bytes(const Options &opt, const uint64_t linear_count, const uint64_t fm_count) {
    long double bytes = 0.0L;
    bytes += (long double)linear_count * 5.0L * sizeof(float);
    bytes += (long double)fm_count * 5.0L * sizeof(float);
    bytes += (long double)opt.batch_size * sizeof(Sample);
    bytes += 256.0L * 1024.0L * 1024.0L;
    return (uint64_t)bytes;
}

std::vector<SampleRequest> generate_requests(
    const uint64_t total_records,
    const uint64_t train_samples,
    const uint64_t val_samples,
    const uint64_t seed,
    const int progress_interval_sec
) {
    const uint64_t need = train_samples + val_samples;
    const uint64_t start_ms = tim_ms();
    uint64_t next_log_ms = start_ms + (uint64_t)progress_interval_sec * 1000ULL;
    std::unordered_set<uint64_t> chosen;
    chosen.reserve((size_t)(need * 13 / 10 + 1024));
    uint64_t sampling_state = seed ^ 0x123456789ABCDEF0ULL;
    while ((uint64_t)chosen.size() < need) {
        chosen.insert(bounded_random_u64(&sampling_state, total_records));
        const uint64_t now = tim_ms();
        if (progress_interval_sec > 0 && now >= next_log_ms) {
            std::cerr << "sample_request_generation chosen " << chosen.size()
                      << " / " << need
                      << " elapsed_ms " << (now - start_ms) << "\n";
            next_log_ms = now + (uint64_t)progress_interval_sec * 1000ULL;
        }
    }
    std::cerr << "sample_request_generation chosen " << chosen.size()
              << " / " << need
              << " elapsed_ms " << (tim_ms() - start_ms)
              << " sorting 1\n";

    std::vector<uint64_t> positions;
    positions.reserve((size_t)need);
    for (const uint64_t pos: chosen) {
        positions.emplace_back(pos);
    }
    chosen.clear();
    chosen.rehash(0);

    std::sort(positions.begin(), positions.end());
    std::vector<std::pair<uint64_t, uint64_t>> split_order;
    split_order.reserve(positions.size());
    for (const uint64_t pos: positions) {
        split_order.emplace_back(sample_split_key(pos, seed), pos);
    }
    positions.clear();
    positions.shrink_to_fit();
    std::sort(split_order.begin(), split_order.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    std::vector<SampleRequest> requests;
    requests.reserve((size_t)need);
    for (uint64_t i = 0; i < need; ++i) {
        requests.push_back({split_order[(size_t)i].second, i >= train_samples});
    }
    split_order.clear();
    split_order.shrink_to_fit();
    std::sort(requests.begin(), requests.end(), [](const SampleRequest &a, const SampleRequest &b) {
        return a.global_index < b.global_index;
    });
    std::cerr << "sample_request_generation sorted " << requests.size()
              << " elapsed_ms " << (tim_ms() - start_ms) << "\n";
    return requests;
}

bool validate_features(const IndexedDatum &datum) {
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        const int eval_idx = adj_feature_to_eval_idx[i];
        if (datum.features[i] >= adj_eval_sizes[eval_idx]) {
            return false;
        }
    }
    return true;
}

bool decode_indexed_record(const char *ptr, IndexedDatum *datum) {
    std::memcpy(&datum->n_discs, ptr, sizeof(int16_t));
    ptr += sizeof(int16_t);
    std::memcpy(&datum->player, ptr, sizeof(int16_t));
    ptr += sizeof(int16_t);
    std::memcpy(datum->features, ptr, sizeof(uint16_t) * ADJ_N_FEATURES);
    ptr += sizeof(uint16_t) * ADJ_N_FEATURES;
    std::memcpy(&datum->score, ptr, sizeof(int16_t));
    return true;
}

bool read_file_span(
    const std::filesystem::path &path,
    const uint64_t offset,
    const uint64_t bytes,
    std::vector<char> *buffer
) {
    constexpr uint64_t READ_CHUNK_BYTES = 256ULL * 1024ULL * 1024ULL;
    buffer->resize((size_t)bytes);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open indexed data " << path.string() << "\n";
        return false;
    }
    in.seekg((std::streamoff)offset, std::ios::beg);
    if (!in) {
        std::cerr << "[ERROR] can't seek indexed data " << path.string()
                  << " offset " << offset << "\n";
        return false;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t step = std::min<uint64_t>(bytes - done, READ_CHUNK_BYTES);
        in.read(buffer->data() + (size_t)done, (std::streamsize)step);
        if (!in) {
            std::cerr << "[ERROR] can't bulk-read indexed data " << path.string()
                      << " offset " << (offset + done)
                      << " bytes " << step << "\n";
            return false;
        }
        done += step;
    }
    return true;
}

void append_loaded_sample(
    const IndexedDatum &datum,
    const FileEntry &entry,
    const bool validation,
    std::vector<Sample> *train,
    std::vector<Sample> *val,
    SampleStats *stats
) {
    if (!validate_features(datum)) {
        ++stats->bad_feature_count;
        return;
    }
    if ((int)datum.n_discs - 4 != entry.phase) {
        ++stats->phase_mismatch_count;
    }
    Sample sample;
    std::memcpy(sample.features, datum.features, sizeof(uint16_t) * ADJ_N_FEATURES);
    sample.score = datum.score;
    sample.phase = (uint16_t)entry.phase;
    if (validation) {
        val->push_back(sample);
        ++stats->val_phase_counts[(size_t)entry.phase];
    } else {
        train->push_back(sample);
        ++stats->train_phase_counts[(size_t)entry.phase];
    }
}

struct SampleRequestGroup {
    size_t entry_idx = 0;
    size_t first_request = 0;
    size_t last_request = 0;
    uint64_t first_local = 0;
    uint64_t last_local = 0;
};

struct SampleRequestGroupResult {
    std::vector<Sample> train;
    std::vector<Sample> val;
    SampleStats stats;
    uint64_t processed = 0;
    uint64_t bytes_read = 0;
    bool ok = true;
};

std::vector<SampleRequestGroup> build_sample_request_groups(
    const std::vector<FileEntry> &entries,
    const std::vector<SampleRequest> &requests
) {
    std::vector<SampleRequestGroup> groups;
    size_t request_idx = 0;
    for (size_t entry_idx = 0; entry_idx < entries.size(); ++entry_idx) {
        const FileEntry &entry = entries[entry_idx];
        const uint64_t entry_end = entry.begin + entry.records;
        while (request_idx < requests.size() && requests[request_idx].global_index < entry.begin) {
            return {};
        }
        const size_t first_request = request_idx;
        while (request_idx < requests.size() && requests[request_idx].global_index < entry_end) {
            ++request_idx;
        }
        const size_t last_request = request_idx;
        if (first_request == last_request) {
            continue;
        }
        SampleRequestGroup group;
        group.entry_idx = entry_idx;
        group.first_request = first_request;
        group.last_request = last_request;
        group.first_local = requests[first_request].global_index - entry.begin;
        group.last_local = requests[last_request - 1].global_index - entry.begin;
        groups.emplace_back(group);
    }
    if (request_idx != requests.size()) {
        groups.clear();
    }
    return groups;
}

bool load_sample_request_group(
    const FileEntry &entry,
    const SampleRequestGroup &group,
    const std::vector<SampleRequest> &requests,
    std::vector<char> *file_span,
    SampleRequestGroupResult *result
) {
    const uint64_t first_offset = group.first_local * INDEXED_PHASE_RECORD_BYTES;
    const uint64_t span_records = group.last_local - group.first_local + 1;
    const uint64_t span_bytes = span_records * INDEXED_PHASE_RECORD_BYTES;
    if (!read_file_span(entry.path, first_offset, span_bytes, file_span)) {
        result->ok = false;
        return false;
    }
    result->bytes_read = span_bytes;
    result->train.reserve(group.last_request - group.first_request);
    result->val.reserve(group.last_request - group.first_request);
    for (size_t i = group.first_request; i < group.last_request; ++i) {
        const uint64_t local_index = requests[i].global_index - entry.begin;
        const uint64_t offset_records = local_index - group.first_local;
        const char *ptr = file_span->data() + (size_t)(offset_records * INDEXED_PHASE_RECORD_BYTES);
        IndexedDatum datum;
        decode_indexed_record(ptr, &datum);
        append_loaded_sample(datum, entry, requests[i].validation, &result->train, &result->val, &result->stats);
        ++result->processed;
    }
    return true;
}

bool load_samples_bulk(
    const std::vector<FileEntry> &entries,
    const std::vector<SampleRequest> &requests,
    std::vector<Sample> *train,
    std::vector<Sample> *val,
    SampleStats *stats,
    const Options &opt
) {
    train->clear();
    val->clear();
    const uint64_t start_ms = tim_ms();
    uint64_t next_log_ms = start_ms + (uint64_t)opt.progress_interval_sec * 1000ULL;
    const std::vector<SampleRequestGroup> groups = build_sample_request_groups(entries, requests);
    if (groups.empty() && !requests.empty()) {
        std::cerr << "[ERROR] failed to group sample requests by indexed data file\n";
        return false;
    }

    const int n_threads = std::max(1, std::min<int>(opt.read_threads, (int)groups.size()));
    std::vector<SampleRequestGroupResult> results(groups.size());
    std::atomic<size_t> next_group(0);
    std::atomic<uint64_t> processed_atomic(0);
    std::atomic<uint64_t> files_read_atomic(0);
    std::atomic<uint64_t> bytes_read_atomic(0);
    std::atomic<bool> failed(false);
    std::mutex log_mutex;

    auto worker = [&]() {
        std::vector<char> file_span;
        while (!failed.load(std::memory_order_relaxed)) {
            const size_t group_idx = next_group.fetch_add(1);
            if (group_idx >= groups.size()) {
                break;
            }
            const SampleRequestGroup &group = groups[group_idx];
            SampleRequestGroupResult &result = results[group_idx];
            const FileEntry &entry = entries[group.entry_idx];
            if (!load_sample_request_group(entry, group, requests, &file_span, &result)) {
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            const uint64_t processed_now = processed_atomic.fetch_add(result.processed) + result.processed;
            const uint64_t files_now = files_read_atomic.fetch_add(1) + 1;
            const uint64_t bytes_now = bytes_read_atomic.fetch_add(result.bytes_read) + result.bytes_read;
            if (opt.progress_interval_sec > 0) {
                std::lock_guard<std::mutex> lock(log_mutex);
                const uint64_t now = tim_ms();
                if (now >= next_log_ms) {
                    std::cerr << "sample_loading mode bulk"
                              << " processed " << processed_now
                              << " / " << requests.size()
                              << " files_read " << files_now
                              << " / " << groups.size()
                              << " read_gib " << bytes_to_gib((long double)bytes_now)
                              << " phase " << entry.phase
                              << " record " << entry.record
                              << " threads " << n_threads
                              << " elapsed_ms " << (now - start_ms) << "\n";
                    next_log_ms = now + (uint64_t)opt.progress_interval_sec * 1000ULL;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve((size_t)n_threads);
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (std::thread &thread: threads) {
        thread.join();
    }
    if (failed.load(std::memory_order_relaxed)) {
        std::cerr << "[ERROR] failed during bulk sample loading\n";
        return false;
    }

    for (SampleRequestGroupResult &result: results) {
        train->insert(train->end(), std::make_move_iterator(result.train.begin()), std::make_move_iterator(result.train.end()));
        val->insert(val->end(), std::make_move_iterator(result.val.begin()), std::make_move_iterator(result.val.end()));
        stats->bad_feature_count += result.stats.bad_feature_count;
        stats->phase_mismatch_count += result.stats.phase_mismatch_count;
        for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
            stats->train_phase_counts[(size_t)phase] += result.stats.train_phase_counts[(size_t)phase];
            stats->val_phase_counts[(size_t)phase] += result.stats.val_phase_counts[(size_t)phase];
        }
        std::vector<Sample>().swap(result.train);
        std::vector<Sample>().swap(result.val);
    }
    std::cerr << "sample_loading mode bulk"
              << " processed " << processed_atomic.load()
              << " / " << requests.size()
              << " train " << train->size()
              << " val " << val->size()
              << " files_read " << files_read_atomic.load()
              << " read_gib " << bytes_to_gib((long double)bytes_read_atomic.load())
              << " threads " << n_threads
              << " elapsed_ms " << (tim_ms() - start_ms) << "\n";
    return true;
}

bool read_indexed_seek(std::ifstream *in, const uint64_t local_index, IndexedDatum *datum) {
    const uint64_t offset = local_index * INDEXED_PHASE_RECORD_BYTES;
    in->clear();
    in->seekg((std::streamoff)offset, std::ios::beg);
    if (!(*in)) {
        return false;
    }
    in->read((char*)&datum->n_discs, sizeof(int16_t));
    in->read((char*)&datum->player, sizeof(int16_t));
    in->read((char*)datum->features, sizeof(uint16_t) * ADJ_N_FEATURES);
    in->read((char*)&datum->score, sizeof(int16_t));
    return (bool)(*in);
}

bool skip_bytes_scan(std::ifstream *in, uint64_t bytes) {
    constexpr uint64_t MAX_IGNORE_BYTES = 64ULL * 1024ULL * 1024ULL;
    while (bytes > 0) {
        const uint64_t step = std::min<uint64_t>(bytes, MAX_IGNORE_BYTES);
        in->ignore((std::streamsize)step);
        if (!(*in)) {
            return false;
        }
        bytes -= step;
    }
    return true;
}

bool read_indexed_scan(std::ifstream *in, const uint64_t local_index, uint64_t *next_local_index, IndexedDatum *datum) {
    if (local_index < *next_local_index) {
        return false;
    }
    const uint64_t skip_records = local_index - *next_local_index;
    if (skip_records != 0 && !skip_bytes_scan(in, skip_records * INDEXED_PHASE_RECORD_BYTES)) {
        return false;
    }
    in->read((char*)&datum->n_discs, sizeof(int16_t));
    in->read((char*)&datum->player, sizeof(int16_t));
    in->read((char*)datum->features, sizeof(uint16_t) * ADJ_N_FEATURES);
    in->read((char*)&datum->score, sizeof(int16_t));
    if (!(*in)) {
        return false;
    }
    *next_local_index = local_index + 1;
    return true;
}

bool load_samples(
    const std::vector<FileEntry> &entries,
    const std::vector<SampleRequest> &requests,
    std::vector<Sample> *train,
    std::vector<Sample> *val,
    SampleStats *stats,
    const Options &opt
) {
    if (opt.read_mode == "bulk") {
        return load_samples_bulk(entries, requests, train, val, stats, opt);
    }
    train->clear();
    val->clear();
    size_t entry_idx = 0;
    std::ifstream in;
    std::filesystem::path open_path;
    uint64_t next_local_index = 0;
    const uint64_t start_ms = tim_ms();
    uint64_t next_log_ms = start_ms + (uint64_t)opt.progress_interval_sec * 1000ULL;
    uint64_t processed = 0;

    for (const SampleRequest &request: requests) {
        while (entry_idx + 1 < entries.size() &&
               entries[entry_idx + 1].begin <= request.global_index) {
            ++entry_idx;
        }
        const FileEntry &entry = entries[entry_idx];
        if (!in.is_open() || open_path != entry.path) {
            if (in.is_open()) {
                in.close();
            }
            open_path = entry.path;
            in.open(open_path, std::ios::binary);
            if (!in) {
                std::cerr << "[ERROR] can't open indexed data " << open_path.string() << "\n";
                return false;
            }
            next_local_index = 0;
            const uint64_t now = tim_ms();
            if (opt.progress_interval_sec > 0 && now >= next_log_ms) {
                std::cerr << "sample_loading mode " << opt.read_mode
                          << " processed " << processed
                          << " / " << requests.size()
                          << " train " << train->size()
                          << " val " << val->size()
                          << " phase " << entry.phase
                          << " record " << entry.record
                          << " elapsed_ms " << (now - start_ms) << "\n";
                next_log_ms = now + (uint64_t)opt.progress_interval_sec * 1000ULL;
            }
        }
        const uint64_t local_index = request.global_index - entry.begin;
        IndexedDatum datum;
        const bool ok = opt.read_mode == "scan"
            ? read_indexed_scan(&in, local_index, &next_local_index, &datum)
            : read_indexed_seek(&in, local_index, &datum);
        if (!ok) {
            std::cerr << "[ERROR] can't read indexed record " << entry.path.string()
                      << " local_index " << local_index << "\n";
            return false;
        }
        ++processed;
        append_loaded_sample(datum, entry, request.validation, train, val, stats);
    }
    std::cerr << "sample_loading mode " << opt.read_mode
              << " processed " << processed
              << " / " << requests.size()
              << " train " << train->size()
              << " val " << val->size()
              << " elapsed_ms " << (tim_ms() - start_ms) << "\n";
    return true;
}

__global__ void accumulate_grad_kernel(
    const Sample *samples,
    const uint64_t n_samples,
    const float *linear,
    const float *fm,
    const int linear_per_phase,
    const int dim,
    const double grad_clip_raw,
    float *linear_grad,
    float *fm_grad
) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) {
        return;
    }
    const Sample &sample = samples[idx];
    float linear_pred = 0.0f;
    const uint64_t phase_offset = (uint64_t)sample.phase * (uint64_t)linear_per_phase;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        linear_pred += linear[phase_offset + (uint64_t)c_linear_starts[i] + sample.features[i]];
    }

    float sum[64];
    float square_sum[64];
    for (int d = 0; d < dim; ++d) {
        sum[d] = 0.0f;
        square_sum[d] = 0.0f;
    }
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        const uint64_t row = (uint64_t)(c_fm_offsets[i] + sample.features[i]) * (uint64_t)dim;
        for (int d = 0; d < dim; ++d) {
            const float x = fm[row + (uint64_t)d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }
    float fm_pred = 0.0f;
    for (int d = 0; d < dim; ++d) {
        fm_pred += 0.5f * (sum[d] * sum[d] - square_sum[d]);
    }

    double err_raw = (double)linear_pred + (double)fm_pred - (double)sample.score * (double)ADJ_STEP;
    if (grad_clip_raw > 0.0) {
        err_raw = fmin(fmax(err_raw, -grad_clip_raw), grad_clip_raw);
    }
    const float common = (float)(2.0 * err_raw / ((double)ADJ_STEP * (double)ADJ_STEP));

    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        const uint64_t row = phase_offset + (uint64_t)c_linear_starts[i] + sample.features[i];
        atomicAdd(&linear_grad[row], common);
    }
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        const uint64_t row = (uint64_t)(c_fm_offsets[i] + sample.features[i]) * (uint64_t)dim;
        for (int d = 0; d < dim; ++d) {
            const uint64_t fm_idx = row + (uint64_t)d;
            atomicAdd(&fm_grad[fm_idx], common * (sum[d] - fm[fm_idx]));
        }
    }
}

__global__ void adam_update_kernel(
    float *param,
    float *m,
    float *v,
    const float *grad_sum,
    const uint64_t n_param,
    const double inv_batch_n,
    const double lr,
    const double beta1,
    const double beta2,
    const double adam_eps,
    const double bias1,
    const double bias2,
    const double l2,
    const double clip_abs
) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_param) {
        return;
    }
    float g = (float)((double)grad_sum[idx] * inv_batch_n);
    if (l2 > 0.0) {
        g += (float)(l2 * (double)param[idx]);
    } else if (g == 0.0f) {
        return;
    }
    float mt = (float)(beta1 * (double)m[idx] + (1.0 - beta1) * (double)g);
    float vt = (float)(beta2 * (double)v[idx] + (1.0 - beta2) * (double)g * (double)g);
    m[idx] = mt;
    v[idx] = vt;
    double next = (double)param[idx] - lr * ((double)mt / bias1) / (sqrt((double)vt / bias2) + adam_eps);
    next = fmin(fmax(next, -clip_abs), clip_abs);
    param[idx] = (float)next;
}

__global__ void loss_kernel(
    const Sample *samples,
    const uint64_t n_samples,
    const float *linear,
    const float *fm,
    const int linear_per_phase,
    const int dim,
    double *loss_sum
) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) {
        return;
    }
    const Sample &sample = samples[idx];
    float linear_pred = 0.0f;
    const uint64_t phase_offset = (uint64_t)sample.phase * (uint64_t)linear_per_phase;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        linear_pred += linear[phase_offset + (uint64_t)c_linear_starts[i] + sample.features[i]];
    }
    float sum[64];
    float square_sum[64];
    for (int d = 0; d < dim; ++d) {
        sum[d] = 0.0f;
        square_sum[d] = 0.0f;
    }
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        const uint64_t row = (uint64_t)(c_fm_offsets[i] + sample.features[i]) * (uint64_t)dim;
        for (int d = 0; d < dim; ++d) {
            const float x = fm[row + (uint64_t)d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }
    float fm_pred = 0.0f;
    for (int d = 0; d < dim; ++d) {
        fm_pred += 0.5f * (sum[d] * sum[d] - square_sum[d]);
    }
    const double err_disc = ((double)linear_pred + (double)fm_pred - (double)sample.score * (double)ADJ_STEP) / (double)ADJ_STEP;
    atomicAdd(&loss_sum[0], err_disc * err_disc);
    atomicAdd(&loss_sum[1], fabs(err_disc));
}

bool cuda_alloc_zero(float **ptr, const uint64_t n) {
    CUDA_CHECK(cudaMalloc(ptr, sizeof(float) * (size_t)n));
    CUDA_CHECK(cudaMemset(*ptr, 0, sizeof(float) * (size_t)n));
    return true;
}

bool copy_vec_to_device(float **device, const std::vector<float> &host) {
    CUDA_CHECK(cudaMalloc(device, sizeof(float) * host.size()));
    CUDA_CHECK(cudaMemcpy(*device, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice));
    return true;
}

bool calc_loss_gpu(
    const std::vector<Sample> &samples,
    const uint64_t limit,
    Sample *d_batch,
    const uint64_t batch_capacity,
    const float *d_linear,
    const float *d_fm,
    const int linear_per_phase,
    const Options &opt,
    double *d_loss,
    Loss *out
) {
    const uint64_t n = limit == 0 ? (uint64_t)samples.size() : std::min<uint64_t>(limit, (uint64_t)samples.size());
    double total[2] = {0.0, 0.0};
    for (uint64_t begin = 0; begin < n; begin += batch_capacity) {
        const uint64_t chunk = std::min<uint64_t>(batch_capacity, n - begin);
        CUDA_CHECK(cudaMemcpy(d_batch, samples.data() + begin, sizeof(Sample) * (size_t)chunk, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double) * 2));
        const uint64_t blocks = (chunk + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        loss_kernel<<<(unsigned int)blocks, CUDA_BLOCK_SIZE>>>(
            d_batch, chunk, d_linear, d_fm, linear_per_phase, opt.dim, d_loss
        );
        CUDA_CHECK(cudaGetLastError());
        double partial[2] = {0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(partial, d_loss, sizeof(double) * 2, cudaMemcpyDeviceToHost));
        total[0] += partial[0];
        total[1] += partial[1];
    }
    out->n = n;
    out->mse = n == 0 ? 0.0 : total[0] / (double)n;
    out->mae = n == 0 ? 0.0 : total[1] / (double)n;
    return true;
}

bool train_epoch_gpu(
    std::vector<Sample> *train,
    Sample *d_batch,
    const uint64_t batch_capacity,
    float *d_linear,
    float *d_fm,
    float *d_linear_m,
    float *d_linear_v,
    float *d_fm_m,
    float *d_fm_v,
    float *d_linear_grad,
    float *d_fm_grad,
    const uint64_t linear_count,
    const uint64_t fm_count,
    const int linear_per_phase,
    const Options &opt,
    const int epoch,
    uint64_t *adam_step
) {
    const uint64_t start_ms = tim_ms();
    uint64_t next_log_ms = start_ms + (uint64_t)opt.progress_interval_sec * 1000ULL;
    deterministic_shuffle(train, opt.seed ^ (0xC001CAFEULL + (uint64_t)epoch));
    for (uint64_t begin = 0; begin < (uint64_t)train->size(); begin += batch_capacity) {
        const uint64_t chunk = std::min<uint64_t>(batch_capacity, (uint64_t)train->size() - begin);
        CUDA_CHECK(cudaMemcpy(d_batch, train->data() + begin, sizeof(Sample) * (size_t)chunk, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_linear_grad, 0, sizeof(float) * (size_t)linear_count));
        CUDA_CHECK(cudaMemset(d_fm_grad, 0, sizeof(float) * (size_t)fm_count));
        const uint64_t sample_blocks = (chunk + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        accumulate_grad_kernel<<<(unsigned int)sample_blocks, CUDA_BLOCK_SIZE>>>(
            d_batch, chunk, d_linear, d_fm, linear_per_phase, opt.dim, opt.grad_clip_raw,
            d_linear_grad, d_fm_grad
        );
        CUDA_CHECK(cudaGetLastError());
        ++(*adam_step);
        const double bias1 = 1.0 - std::pow(opt.beta1, (double)*adam_step);
        const double bias2 = 1.0 - std::pow(opt.beta2, (double)*adam_step);
        const double inv_batch_n = 1.0 / (double)chunk;
        const uint64_t linear_blocks = (linear_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        adam_update_kernel<<<(unsigned int)linear_blocks, CUDA_BLOCK_SIZE>>>(
            d_linear, d_linear_m, d_linear_v, d_linear_grad, linear_count, inv_batch_n,
            opt.linear_lr, opt.beta1, opt.beta2, opt.adam_eps, bias1, bias2,
            opt.linear_l2, opt.linear_param_clip
        );
        CUDA_CHECK(cudaGetLastError());
        const uint64_t fm_blocks = (fm_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        adam_update_kernel<<<(unsigned int)fm_blocks, CUDA_BLOCK_SIZE>>>(
            d_fm, d_fm_m, d_fm_v, d_fm_grad, fm_count, inv_batch_n,
            opt.fm_lr, opt.beta1, opt.beta2, opt.adam_eps, bias1, bias2,
            opt.fm_l2, opt.fm_vector_clip
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const uint64_t now = tim_ms();
        if (opt.progress_interval_sec > 0 && now >= next_log_ms) {
            std::cerr << "epoch_training_cuda processed " << (begin + chunk)
                      << " / " << train->size()
                      << " adam_step " << *adam_step
                      << " elapsed_ms " << (now - start_ms) << "\n";
            next_log_ms = now + (uint64_t)opt.progress_interval_sec * 1000ULL;
        }
    }
    return true;
}

std::vector<int16_t> quantize_linear(const std::vector<float> &linear, const double clip_abs) {
    std::vector<int16_t> out(linear.size());
    for (size_t i = 0; i < linear.size(); ++i) {
        int v = (int)std::lrint(linear[i]);
        v = std::clamp(v, (int)-clip_abs, (int)clip_abs);
        v = std::clamp(v, (int)std::numeric_limits<int16_t>::min(), (int)std::numeric_limits<int16_t>::max());
        out[i] = (int16_t)v;
    }
    return out;
}

bool write_fm_file(
    const std::string &out_file,
    const std::vector<float> &linear_float,
    const std::vector<float> &fm,
    const uint64_t total_vectors,
    const Options &opt,
    const uint64_t total_records,
    const std::array<uint64_t, ADJ_N_PHASES> &manifest_phase_counts,
    const SampleStats &sample_stats,
    const int best_epoch,
    const Loss &best_train_loss,
    const Loss &best_val_loss,
    const uint64_t stopped_epoch
) {
    std::filesystem::path out_path(out_file);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    const std::vector<int16_t> linear = quantize_linear(linear_float, opt.linear_param_clip);
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << "\n";
        return false;
    }
    const uint32_t version = FM_FILE_VERSION;
    const uint32_t n_phases = ADJ_N_PHASES;
    const uint32_t linear_per_phase = (uint32_t)linear_params_per_phase();
    const uint32_t fm_phases = 1;
    const uint32_t n_features = FM_N_PATTERN_FEATURES;
    const uint32_t dim = (uint32_t)opt.dim;
    const int32_t scale = opt.scale;
    const uint32_t flags = 0U;
    const uint64_t linear_count = (uint64_t)linear.size();
    const uint64_t fm_count = total_vectors * (uint64_t)opt.dim;

    out.write(FM_FILE_MAGIC, sizeof(FM_FILE_MAGIC));
    write_scalar(out, version);
    write_scalar(out, n_phases);
    write_scalar(out, linear_per_phase);
    write_scalar(out, fm_phases);
    write_scalar(out, n_features);
    write_scalar(out, dim);
    write_scalar(out, scale);
    write_scalar(out, flags);
    write_scalar(out, linear_count);
    write_scalar(out, fm_count);
    out.write((const char*)linear.data(), sizeof(int16_t) * linear.size());

    uint64_t nonzero = 0;
    int max_abs = 0;
    for (const float x: fm) {
        int q = (int)std::lrint((double)x * (double)opt.scale);
        q = std::clamp(q, -127, 127);
        nonzero += q != 0;
        max_abs = std::max(max_abs, std::abs(q));
        const int8_t q8 = (int8_t)q;
        out.write((const char*)&q8, sizeof(int8_t));
    }
    out.close();

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << std::setprecision(12);
        summary << "created_at_ms " << tim_ms() << "\n";
        summary << "definition " << EVAL_DEFINITION_NAME << "\n";
        summary << "learning_method joint_linear_fm_adam_sampled_mse_cuda\n";
        summary << "base_eval " << opt.base_eval << "\n";
        summary << "data_root " << opt.data_root << "\n";
        summary << "record_start " << opt.record_start << "\n";
        summary << "record_end " << opt.record_end << "\n";
        summary << "phase_start " << opt.phase_start << "\n";
        summary << "phase_end " << opt.phase_end << "\n";
        summary << "total_available_records " << total_records << "\n";
        summary << "train_samples_requested " << opt.train_samples << "\n";
        summary << "val_samples_requested " << opt.val_samples << "\n";
        summary << "train_samples_loaded " << std::accumulate(sample_stats.train_phase_counts.begin(), sample_stats.train_phase_counts.end(), 0ULL) << "\n";
        summary << "val_samples_loaded " << std::accumulate(sample_stats.val_phase_counts.begin(), sample_stats.val_phase_counts.end(), 0ULL) << "\n";
        summary << "bad_feature_count " << sample_stats.bad_feature_count << "\n";
        summary << "phase_mismatch_count " << sample_stats.phase_mismatch_count << "\n";
        summary << "linear_phases " << ADJ_N_PHASES << "\n";
        summary << "fm_phases 1\n";
        summary << "dim " << opt.dim << "\n";
        summary << "scale " << opt.scale << "\n";
        summary << "linear_params " << linear_count << "\n";
        summary << "total_vectors_per_fm_phase " << total_vectors << "\n";
        summary << "fm_values " << fm_count << "\n";
        summary << "fm_pattern_features all\n";
        summary << "epochs_requested " << opt.epochs << "\n";
        summary << "batch_size " << opt.batch_size << "\n";
        summary << "linear_lr " << opt.linear_lr << "\n";
        summary << "fm_lr " << opt.fm_lr << "\n";
        summary << "beta1 " << opt.beta1 << "\n";
        summary << "beta2 " << opt.beta2 << "\n";
        summary << "adam_eps " << opt.adam_eps << "\n";
        summary << "linear_l2 " << opt.linear_l2 << "\n";
        summary << "fm_l2 " << opt.fm_l2 << "\n";
        summary << "grad_clip_raw " << opt.grad_clip_raw << "\n";
        summary << "linear_param_clip " << opt.linear_param_clip << "\n";
        summary << "fm_vector_clip " << opt.fm_vector_clip << "\n";
        summary << "init_std " << opt.init_std << "\n";
        summary << "seed " << opt.seed << "\n";
        summary << "read_mode " << opt.read_mode << "\n";
        summary << "read_threads " << opt.read_threads << "\n";
        summary << "progress_interval_sec " << opt.progress_interval_sec << "\n";
        summary << "early_stop_patience " << opt.early_stop_patience << "\n";
        summary << "stopped_epoch " << stopped_epoch << "\n";
        summary << "best_epoch " << best_epoch << "\n";
        summary << "best_train_mse " << best_train_loss.mse << "\n";
        summary << "best_train_mae " << best_train_loss.mae << "\n";
        summary << "best_train_metric_samples " << best_train_loss.n << "\n";
        summary << "best_val_mse " << best_val_loss.mse << "\n";
        summary << "best_val_mae " << best_val_loss.mae << "\n";
        summary << "best_val_metric_samples " << best_val_loss.n << "\n";
        summary << "nonzero_quantized " << nonzero << "\n";
        summary << "max_abs_quantized " << max_abs << "\n";
        summary << "manifest_phase_counts";
        for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
            summary << " " << manifest_phase_counts[(size_t)phase];
        }
        summary << "\n";
        summary << "train_phase_counts";
        for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
            summary << " " << sample_stats.train_phase_counts[(size_t)phase];
        }
        summary << "\n";
        summary << "val_phase_counts";
        for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
            summary << " " << sample_stats.val_phase_counts[(size_t)phase];
        }
        summary << "\n";
    }
    std::cerr << "wrote " << out_file
              << " nonzero_quantized " << nonzero
              << " max_abs_quantized " << max_abs << "\n";
    return true;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt) || !validate_options(opt)) {
        usage();
        return 1;
    }

    const auto linear_starts = make_linear_starts();
    uint64_t total_vectors = 0;
    const auto fm_offsets = make_fm_feature_offsets(&total_vectors);
    const int linear_per_phase = linear_params_per_phase();
    const uint64_t linear_count = (uint64_t)ADJ_N_PHASES * (uint64_t)linear_per_phase;
    const uint64_t fm_count = total_vectors * (uint64_t)opt.dim;

    const std::vector<int16_t> base_linear_i16 = load_unzip_egev2_local(opt.base_eval);
    if ((uint64_t)base_linear_i16.size() != linear_count) {
        std::cerr << "[ERROR] invalid base eval element count " << base_linear_i16.size()
                  << " expected " << linear_count << "\n";
        return 1;
    }
    std::array<uint64_t, ADJ_N_PHASES> manifest_phase_counts = {};
    std::vector<FileEntry> manifest = build_manifest(opt, &manifest_phase_counts);
    const uint64_t total_records = manifest_total_count(manifest);
    if (manifest.empty() || total_records == 0) {
        std::cerr << "[ERROR] no indexed data found in " << opt.data_root << "\n";
        return 1;
    }
    if (opt.train_samples + opt.val_samples > total_records) {
        std::cerr << "[ERROR] requested unique samples exceed available records"
                  << " requested " << (opt.train_samples + opt.val_samples)
                  << " available " << total_records << "\n";
        return 1;
    }

    const HostMemoryEstimate host_memory = estimate_host_memory_bytes(opt, linear_count, fm_count, manifest);
    const double estimated_host_gib = bytes_to_gib((long double)host_memory.peak_bytes);
    const double estimated_device_gib = bytes_to_gib((long double)estimate_device_memory_bytes(opt, linear_count, fm_count));
    std::cerr << std::setprecision(6)
              << "manifest_files " << manifest.size()
              << " total_records " << total_records
              << " train_samples " << opt.train_samples
              << " val_samples " << opt.val_samples
              << " read_mode " << opt.read_mode
              << " linear_params " << linear_count
              << " fm_values " << fm_count
              << " estimated_host_memory_gib " << estimated_host_gib
              << " estimated_request_generation_gib " << bytes_to_gib((long double)host_memory.request_generation_bytes)
              << " estimated_sample_loading_gib " << bytes_to_gib((long double)host_memory.sample_loading_bytes)
              << " estimated_training_gib " << bytes_to_gib((long double)host_memory.training_bytes)
              << " estimated_device_memory_gib " << estimated_device_gib
              << " estimated_bulk_buffer_gib " << bytes_to_gib((long double)host_memory.bulk_buffer_bytes)
              << " max_memory_gib " << opt.max_memory_gib << "\n";
    if (estimated_host_gib > opt.max_memory_gib) {
        std::cerr << "[ERROR] estimated host memory exceeds limit\n";
        return 1;
    }
    if (opt.dry_run) {
        std::cerr << "dry_run complete\n";
        return 0;
    }

    cudaDeviceProp prop;
    CUDA_CHECK_MAIN(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "cuda_device " << prop.name
              << " global_mem_gib " << bytes_to_gib((long double)prop.totalGlobalMem)
              << " compute_capability " << prop.major << "." << prop.minor << "\n";

    const uint64_t load_start_ms = tim_ms();
    std::vector<SampleRequest> requests = generate_requests(
        total_records, opt.train_samples, opt.val_samples, opt.seed, opt.progress_interval_sec
    );
    std::vector<Sample> train_samples;
    std::vector<Sample> val_samples;
    train_samples.reserve((size_t)opt.train_samples);
    val_samples.reserve((size_t)opt.val_samples);
    SampleStats sample_stats;
    if (!load_samples(manifest, requests, &train_samples, &val_samples, &sample_stats, opt)) {
        return 1;
    }
    requests.clear();
    requests.shrink_to_fit();
    std::cerr << "loaded train " << train_samples.size()
              << " val " << val_samples.size()
              << " bad_features " << sample_stats.bad_feature_count
              << " phase_mismatches " << sample_stats.phase_mismatch_count
              << " elapsed_ms " << (tim_ms() - load_start_ms) << "\n";
    if (train_samples.empty() || val_samples.empty()) {
        std::cerr << "[ERROR] no usable train or validation samples\n";
        return 1;
    }

    CUDA_CHECK_MAIN(cudaMemcpyToSymbol(c_linear_starts, linear_starts.data(), sizeof(int) * ADJ_N_FEATURES));
    CUDA_CHECK_MAIN(cudaMemcpyToSymbol(c_fm_offsets, fm_offsets.data(), sizeof(int) * FM_N_PATTERN_FEATURES));

    std::vector<float> host_linear(base_linear_i16.begin(), base_linear_i16.end());
    std::vector<float> host_fm((size_t)fm_count, 0.0f);
    for (uint64_t i = 0; i < (uint64_t)host_fm.size(); ++i) {
        host_fm[(size_t)i] = deterministic_normal(opt.seed + 1, i, opt.init_std);
    }

    float *d_linear = nullptr, *d_fm = nullptr;
    float *d_linear_m = nullptr, *d_linear_v = nullptr, *d_fm_m = nullptr, *d_fm_v = nullptr;
    float *d_linear_grad = nullptr, *d_fm_grad = nullptr;
    float *d_best_linear = nullptr, *d_best_fm = nullptr;
    Sample *d_batch = nullptr;
    double *d_loss = nullptr;

    if (!copy_vec_to_device(&d_linear, host_linear) || !copy_vec_to_device(&d_fm, host_fm) ||
        !cuda_alloc_zero(&d_linear_m, linear_count) || !cuda_alloc_zero(&d_linear_v, linear_count) ||
        !cuda_alloc_zero(&d_fm_m, fm_count) || !cuda_alloc_zero(&d_fm_v, fm_count) ||
        !cuda_alloc_zero(&d_linear_grad, linear_count) || !cuda_alloc_zero(&d_fm_grad, fm_count) ||
        !cuda_alloc_zero(&d_best_linear, linear_count) || !cuda_alloc_zero(&d_best_fm, fm_count)) {
        return 1;
    }
    CUDA_CHECK_MAIN(cudaMemcpy(d_best_linear, d_linear, sizeof(float) * (size_t)linear_count, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_MAIN(cudaMemcpy(d_best_fm, d_fm, sizeof(float) * (size_t)fm_count, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_MAIN(cudaMalloc(&d_batch, sizeof(Sample) * (size_t)opt.batch_size));
    CUDA_CHECK_MAIN(cudaMalloc(&d_loss, sizeof(double) * 2));

    Loss best_train_loss;
    Loss best_val_loss;
    if (!calc_loss_gpu(train_samples, opt.train_metric_limit, d_batch, opt.batch_size, d_linear, d_fm, linear_per_phase, opt, d_loss, &best_train_loss) ||
        !calc_loss_gpu(val_samples, opt.val_metric_limit, d_batch, opt.batch_size, d_linear, d_fm, linear_per_phase, opt, d_loss, &best_val_loss)) {
        return 1;
    }
    int best_epoch = 0;
    int no_improve_epochs = 0;
    uint64_t adam_step = 0;
    uint64_t stopped_epoch = 0;
    std::cerr << "initial train_mse " << best_train_loss.mse
              << " train_mae " << best_train_loss.mae
              << " train_metric_samples " << best_train_loss.n
              << " val_mse " << best_val_loss.mse
              << " val_mae " << best_val_loss.mae
              << " val_metric_samples " << best_val_loss.n << "\n";

    for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
        const uint64_t epoch_start_ms = tim_ms();
        if (!train_epoch_gpu(
                &train_samples, d_batch, opt.batch_size, d_linear, d_fm,
                d_linear_m, d_linear_v, d_fm_m, d_fm_v,
                d_linear_grad, d_fm_grad, linear_count, fm_count,
                linear_per_phase, opt, epoch, &adam_step)) {
            return 1;
        }
        Loss train_loss;
        Loss val_loss;
        if (!calc_loss_gpu(train_samples, opt.train_metric_limit, d_batch, opt.batch_size, d_linear, d_fm, linear_per_phase, opt, d_loss, &train_loss) ||
            !calc_loss_gpu(val_samples, opt.val_metric_limit, d_batch, opt.batch_size, d_linear, d_fm, linear_per_phase, opt, d_loss, &val_loss)) {
            return 1;
        }
        if (val_loss.mse < best_val_loss.mse) {
            best_val_loss = val_loss;
            best_train_loss = train_loss;
            best_epoch = epoch;
            no_improve_epochs = 0;
            CUDA_CHECK_MAIN(cudaMemcpy(d_best_linear, d_linear, sizeof(float) * (size_t)linear_count, cudaMemcpyDeviceToDevice));
            CUDA_CHECK_MAIN(cudaMemcpy(d_best_fm, d_fm, sizeof(float) * (size_t)fm_count, cudaMemcpyDeviceToDevice));
        } else {
            ++no_improve_epochs;
        }
        stopped_epoch = (uint64_t)epoch;
        std::cerr << "epoch " << epoch
                  << " elapsed_ms " << (tim_ms() - epoch_start_ms)
                  << " train_mse " << train_loss.mse
                  << " train_mae " << train_loss.mae
                  << " val_mse " << val_loss.mse
                  << " val_mae " << val_loss.mae
                  << " best_epoch " << best_epoch
                  << " best_val_mse " << best_val_loss.mse
                  << " no_improve_epochs " << no_improve_epochs << "\n";
        if (opt.early_stop_patience > 0 && no_improve_epochs >= opt.early_stop_patience) {
            std::cerr << "early_stop epoch " << epoch
                      << " best_epoch " << best_epoch
                      << " best_val_mse " << best_val_loss.mse << "\n";
            break;
        }
    }

    host_linear.resize((size_t)linear_count);
    host_fm.resize((size_t)fm_count);
    CUDA_CHECK_MAIN(cudaMemcpy(host_linear.data(), d_best_linear, sizeof(float) * (size_t)linear_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK_MAIN(cudaMemcpy(host_fm.data(), d_best_fm, sizeof(float) * (size_t)fm_count, cudaMemcpyDeviceToHost));

    const bool write_ok = write_fm_file(
        opt.out_file, host_linear, host_fm, total_vectors, opt, total_records,
        manifest_phase_counts, sample_stats, best_epoch, best_train_loss, best_val_loss, stopped_epoch
    );

    cudaFree(d_linear);
    cudaFree(d_fm);
    cudaFree(d_linear_m);
    cudaFree(d_linear_v);
    cudaFree(d_fm_m);
    cudaFree(d_fm_v);
    cudaFree(d_linear_grad);
    cudaFree(d_fm_grad);
    cudaFree(d_best_linear);
    cudaFree(d_best_fm);
    cudaFree(d_batch);
    cudaFree(d_loss);
    return write_ok ? 0 : 1;
}
