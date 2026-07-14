/*
    Egaroucid evaluation experiment

    Fine-tune both linear weights and FM vectors of an existing 7.7-beta + FM
    EGEV4 file. This keeps the experiment separate from the FM-only and
    linear-only optimizers and preserves the original evaluation files.
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int N_PHASES = 60;
constexpr int N_PATTERN_FEATURES = 64;
constexpr int N_FEATURES = N_PATTERN_FEATURES + 1;
constexpr int N_PATTERN_PARAMS_RAW = 944784;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + 65;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;
constexpr int MAX_DIM = 16;
constexpr int RECORD_BYTES = 136;

const std::array<int, 16> PATTERN_OFFSETS = {
    0, 59049, 118098, 177147,
    236196, 295245, 354294, 413343,
    472392, 531441, 590490, 649539,
    708588, 767637, 826686, 885735
};

struct Config {
    std::string input_egev4;
    std::string data_root;
    std::string output_egev4;
    std::string summary;
    std::string timestamp;
    std::vector<int> phases;
    std::vector<int> train_ids;
    std::vector<int> holdout_ids;
    int epochs = 12;
    int max_records_per_file = 0;
    int holdout_max_records_per_file = -1;
    int train_metric_max_records_per_file = -1;
    int chunk_records = 65536;
    int early_stop_patience = 2;
    double linear_lr = 0.00002;
    double vector_lr = 0.000002;
    double residual_clip = 16.0;
    std::string loss = "pseudo-huber";
    double huber_delta = 4.0;
    double linear_l2 = 0.00001;
    double vector_l2 = 0.00001;
    double weight_decay = 0.0;
    uint64_t seed = 20260714;
    bool shuffle_files = true;
};

struct Egev4Model {
    int dim = 0;
    std::array<float, N_PHASES> linear_scale{};
    std::array<float, N_PHASES> vector_scale{};
    std::vector<int16_t> linear_quant;
    std::vector<float> linear_value;
    std::array<std::vector<int8_t>, N_PHASES> vector_quant;
};

struct Record {
    std::array<uint32_t, N_FEATURES> active_ids{};
    float score = 0.0f;
};

struct Metrics {
    uint64_t n = 0;
    double mse = 0.0;
    double mae = 0.0;
    double max_abs_error = 0.0;
};

struct EpochRow {
    int epoch = 0;
    uint64_t n_initialized = 0;
    Metrics train;
    Metrics holdout;
    uint64_t elapsed_ms = 0;
};

struct PhaseResult {
    int phase = 0;
    uint64_t n_train = 0;
    uint64_t n_holdout = 0;
    int best_epoch = 0;
    double max_abs_delta = 0.0;
    double vector_scale = 1.0;
    double vector_max_abs = 0.0;
    double max_abs_vector_delta = 0.0;
    double mean_abs_delta_initialized = 0.0;
    uint64_t n_initialized_best = 0;
    uint64_t changed_linear_values = 0;
    uint64_t changed_vector_values = 0;
    uint64_t clamped_linear_values = 0;
    uint64_t clamped_vector_values = 0;
    std::vector<EpochRow> history;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message + "\n"
        "usage: eval_optimizer_7_7_fm_joint_finetune_experiment "
        "--input-egev4 FILE --data-root DIR --phases 6-13 "
        "--train-ids 0,1,2 --holdout-ids 3 "
        "--output-egev4 FILE --summary FILE [options]");
}

std::string require_value(int &idx, int argc, char **argv) {
    if (idx + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[idx]);
    }
    ++idx;
    return argv[idx];
}

std::vector<int> parse_int_list(const std::string &text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string elem;
    while (std::getline(ss, elem, ',')) {
        if (elem.empty()) {
            continue;
        }
        const size_t dash = elem.find('-');
        if (dash == std::string::npos) {
            values.push_back(std::stoi(elem));
        } else {
            const int start = std::stoi(elem.substr(0, dash));
            const int end = std::stoi(elem.substr(dash + 1));
            if (end < start) {
                usage_error("invalid range: " + elem);
            }
            for (int value = start; value <= end; ++value) {
                values.push_back(value);
            }
        }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::string join_ints(const std::vector<int> &values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    return out.str();
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-egev4") {
            cfg.input_egev4 = require_value(i, argc, argv);
        } else if (arg == "--data-root") {
            cfg.data_root = require_value(i, argc, argv);
        } else if (arg == "--phases") {
            cfg.phases = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--train-ids") {
            cfg.train_ids = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--holdout-ids") {
            cfg.holdout_ids = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--output-egev4") {
            cfg.output_egev4 = require_value(i, argc, argv);
        } else if (arg == "--summary") {
            cfg.summary = require_value(i, argc, argv);
        } else if (arg == "--epochs") {
            cfg.epochs = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--lr" || arg == "--linear-lr") {
            cfg.linear_lr = std::stod(require_value(i, argc, argv));
        } else if (arg == "--vector-lr") {
            cfg.vector_lr = std::stod(require_value(i, argc, argv));
        } else if (arg == "--residual-clip") {
            cfg.residual_clip = std::stod(require_value(i, argc, argv));
        } else if (arg == "--loss") {
            cfg.loss = require_value(i, argc, argv);
        } else if (arg == "--huber-delta") {
            cfg.huber_delta = std::stod(require_value(i, argc, argv));
        } else if (arg == "--linear-l2") {
            cfg.linear_l2 = std::stod(require_value(i, argc, argv));
        } else if (arg == "--vector-l2") {
            cfg.vector_l2 = std::stod(require_value(i, argc, argv));
        } else if (arg == "--weight-decay") {
            cfg.weight_decay = std::stod(require_value(i, argc, argv));
        } else if (arg == "--max-records-per-file") {
            cfg.max_records_per_file = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--holdout-max-records-per-file") {
            cfg.holdout_max_records_per_file = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--train-metric-max-records-per-file") {
            cfg.train_metric_max_records_per_file = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--chunk-records") {
            cfg.chunk_records = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--early-stop-patience") {
            cfg.early_stop_patience = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--seed") {
            cfg.seed = std::stoull(require_value(i, argc, argv));
        } else if (arg == "--timestamp") {
            cfg.timestamp = require_value(i, argc, argv);
        } else if (arg == "--no-shuffle-files") {
            cfg.shuffle_files = false;
        } else {
            usage_error("unknown argument: " + arg);
        }
    }
    if (cfg.input_egev4.empty() || cfg.data_root.empty() || cfg.output_egev4.empty() || cfg.summary.empty()) {
        usage_error("--input-egev4, --data-root, --output-egev4, and --summary are required");
    }
    if (cfg.phases.empty() || cfg.train_ids.empty() || cfg.holdout_ids.empty()) {
        usage_error("--phases, --train-ids, and --holdout-ids are required");
    }
    if (cfg.epochs <= 0 || cfg.chunk_records <= 0) {
        usage_error("--epochs and --chunk-records must be positive");
    }
    if (cfg.loss != "clipped-residual" && cfg.loss != "pseudo-huber") {
        usage_error("--loss must be clipped-residual or pseudo-huber");
    }
    if (cfg.holdout_max_records_per_file < 0) {
        cfg.holdout_max_records_per_file = cfg.max_records_per_file;
    }
    if (cfg.train_metric_max_records_per_file < 0) {
        cfg.train_metric_max_records_per_file = cfg.max_records_per_file;
    }
    for (int phase : cfg.phases) {
        if (phase < 0 || phase >= N_PHASES) {
            usage_error("phase out of range: " + std::to_string(phase));
        }
    }
    if (!cfg.timestamp.empty() && (cfg.timestamp.size() != 14 || !std::all_of(cfg.timestamp.begin(), cfg.timestamp.end(), ::isdigit))) {
        usage_error("--timestamp must be 14 digits");
    }
    return cfg;
}

uint64_t now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

template <typename T>
T clamp_value(T value, T lo, T hi) {
    return std::max(lo, std::min(hi, value));
}

double residual_for_loss(double residual, const Config &cfg) {
    if (cfg.loss == "pseudo-huber") {
        const double ratio = residual / cfg.huber_delta;
        return residual / std::sqrt(1.0 + ratio * ratio);
    }
    if (cfg.residual_clip > 0.0) {
        return clamp_value(residual, -cfg.residual_clip, cfg.residual_clip);
    }
    return residual;
}

int32_t read_i32(const unsigned char *p) {
    int32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

Egev4Model load_egev4_model(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    unsigned char header[34]{};
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!in || std::memcmp(header + 14, "EGEV", 4) != 0) {
        throw std::runtime_error("broken EGEV4 header: " + path.string());
    }
    const int version = read_i32(header + 18);
    const int n_phase = read_i32(header + 22);
    const int n_param = read_i32(header + 26);
    const int dim = read_i32(header + 30);
    if (version != VERSION_LINEAR_FM_INT16_INT8 || n_phase != N_PHASES ||
        n_param != N_PARAMS_PER_PHASE || dim <= 0 || dim > MAX_DIM) {
        throw std::runtime_error("EGEV4 header mismatch: " + path.string());
    }

    Egev4Model model;
    model.dim = dim;
    in.read(reinterpret_cast<char *>(model.linear_scale.data()), sizeof(float) * N_PHASES);
    in.read(reinterpret_cast<char *>(model.vector_scale.data()), sizeof(float) * N_PHASES);
    if (!in) {
        throw std::runtime_error("broken EGEV4 scales: " + path.string());
    }
    model.linear_quant.assign(static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE, 0);
    model.linear_value.assign(static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE, 0.0f);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        model.vector_quant[phase].assign(static_cast<size_t>(N_PARAMS_PER_PHASE) * dim, 0);
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            int16_t q_linear = 0;
            in.read(reinterpret_cast<char *>(&q_linear), sizeof(q_linear));
            const size_t vector_idx = static_cast<size_t>(param_id) * dim;
            in.read(reinterpret_cast<char *>(&model.vector_quant[phase][vector_idx]), dim);
            if (!in) {
                throw std::runtime_error("broken EGEV4 payload: " + path.string());
            }
            const size_t linear_idx = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE + param_id;
            model.linear_quant[linear_idx] = q_linear;
            model.linear_value[linear_idx] = static_cast<float>(static_cast<double>(q_linear) * model.linear_scale[phase]);
        }
    }
    return model;
}

fs::path data_path(const fs::path &root, int phase, int file_id) {
    const fs::path phase_dir = root / std::to_string(phase);
    const fs::path plain = phase_dir / (std::to_string(file_id) + ".dat");
    if (fs::exists(plain)) {
        return plain;
    }
    const fs::path records = phase_dir / ("records" + std::to_string(file_id) + ".dat");
    if (fs::exists(records)) {
        return records;
    }
    const fs::path batch = phase_dir / ("batch" + std::to_string(file_id) + ".dat");
    if (fs::exists(batch)) {
        return batch;
    }
    return plain;
}

uint64_t count_file_records(const fs::path &path, int max_records_per_file) {
    const uint64_t file_records = static_cast<uint64_t>(fs::file_size(path) / RECORD_BYTES);
    return max_records_per_file > 0 ? std::min<uint64_t>(file_records, static_cast<uint64_t>(max_records_per_file)) : file_records;
}

uint64_t count_records(const fs::path &root, int phase, const std::vector<int> &file_ids, int max_records_per_file) {
    uint64_t total = 0;
    for (int file_id : file_ids) {
        const fs::path path = data_path(root, phase, file_id);
        if (!fs::exists(path)) {
            throw std::runtime_error("missing data file: " + path.string());
        }
        total += count_file_records(path, max_records_per_file);
    }
    return total;
}

bool read_record(std::ifstream &in, Record &record) {
    std::array<char, RECORD_BYTES> raw{};
    in.read(raw.data(), raw.size());
    if (!in) {
        if (in.eof() && in.gcount() == 0) {
            return false;
        }
        throw std::runtime_error("partial record encountered");
    }
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        uint16_t feature = 0;
        std::memcpy(&feature, raw.data() + 4 + i * 2, sizeof(feature));
        record.active_ids[i] = static_cast<uint32_t>(PATTERN_OFFSETS[i / 4] + static_cast<int>(feature));
    }
    uint16_t stone_count_feature = 0;
    std::memcpy(&stone_count_feature, raw.data() + 4 + N_PATTERN_FEATURES * 2, sizeof(stone_count_feature));
    record.active_ids[N_PATTERN_FEATURES] = static_cast<uint32_t>(N_PATTERN_PARAMS_RAW + static_cast<int>(stone_count_feature));
    int16_t score = 0;
    std::memcpy(&score, raw.data() + 4 + N_FEATURES * 2, sizeof(score));
    record.score = static_cast<float>(score);
    return true;
}

double predict_record(
    const float *linear,
    const float *vectors,
    const Record &record,
    int dim,
    std::array<double, MAX_DIM> &sums,
    std::array<double, MAX_DIM> &sums_sq
) {
    sums.fill(0.0);
    sums_sq.fill(0.0);
    double value = 0.0;
    for (uint32_t param_id : record.active_ids) {
        value += linear[param_id];
        const size_t base = static_cast<size_t>(param_id) * dim;
        for (int d = 0; d < dim; ++d) {
            const double vec = static_cast<double>(vectors[base + d]);
            sums[d] += vec;
            sums_sq[d] += vec * vec;
        }
    }
    double interaction = 0.0;
    for (int d = 0; d < dim; ++d) {
        interaction += sums[d] * sums[d] - sums_sq[d];
    }
    return value + 0.5 * interaction;
}

Metrics compute_metrics(
    const fs::path &root,
    int phase,
    const std::vector<int> &file_ids,
    int max_records_per_file,
    const float *linear,
    const float *vectors,
    int dim
) {
    Metrics metrics;
    std::array<double, MAX_DIM> sums{};
    std::array<double, MAX_DIM> sums_sq{};
    for (int file_id : file_ids) {
        const fs::path path = data_path(root, phase, file_id);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open " + path.string());
        }
        uint64_t n_file = 0;
        while (max_records_per_file <= 0 || n_file < static_cast<uint64_t>(max_records_per_file)) {
            Record record;
            if (!read_record(in, record)) {
                break;
            }
            const double pred = predict_record(linear, vectors, record, dim, sums, sums_sq);
            const double err = static_cast<double>(record.score) - pred;
            const double abs_err = std::abs(err);
            metrics.mse += err * err;
            metrics.mae += abs_err;
            metrics.max_abs_error = std::max(metrics.max_abs_error, abs_err);
            ++metrics.n;
            ++n_file;
        }
    }
    if (metrics.n > 0) {
        metrics.mse /= static_cast<double>(metrics.n);
        metrics.mae /= static_cast<double>(metrics.n);
    }
    return metrics;
}

uint64_t count_initialized(const std::vector<uint8_t> &initialized) {
    uint64_t n = 0;
    for (uint8_t value : initialized) {
        n += value != 0;
    }
    return n;
}

void update_record(
    const Record &record,
    const Config &cfg,
    const float *base_linear,
    std::vector<float> &linear,
    const float *base_vectors,
    std::vector<float> &vectors,
    int dim,
    std::vector<float> &linear_m,
    std::vector<float> &linear_v,
    std::vector<float> &vector_m,
    std::vector<float> &vector_v,
    std::vector<uint8_t> &initialized,
    uint64_t &step,
    double &beta1_power,
    double &beta2_power,
    std::array<double, MAX_DIM> &sums,
    std::array<double, MAX_DIM> &sums_sq
) {
    const double pred = predict_record(linear.data(), vectors.data(), record, dim, sums, sums_sq);
    const double residual = residual_for_loss(static_cast<double>(record.score) - pred, cfg);

    ++step;
    beta1_power *= 0.9;
    beta2_power *= 0.999;
    const double adam_correction = std::sqrt(1.0 - beta2_power) / (1.0 - beta1_power);
    const double linear_lr_t = cfg.linear_lr * adam_correction;
    const double vector_lr_t = cfg.vector_lr * adam_correction;
    const double linear_decay = cfg.weight_decay > 0.0 ? std::max(0.0, 1.0 - cfg.linear_lr * cfg.weight_decay) : 1.0;
    const double vector_decay = cfg.weight_decay > 0.0 ? std::max(0.0, 1.0 - cfg.vector_lr * cfg.weight_decay) : 1.0;

    for (uint32_t param_id : record.active_ids) {
        initialized[param_id] = 1;
        const double value = linear[param_id];
        const double delta = value - static_cast<double>(base_linear[param_id]);
        const double grad = -residual + cfg.linear_l2 * delta;
        const double next_m = 0.9 * linear_m[param_id] + 0.1 * grad;
        const double next_v = 0.999 * linear_v[param_id] + 0.001 * grad * grad;
        double next_value = value - linear_lr_t * next_m / (std::sqrt(next_v) + 1.0e-8);
        if (linear_decay != 1.0) {
            next_value = static_cast<double>(base_linear[param_id]) + (next_value - static_cast<double>(base_linear[param_id])) * linear_decay;
        }
        linear_m[param_id] = static_cast<float>(next_m);
        linear_v[param_id] = static_cast<float>(next_v);
        linear[param_id] = static_cast<float>(next_value);

        const size_t base = static_cast<size_t>(param_id) * dim;
        for (int d = 0; d < dim; ++d) {
            const size_t idx = base + static_cast<size_t>(d);
            const double vec = vectors[idx];
            const double vec_delta = vec - static_cast<double>(base_vectors[idx]);
            const double vec_grad = -residual * (sums[d] - vec) + cfg.vector_l2 * vec_delta;
            const double next_vec_m = 0.9 * vector_m[idx] + 0.1 * vec_grad;
            const double next_vec_v = 0.999 * vector_v[idx] + 0.001 * vec_grad * vec_grad;
            double next_vec = vec - vector_lr_t * next_vec_m / (std::sqrt(next_vec_v) + 1.0e-8);
            if (vector_decay != 1.0) {
                next_vec = static_cast<double>(base_vectors[idx]) + (next_vec - static_cast<double>(base_vectors[idx])) * vector_decay;
            }
            vector_m[idx] = static_cast<float>(next_vec_m);
            vector_v[idx] = static_cast<float>(next_vec_v);
            vectors[idx] = static_cast<float>(next_vec);
        }
    }
}

uint64_t train_one_epoch(
    const fs::path &root,
    int phase,
    const std::vector<int> &train_ids,
    int epoch,
    const Config &cfg,
    const float *base_linear,
    std::vector<float> &linear,
    const float *base_vectors,
    std::vector<float> &vectors,
    int dim,
    std::vector<float> &linear_m,
    std::vector<float> &linear_v,
    std::vector<float> &vector_m,
    std::vector<float> &vector_v,
    std::vector<uint8_t> &initialized,
    uint64_t &step,
    double &beta1_power,
    double &beta2_power
) {
    std::vector<int> file_order = train_ids;
    if (cfg.shuffle_files) {
        std::mt19937_64 file_rng(cfg.seed ^ (static_cast<uint64_t>(phase + 1) << 32) ^ static_cast<uint64_t>(epoch));
        std::shuffle(file_order.begin(), file_order.end(), file_rng);
    }
    std::vector<Record> chunk;
    chunk.reserve(static_cast<size_t>(cfg.chunk_records));
    std::array<double, MAX_DIM> sums{};
    std::array<double, MAX_DIM> sums_sq{};
    uint64_t updates = 0;

    for (int file_id : file_order) {
        const fs::path path = data_path(root, phase, file_id);
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open " + path.string());
        }
        uint64_t n_file = 0;
        uint64_t chunk_idx = 0;
        while (cfg.max_records_per_file <= 0 || n_file < static_cast<uint64_t>(cfg.max_records_per_file)) {
            chunk.clear();
            while (static_cast<int>(chunk.size()) < cfg.chunk_records &&
                   (cfg.max_records_per_file <= 0 || n_file < static_cast<uint64_t>(cfg.max_records_per_file))) {
                Record record;
                if (!read_record(in, record)) {
                    break;
                }
                chunk.push_back(record);
                ++n_file;
            }
            if (chunk.empty()) {
                break;
            }
            std::mt19937_64 chunk_rng(
                cfg.seed ^
                (static_cast<uint64_t>(phase + 1) << 48) ^
                (static_cast<uint64_t>(epoch + 1) << 32) ^
                (static_cast<uint64_t>(file_id + 1) << 16) ^
                chunk_idx
            );
            std::shuffle(chunk.begin(), chunk.end(), chunk_rng);
            for (const Record &record : chunk) {
                update_record(record, cfg, base_linear, linear, base_vectors, vectors, dim,
                    linear_m, linear_v, vector_m, vector_v, initialized, step, beta1_power, beta2_power, sums, sums_sq);
                ++updates;
            }
            ++chunk_idx;
        }
    }
    return updates;
}

double vector_scale_for(const std::vector<float> &vectors, double &max_abs) {
    max_abs = 0.0;
    for (float value : vectors) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
    }
    return max_abs > 0.0 ? max_abs / 127.0 : 1.0;
}

void fill_phase_stats(
    PhaseResult &result,
    const float *base_linear,
    const std::vector<float> &best_linear,
    const std::vector<float> &base_vectors,
    const std::vector<float> &best_vectors,
    const std::vector<uint8_t> &best_initialized,
    const int16_t *base_quant,
    const int8_t *base_vector_quant,
    int dim,
    float linear_scale
) {
    double sum_abs_delta_initialized = 0.0;
    for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
        const double delta = static_cast<double>(best_linear[param_id]) - static_cast<double>(base_linear[param_id]);
        const double abs_delta = std::abs(delta);
        result.max_abs_delta = std::max(result.max_abs_delta, abs_delta);
        if (best_initialized[param_id] != 0) {
            sum_abs_delta_initialized += abs_delta;
            ++result.n_initialized_best;
        }
        const int raw_q = static_cast<int>(std::llround(static_cast<double>(best_linear[param_id]) / static_cast<double>(linear_scale)));
        const int q = clamp_value(raw_q, -32768, 32767);
        result.clamped_linear_values += raw_q != q;
        result.changed_linear_values += static_cast<int16_t>(q) != base_quant[param_id];
    }
    if (result.n_initialized_best > 0) {
        result.mean_abs_delta_initialized = sum_abs_delta_initialized / static_cast<double>(result.n_initialized_best);
    }
    result.vector_scale = vector_scale_for(best_vectors, result.vector_max_abs);
    for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
        const size_t base = static_cast<size_t>(param_id) * dim;
        for (int d = 0; d < dim; ++d) {
            const size_t idx = base + static_cast<size_t>(d);
            const double delta = static_cast<double>(best_vectors[idx]) - static_cast<double>(base_vectors[idx]);
            result.max_abs_vector_delta = std::max(result.max_abs_vector_delta, std::abs(delta));
            const int raw_q = static_cast<int>(std::llround(static_cast<double>(best_vectors[idx]) / result.vector_scale));
            const int q = clamp_value(raw_q, -127, 127);
            result.clamped_vector_values += raw_q != q;
            result.changed_vector_values += static_cast<int8_t>(q) != base_vector_quant[idx];
        }
    }
}

PhaseResult train_phase(
    const Config &cfg,
    const fs::path &data_root,
    int phase,
    const Egev4Model &model,
    std::vector<float> &best_linear_out,
    std::vector<float> &best_vectors_out
) {
    const uint64_t start_ms = now_ms();
    const size_t phase_base = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
    const float *base_linear = model.linear_value.data() + phase_base;
    const int16_t *base_quant = model.linear_quant.data() + phase_base;
    const int8_t *base_vector_quant = model.vector_quant[phase].data();
    const float vector_scale = model.vector_scale[phase];
    std::vector<float> linear(base_linear, base_linear + N_PARAMS_PER_PHASE);
    std::vector<float> base_vectors(static_cast<size_t>(N_PARAMS_PER_PHASE) * model.dim, 0.0f);
    for (size_t i = 0; i < base_vectors.size(); ++i) {
        base_vectors[i] = static_cast<float>(static_cast<double>(base_vector_quant[i]) * static_cast<double>(vector_scale));
    }
    std::vector<float> vectors = base_vectors;
    std::vector<float> linear_m(static_cast<size_t>(N_PARAMS_PER_PHASE), 0.0f);
    std::vector<float> linear_v(static_cast<size_t>(N_PARAMS_PER_PHASE), 0.0f);
    std::vector<float> vector_m(static_cast<size_t>(N_PARAMS_PER_PHASE) * model.dim, 0.0f);
    std::vector<float> vector_v(static_cast<size_t>(N_PARAMS_PER_PHASE) * model.dim, 0.0f);
    std::vector<uint8_t> initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);
    std::vector<uint8_t> best_initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);

    PhaseResult result;
    result.phase = phase;
    result.n_train = count_records(data_root, phase, cfg.train_ids, cfg.max_records_per_file);
    result.n_holdout = count_records(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file);

    uint64_t step = 0;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    EpochRow initial;
    initial.epoch = 0;
    initial.train = compute_metrics(data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file,
        linear.data(), vectors.data(), model.dim);
    initial.holdout = compute_metrics(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file,
        linear.data(), vectors.data(), model.dim);
    initial.elapsed_ms = now_ms() - start_ms;
    result.history.push_back(initial);

    double best_holdout_mae = initial.holdout.mae;
    best_linear_out = linear;
    best_vectors_out = vectors;
    result.best_epoch = 0;

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const uint64_t updates = train_one_epoch(
            data_root, phase, cfg.train_ids, epoch, cfg, base_linear, linear, base_vectors.data(), vectors, model.dim,
            linear_m, linear_v, vector_m, vector_v, initialized, step, beta1_power, beta2_power
        );
        if (updates == 0) {
            throw std::runtime_error("no records trained for phase " + std::to_string(phase));
        }
        EpochRow row;
        row.epoch = epoch;
        row.n_initialized = count_initialized(initialized);
        row.train = compute_metrics(data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file,
            linear.data(), vectors.data(), model.dim);
        row.holdout = compute_metrics(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file,
            linear.data(), vectors.data(), model.dim);
        row.elapsed_ms = now_ms() - start_ms;
        result.history.push_back(row);

        if (row.holdout.mae < best_holdout_mae) {
            best_holdout_mae = row.holdout.mae;
            result.best_epoch = epoch;
            best_linear_out = linear;
            best_vectors_out = vectors;
            best_initialized = initialized;
        } else if (cfg.early_stop_patience > 0 && epoch - result.best_epoch >= cfg.early_stop_patience) {
            break;
        }
    }

    fill_phase_stats(result, base_linear, best_linear_out, base_vectors, best_vectors_out,
        best_initialized, base_quant, base_vector_quant, model.dim, model.linear_scale[phase]);
    return result;
}

std::string make_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tm, &tt);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d%H%M%S");
    return out.str();
}

template <typename T>
void write_value(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_egev4(
    const Config &cfg,
    const Egev4Model &model,
    const std::map<int, std::vector<float>> &phase_linears,
    const std::map<int, std::vector<float>> &phase_vectors,
    const std::map<int, PhaseResult> &phase_results
) {
    const fs::path output_path(cfg.output_egev4);
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write " + output_path.string());
    }
    const std::string timestamp = cfg.timestamp.empty() ? make_timestamp() : cfg.timestamp;
    out.write(timestamp.data(), 14);
    out.write("EGEV", 4);
    write_value<int32_t>(out, VERSION_LINEAR_FM_INT16_INT8);
    write_value<int32_t>(out, N_PHASES);
    write_value<int32_t>(out, N_PARAMS_PER_PHASE);
    write_value<int32_t>(out, model.dim);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        write_value<float>(out, model.linear_scale[phase]);
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const auto result_it = phase_results.find(phase);
        write_value<float>(out, result_it == phase_results.end() ? model.vector_scale[phase] : static_cast<float>(result_it->second.vector_scale));
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const auto linear_it = phase_linears.find(phase);
        const auto vector_it = phase_vectors.find(phase);
        const auto result_it = phase_results.find(phase);
        const std::vector<float> *linear = linear_it == phase_linears.end() ? nullptr : &linear_it->second;
        const std::vector<float> *vectors = vector_it == phase_vectors.end() ? nullptr : &vector_it->second;
        const double output_vector_scale = result_it == phase_results.end() ? model.vector_scale[phase] : result_it->second.vector_scale;
        const size_t phase_base = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            int16_t q_linear = model.linear_quant[phase_base + param_id];
            if (linear != nullptr) {
                const int raw_q = static_cast<int>(std::llround(static_cast<double>((*linear)[param_id]) / static_cast<double>(model.linear_scale[phase])));
                q_linear = static_cast<int16_t>(clamp_value(raw_q, -32768, 32767));
            }
            write_value<int16_t>(out, q_linear);
            const size_t vector_idx = static_cast<size_t>(param_id) * model.dim;
            if (vectors == nullptr) {
                out.write(reinterpret_cast<const char *>(&model.vector_quant[phase][vector_idx]), model.dim);
            } else {
                for (int d = 0; d < model.dim; ++d) {
                    const size_t idx = vector_idx + static_cast<size_t>(d);
                    const int raw_q = static_cast<int>(std::llround(static_cast<double>((*vectors)[idx]) / output_vector_scale));
                    const int8_t q_vec = static_cast<int8_t>(clamp_value(raw_q, -127, 127));
                    write_value<int8_t>(out, q_vec);
                }
            }
        }
    }
}

void write_summary(const Config &cfg, const Egev4Model &model, const std::map<int, PhaseResult> &phase_results, uint64_t total_elapsed_ms) {
    const fs::path summary_path(cfg.summary);
    if (!summary_path.parent_path().empty()) {
        fs::create_directories(summary_path.parent_path());
    }
    std::ofstream out(summary_path);
    if (!out) {
        throw std::runtime_error("cannot write " + summary_path.string());
    }
    out << std::setprecision(10);
    out << "optimizer=7.7-beta linear+FM joint-finetune streaming Adam\n";
    out << "comparison_rule=Only compare strength against candidates trained with the same input_egev4, data_root, train_ids, holdout_ids, and record caps.\n";
    out << "input_egev4=" << cfg.input_egev4 << "\n";
    out << "data_root=" << cfg.data_root << "\n";
    out << "phases=" << join_ints(cfg.phases) << "\n";
    out << "train_ids=" << join_ints(cfg.train_ids) << "\n";
    out << "holdout_ids=" << join_ints(cfg.holdout_ids) << "\n";
    out << "max_records_per_file=" << cfg.max_records_per_file << "\n";
    out << "holdout_max_records_per_file=" << cfg.holdout_max_records_per_file << "\n";
    out << "train_metric_max_records_per_file=" << cfg.train_metric_max_records_per_file << "\n";
    out << "chunk_records=" << cfg.chunk_records << "\n";
    out << "shuffle_files=" << (cfg.shuffle_files ? "true" : "false") << "\n";
    out << "dim=" << model.dim << "\n";
    out << "epochs=" << cfg.epochs << "\n";
    out << "linear_lr=" << cfg.linear_lr << "\n";
    out << "vector_lr=" << cfg.vector_lr << "\n";
    out << "loss=" << cfg.loss << "\n";
    out << "residual_clip=" << cfg.residual_clip << "\n";
    out << "huber_delta=" << cfg.huber_delta << "\n";
    out << "linear_l2=" << cfg.linear_l2 << "\n";
    out << "vector_l2=" << cfg.vector_l2 << "\n";
    out << "weight_decay=" << cfg.weight_decay << "\n";
    out << "early_stop_patience=" << cfg.early_stop_patience << "\n";
    out << "seed=" << cfg.seed << "\n";
    out << "output_egev4=" << cfg.output_egev4 << "\n";
    out << "total_elapsed_ms=" << total_elapsed_ms << "\n";
    uint64_t total_n_train = 0;
    uint64_t total_n_holdout = 0;
    for (const auto &[phase, result] : phase_results) {
        (void)phase;
        total_n_train += result.n_train;
        total_n_holdout += result.n_holdout;
    }
    out << "total_n_train=" << total_n_train << "\n";
    out << "total_n_holdout=" << total_n_holdout << "\n";
    for (const auto &[phase, result] : phase_results) {
        out << "\n[phase " << phase << "]\n";
        out << "n_train=" << result.n_train << "\n";
        out << "n_holdout=" << result.n_holdout << "\n";
        out << "best_epoch=" << result.best_epoch << "\n";
        out << "linear_scale=" << model.linear_scale[phase] << "\n";
        out << "input_vector_scale=" << model.vector_scale[phase] << "\n";
        out << "output_vector_scale=" << result.vector_scale << "\n";
        out << "n_initialized_best=" << result.n_initialized_best << "\n";
        out << "max_abs_linear_delta=" << result.max_abs_delta << "\n";
        out << "max_abs_vector_delta=" << result.max_abs_vector_delta << "\n";
        out << "vector_max_abs=" << result.vector_max_abs << "\n";
        out << "mean_abs_delta_initialized=" << result.mean_abs_delta_initialized << "\n";
        out << "changed_linear_values=" << result.changed_linear_values << "\n";
        out << "changed_vector_values=" << result.changed_vector_values << "\n";
        out << "clamped_linear_values=" << result.clamped_linear_values << "\n";
        out << "clamped_vector_values=" << result.clamped_vector_values << "\n";
        out << "epoch\ttrain_n\ttrain_mse\ttrain_mae\tholdout_n\tholdout_mse\tholdout_mae\tmax_abs_error\tn_initialized\telapsed_ms\n";
        for (const EpochRow &row : result.history) {
            out << row.epoch << '\t'
                << row.train.n << '\t'
                << row.train.mse << '\t'
                << row.train.mae << '\t'
                << row.holdout.n << '\t'
                << row.holdout.mse << '\t'
                << row.holdout.mae << '\t'
                << row.holdout.max_abs_error << '\t'
                << row.n_initialized << '\t'
                << row.elapsed_ms << "\n";
        }
    }
}

int main(int argc, char **argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const uint64_t start_ms = now_ms();
        const Egev4Model model = load_egev4_model(cfg.input_egev4);
        std::map<int, std::vector<float>> phase_linears;
        std::map<int, std::vector<float>> phase_vectors;
        std::map<int, PhaseResult> phase_results;
        for (int phase : cfg.phases) {
            std::vector<float> best_linear;
            std::vector<float> best_vectors;
            PhaseResult result = train_phase(cfg, fs::path(cfg.data_root), phase, model, best_linear, best_vectors);
            phase_linears[phase] = std::move(best_linear);
            phase_vectors[phase] = std::move(best_vectors);
            phase_results[phase] = result;
            const EpochRow &best_row = result.history[static_cast<size_t>(result.best_epoch)];
            std::cout << "phase " << phase
                      << " n_train " << result.n_train
                      << " n_holdout " << result.n_holdout
                      << " best_epoch " << result.best_epoch
                      << " best_holdout_MAE " << std::setprecision(8) << best_row.holdout.mae
                      << " changed_linear_values " << result.changed_linear_values
                      << " changed_vector_values " << result.changed_vector_values
                      << " max_abs_linear_delta " << result.max_abs_delta
                      << " max_abs_vector_delta " << result.max_abs_vector_delta
                      << std::endl;
        }
        write_egev4(cfg, model, phase_linears, phase_vectors, phase_results);
        const uint64_t total_elapsed_ms = now_ms() - start_ms;
        write_summary(cfg, model, phase_results, total_elapsed_ms);
        std::cout << "elapsed_ms " << total_elapsed_ms << std::endl;
        std::cout << "wrote " << cfg.output_egev4 << std::endl;
        std::cout << "summary " << cfg.summary << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
