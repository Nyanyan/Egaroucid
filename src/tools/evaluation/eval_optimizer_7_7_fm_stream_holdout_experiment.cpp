/*
    Egaroucid evaluation experiment

    7.7-beta-linear fixed FM-only optimizer with streaming shuffled chunks.
    This tool is separate from eval_optimizer_cuda.cu and from the current-model
    FM optimizers so that 7.7-beta experiments keep their own data contract.
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    int dim = 16;
    int epochs = 6;
    int max_records_per_file = 20000;
    int holdout_max_records_per_file = -1;
    int train_metric_max_records_per_file = -1;
    int chunk_records = 65536;
    int early_stop_patience = 1;
    double lr = 0.0004;
    double init_scale = 0.005;
    double residual_clip = 16.0;
    std::string loss = "pseudo-huber";
    double huber_delta = 8.0;
    double l2 = 0.00001;
    double weight_decay = 0.0;
    uint64_t seed = 20260710;
    bool shuffle_files = true;
};

struct Egev4Linear {
    int input_dim = 0;
    std::array<float, N_PHASES> linear_scale{};
    std::vector<int16_t> linear_quant;
    std::vector<float> linear_value;
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
    double vector_scale = 1.0;
    double vector_max_abs = 0.0;
    uint64_t nonzero_vector_values = 0;
    std::vector<EpochRow> history;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message + "\n"
        "usage: eval_optimizer_7_7_fm_stream_holdout_experiment "
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
        } else if (arg == "--dim") {
            cfg.dim = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--epochs") {
            cfg.epochs = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--lr") {
            cfg.lr = std::stod(require_value(i, argc, argv));
        } else if (arg == "--init-scale") {
            cfg.init_scale = std::stod(require_value(i, argc, argv));
        } else if (arg == "--residual-clip") {
            cfg.residual_clip = std::stod(require_value(i, argc, argv));
        } else if (arg == "--loss") {
            cfg.loss = require_value(i, argc, argv);
        } else if (arg == "--huber-delta") {
            cfg.huber_delta = std::stod(require_value(i, argc, argv));
        } else if (arg == "--l2") {
            cfg.l2 = std::stod(require_value(i, argc, argv));
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
    if (cfg.dim <= 0 || cfg.dim > MAX_DIM) {
        usage_error("--dim must be in 1..16");
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

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

float deterministic_init(uint64_t seed, int phase, uint32_t param_id, int dim_idx, double init_scale) {
    uint64_t x = seed;
    x ^= static_cast<uint64_t>(phase + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= static_cast<uint64_t>(param_id + 1) * 0xbf58476d1ce4e5b9ULL;
    x ^= static_cast<uint64_t>(dim_idx + 1) * 0x94d049bb133111ebULL;
    const uint64_t h = splitmix64(x);
    const double unit = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
    return static_cast<float>((unit * 2.0 - 1.0) * init_scale);
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

Egev4Linear load_egev4_linear(const fs::path &path) {
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
    const int input_dim = read_i32(header + 30);
    if (version != VERSION_LINEAR_FM_INT16_INT8 || n_phase != N_PHASES || n_param != N_PARAMS_PER_PHASE || input_dim <= 0) {
        throw std::runtime_error("EGEV4 header mismatch: " + path.string());
    }

    Egev4Linear out;
    out.input_dim = input_dim;
    in.read(reinterpret_cast<char *>(out.linear_scale.data()), sizeof(float) * N_PHASES);
    std::array<float, N_PHASES> ignored_vector_scale{};
    in.read(reinterpret_cast<char *>(ignored_vector_scale.data()), sizeof(float) * N_PHASES);
    if (!in) {
        throw std::runtime_error("broken EGEV4 scales: " + path.string());
    }
    out.linear_quant.assign(static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE, 0);
    out.linear_value.assign(static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE, 0.0f);
    std::vector<char> skip_vec(static_cast<size_t>(input_dim), 0);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            int16_t q = 0;
            in.read(reinterpret_cast<char *>(&q), sizeof(q));
            in.read(skip_vec.data(), input_dim);
            if (!in) {
                throw std::runtime_error("broken EGEV4 payload: " + path.string());
            }
            const size_t idx = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE + param_id;
            out.linear_quant[idx] = q;
            out.linear_value[idx] = static_cast<float>(static_cast<double>(q) * out.linear_scale[phase]);
        }
    }
    return out;
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

void initialize_param(uint32_t param_id, int phase, const Config &cfg, std::vector<float> &vectors, std::vector<float> &m, std::vector<float> &v, std::vector<uint8_t> &initialized) {
    if (initialized[param_id] != 0) {
        return;
    }
    const size_t base = static_cast<size_t>(param_id) * cfg.dim;
    for (int d = 0; d < cfg.dim; ++d) {
        vectors[base + d] = deterministic_init(cfg.seed, phase, param_id, d, cfg.init_scale);
        m[base + d] = 0.0f;
        v[base + d] = 0.0f;
    }
    initialized[param_id] = 1;
}

double predict_record(
    const float *linear,
    const std::vector<float> &vectors,
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
            const double vec = vectors[base + d];
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
    const std::vector<float> &vectors,
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
    int phase,
    const Config &cfg,
    const float *linear,
    std::vector<float> &vectors,
    std::vector<float> &m,
    std::vector<float> &v,
    std::vector<uint8_t> &initialized,
    uint64_t &step,
    double &beta1_power,
    double &beta2_power,
    std::array<double, MAX_DIM> &sums,
    std::array<double, MAX_DIM> &sums_sq
) {
    for (uint32_t param_id : record.active_ids) {
        initialize_param(param_id, phase, cfg, vectors, m, v, initialized);
    }
    const double pred = predict_record(linear, vectors, record, cfg.dim, sums, sums_sq);
    double residual = residual_for_loss(static_cast<double>(record.score) - pred, cfg);

    ++step;
    beta1_power *= 0.9;
    beta2_power *= 0.999;
    const double lr_t = cfg.lr * std::sqrt(1.0 - beta2_power) / (1.0 - beta1_power);
    const double decay = cfg.weight_decay > 0.0 ? std::max(0.0, 1.0 - cfg.lr * cfg.weight_decay) : 1.0;

    for (uint32_t param_id : record.active_ids) {
        const size_t base = static_cast<size_t>(param_id) * cfg.dim;
        for (int d = 0; d < cfg.dim; ++d) {
            const double vec = vectors[base + d];
            const double grad = -residual * (sums[d] - vec) + cfg.l2 * vec;
            const double next_m = 0.9 * m[base + d] + 0.1 * grad;
            const double next_v = 0.999 * v[base + d] + 0.001 * grad * grad;
            double next_vec = vec - lr_t * next_m / (std::sqrt(next_v) + 1.0e-8);
            next_vec *= decay;
            m[base + d] = static_cast<float>(next_m);
            v[base + d] = static_cast<float>(next_v);
            vectors[base + d] = static_cast<float>(next_vec);
        }
    }
}

uint64_t train_one_epoch(
    const fs::path &root,
    int phase,
    const std::vector<int> &train_ids,
    int epoch,
    const Config &cfg,
    const float *linear,
    std::vector<float> &vectors,
    std::vector<float> &m,
    std::vector<float> &v,
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
                update_record(record, phase, cfg, linear, vectors, m, v, initialized, step, beta1_power, beta2_power, sums, sums_sq);
                ++updates;
            }
            ++chunk_idx;
        }
    }
    return updates;
}

double vector_scale_for(const std::vector<float> &vectors, const std::vector<uint8_t> &initialized, int dim, double &max_abs) {
    max_abs = 0.0;
    for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
        if (initialized[param_id] == 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(param_id) * dim;
        for (int d = 0; d < dim; ++d) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(vectors[base + d])));
        }
    }
    return max_abs > 0.0 ? max_abs / 127.0 : 1.0;
}

PhaseResult train_phase(const Config &cfg, const fs::path &data_root, int phase, const Egev4Linear &linear_file, std::vector<float> &best_vectors_out) {
    const uint64_t start_ms = now_ms();
    const float *linear = linear_file.linear_value.data() + static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
    std::vector<float> vectors(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> m(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> v(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
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
    initial.train = compute_metrics(data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file, linear, vectors, cfg.dim);
    initial.holdout = compute_metrics(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file, linear, vectors, cfg.dim);
    initial.elapsed_ms = now_ms() - start_ms;
    result.history.push_back(initial);

    double best_holdout_mae = initial.holdout.mae;
    best_vectors_out.assign(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    result.best_epoch = 0;

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const uint64_t updates = train_one_epoch(
            data_root, phase, cfg.train_ids, epoch, cfg, linear, vectors, m, v,
            initialized, step, beta1_power, beta2_power
        );
        if (updates == 0) {
            throw std::runtime_error("no records trained for phase " + std::to_string(phase));
        }
        EpochRow row;
        row.epoch = epoch;
        row.n_initialized = count_initialized(initialized);
        row.train = compute_metrics(data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file, linear, vectors, cfg.dim);
        row.holdout = compute_metrics(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file, linear, vectors, cfg.dim);
        row.elapsed_ms = now_ms() - start_ms;
        result.history.push_back(row);

        if (row.holdout.mae < best_holdout_mae) {
            best_holdout_mae = row.holdout.mae;
            result.best_epoch = epoch;
            best_vectors_out = vectors;
            best_initialized = initialized;
        } else if (cfg.early_stop_patience > 0 && epoch - result.best_epoch >= cfg.early_stop_patience) {
            break;
        }
    }

    result.vector_scale = vector_scale_for(best_vectors_out, best_initialized, cfg.dim, result.vector_max_abs);
    for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
        if (best_initialized[param_id] == 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(param_id) * cfg.dim;
        for (int d = 0; d < cfg.dim; ++d) {
            const int q = clamp_value(static_cast<int>(std::llround(best_vectors_out[base + d] / result.vector_scale)), -127, 127);
            result.nonzero_vector_values += q != 0;
        }
    }
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

void write_egev4(const Config &cfg, const Egev4Linear &linear_file, const std::map<int, std::vector<float>> &phase_vectors, const std::map<int, PhaseResult> &phase_results) {
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
    write_value<int32_t>(out, cfg.dim);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        write_value<float>(out, linear_file.linear_scale[phase]);
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        auto it = phase_results.find(phase);
        write_value<float>(out, it == phase_results.end() ? 1.0f : static_cast<float>(it->second.vector_scale));
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const auto vec_it = phase_vectors.find(phase);
        const std::vector<float> *vectors = vec_it == phase_vectors.end() ? nullptr : &vec_it->second;
        const double vector_scale = phase_results.count(phase) ? phase_results.at(phase).vector_scale : 1.0;
        const size_t phase_base = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            write_value<int16_t>(out, linear_file.linear_quant[phase_base + param_id]);
            for (int d = 0; d < cfg.dim; ++d) {
                int q_vec = 0;
                if (vectors != nullptr && vector_scale > 0.0) {
                    const size_t base = static_cast<size_t>(param_id) * cfg.dim;
                    q_vec = clamp_value(static_cast<int>(std::llround((*vectors)[base + d] / vector_scale)), -127, 127);
                }
                write_value<int8_t>(out, static_cast<int8_t>(q_vec));
            }
        }
    }
}

void write_summary(const Config &cfg, const std::map<int, PhaseResult> &phase_results, uint64_t total_elapsed_ms) {
    const fs::path summary_path(cfg.summary);
    if (!summary_path.parent_path().empty()) {
        fs::create_directories(summary_path.parent_path());
    }
    std::ofstream out(summary_path);
    if (!out) {
        throw std::runtime_error("cannot write " + summary_path.string());
    }
    out << std::setprecision(10);
    out << "optimizer=7.7-beta-linear-fixed FM-only streaming Adam\n";
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
    out << "dim=" << cfg.dim << "\n";
    out << "epochs=" << cfg.epochs << "\n";
    out << "lr=" << cfg.lr << "\n";
    out << "init_scale=" << cfg.init_scale << "\n";
    out << "loss=" << cfg.loss << "\n";
    out << "residual_clip=" << cfg.residual_clip << "\n";
    out << "huber_delta=" << cfg.huber_delta << "\n";
    out << "l2=" << cfg.l2 << "\n";
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
        out << "vector_scale=" << result.vector_scale << "\n";
        out << "vector_max_abs=" << result.vector_max_abs << "\n";
        out << "nonzero_vector_values=" << result.nonzero_vector_values << "\n";
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
        const Egev4Linear linear_file = load_egev4_linear(cfg.input_egev4);
        std::map<int, std::vector<float>> phase_vectors;
        std::map<int, PhaseResult> phase_results;
        for (int phase : cfg.phases) {
            std::vector<float> best_vectors;
            PhaseResult result = train_phase(cfg, fs::path(cfg.data_root), phase, linear_file, best_vectors);
            phase_vectors[phase] = std::move(best_vectors);
            phase_results[phase] = result;
            const EpochRow &best_row = result.history[static_cast<size_t>(result.best_epoch)];
            std::cout << "phase " << phase
                      << " n_train " << result.n_train
                      << " n_holdout " << result.n_holdout
                      << " best_epoch " << result.best_epoch
                      << " best_holdout_MAE " << std::setprecision(8) << best_row.holdout.mae
                      << " nonzero_vector_values " << result.nonzero_vector_values
                      << std::endl;
        }
        write_egev4(cfg, linear_file, phase_vectors, phase_results);
        const uint64_t total_elapsed_ms = now_ms() - start_ms;
        write_summary(cfg, phase_results, total_elapsed_ms);
        std::cout << "elapsed_ms " << total_elapsed_ms << std::endl;
        std::cout << "wrote " << cfg.output_egev4 << std::endl;
        std::cout << "summary " << cfg.summary << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
