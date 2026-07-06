/*
    Egaroucid evaluation experiment

    Current-linear fixed joint dual-type FM optimizer with streaming chunk
    shuffling and pseudo-Huber loss experiment.

    This file is intentionally separate from eval_optimizer_cuda.cu.  It is an
    experiment tool for comparing training-code changes while preserving the
    current source files.  The linear term still uses every current-model
    feature, including the stone-count feature.  Two independent FM vector
    banks are trained against the same residual:

    - cross-type bank: pairs between different current-model pattern types
    - same-type bank: pairs among the four orientations of each pattern type

    Interactions with the stone-count feature are excluded from both FM terms.
    The tool writes two EGEV4 files, one for each vector bank, so the existing
    dual-type FM evaluator can load them as cross.egev4@same.egev4@w_cross@w_same.
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
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

constexpr int N_PHASES = 60;
constexpr int N_PATTERN_FEATURES = 64;
constexpr int N_PATTERN_TYPES = 16;
constexpr int N_FEATURES = 65;
constexpr int N_PATTERN_PARAMS_RAW = 612360;
constexpr int MAX_STONE_NUM = 65;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int STEP = 32;
constexpr int N_ZEROS_PLUS = 1 << 12;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;
constexpr int RECORD_BYTES = 136;
constexpr uint64_t FM_BANK_SALT_CROSS = 0x43524f53535f464dULL; // "CROSS_FM"
constexpr uint64_t FM_BANK_SALT_SAME = 0x53414d455f464d31ULL;  // "SAME_FM1"

const std::array<int, 17> EVAL_SIZES = {
    6561, 19683, 6561, 19683, 6561, 19683, 2187, 59049, 59049,
    59049, 59049, 59049, 59049, 59049, 59049, 59049, MAX_STONE_NUM,
};

std::array<int, N_FEATURES> make_feature_offsets() {
    std::array<int, N_FEATURES> offsets{};
    int start = 0;
    int last_eval = -1;
    for (int i = 0; i < N_FEATURES; ++i) {
        const int eval_idx = (i < N_PATTERN_FEATURES) ? (i / 4) : 16;
        if (last_eval >= 0 && eval_idx > last_eval) {
            start += EVAL_SIZES[last_eval];
        }
        offsets[i] = start;
        last_eval = eval_idx;
    }
    return offsets;
}

const std::array<int, N_FEATURES> FEATURE_OFFSETS = make_feature_offsets();

int pattern_type_from_feature(int feature_idx) {
    return feature_idx < N_PATTERN_FEATURES ? feature_idx / 4 : -1;
}

struct Config {
    std::string input_egev2;
    std::string data_root;
    std::string output_cross_egev4;
    std::string output_same_egev4;
    std::string summary;
    std::string timestamp;
    std::vector<int> phases;
    std::vector<int> train_ids;
    std::vector<int> holdout_ids;
    int dim = 8;
    int epochs = 6;
    int max_records_per_file = 20000;
    int holdout_max_records_per_file = -1;
    int train_metric_max_records_per_file = -1;
    int chunk_records = 65536;
    int early_stop_patience = 1;
    double lr = 0.0004;
    double init_scale = 0.005;
    double residual_clip = 16.0;
    std::string loss = "clipped-residual";
    double huber_delta = 16.0;
    double l2 = 0.00001;
    double weight_decay = 0.0;
    double cross_weight = 1.0;
    double same_weight = 1.0;
    uint64_t seed = 20260702;
    bool shuffle_files = true;
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
    double cross_vector_scale = 1.0;
    double cross_vector_max_abs = 0.0;
    uint64_t cross_nonzero_vector_values = 0;
    double same_vector_scale = 1.0;
    double same_vector_max_abs = 0.0;
    uint64_t same_nonzero_vector_values = 0;
    std::vector<EpochRow> history;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message + "\n"
        "usage: eval_optimizer_current_fm_stream_holdout_experiment "
        "--input-egev2 FILE --data-root DIR --phases 6-10 "
        "--train-ids 0,1,2 --holdout-ids 3 "
        "--output-cross-egev4 FILE --output-same-egev4 FILE --summary FILE [options]");
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
            continue;
        }
        const int start = std::stoi(elem.substr(0, dash));
        const int end = std::stoi(elem.substr(dash + 1));
        if (end < start) {
            usage_error("invalid range: " + elem);
        }
        for (int value = start; value <= end; ++value) {
            values.push_back(value);
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
        if (arg == "--input-egev2") {
            cfg.input_egev2 = require_value(i, argc, argv);
        } else if (arg == "--data-root") {
            cfg.data_root = require_value(i, argc, argv);
        } else if (arg == "--phase") {
            cfg.phases = {std::stoi(require_value(i, argc, argv))};
        } else if (arg == "--phases") {
            cfg.phases = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--train-ids") {
            cfg.train_ids = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--holdout-ids") {
            cfg.holdout_ids = parse_int_list(require_value(i, argc, argv));
        } else if (arg == "--output-cross-egev4") {
            cfg.output_cross_egev4 = require_value(i, argc, argv);
        } else if (arg == "--output-same-egev4") {
            cfg.output_same_egev4 = require_value(i, argc, argv);
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
        } else if (arg == "--cross-weight") {
            cfg.cross_weight = std::stod(require_value(i, argc, argv));
        } else if (arg == "--same-weight") {
            cfg.same_weight = std::stod(require_value(i, argc, argv));
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
        } else if (arg == "--help" || arg == "-h") {
            usage_error("");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }

    if (cfg.input_egev2.empty()) {
        usage_error("--input-egev2 is required");
    }
    if (cfg.data_root.empty()) {
        usage_error("--data-root is required");
    }
    if (cfg.output_cross_egev4.empty()) {
        usage_error("--output-cross-egev4 is required");
    }
    if (cfg.output_same_egev4.empty()) {
        usage_error("--output-same-egev4 is required");
    }
    if (cfg.summary.empty()) {
        usage_error("--summary is required");
    }
    if (cfg.phases.empty()) {
        usage_error("--phase or --phases is required");
    }
    if (cfg.train_ids.empty() || cfg.holdout_ids.empty()) {
        usage_error("--train-ids and --holdout-ids are required");
    }
    if (cfg.dim <= 0 || cfg.dim > 64) {
        usage_error("--dim must be in 1..64");
    }
    if (cfg.epochs <= 0) {
        usage_error("--epochs must be positive");
    }
    if (cfg.chunk_records <= 0) {
        usage_error("--chunk-records must be positive");
    }
    if (cfg.loss != "clipped-residual" && cfg.loss != "pseudo-huber") {
        usage_error("--loss must be clipped-residual or pseudo-huber");
    }
    if (cfg.loss == "pseudo-huber" && cfg.huber_delta <= 0.0) {
        usage_error("--huber-delta must be positive when --loss pseudo-huber is used");
    }
    if (!std::isfinite(cfg.cross_weight) || !std::isfinite(cfg.same_weight)) {
        usage_error("--cross-weight and --same-weight must be finite");
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

float deterministic_init(uint64_t seed, int phase, uint32_t param_id, int dim_idx, uint64_t bank_salt, double init_scale) {
    uint64_t x = seed;
    x ^= static_cast<uint64_t>(phase + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= static_cast<uint64_t>(param_id + 1) * 0xbf58476d1ce4e5b9ULL;
    x ^= static_cast<uint64_t>(dim_idx + 1) * 0x94d049bb133111ebULL;
    x ^= bank_salt;
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

std::vector<int16_t> load_unzip_egev2(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    int32_t n_compressed = 0;
    in.read(reinterpret_cast<char *>(&n_compressed), sizeof(n_compressed));
    if (!in || n_compressed <= 0) {
        throw std::runtime_error("broken egev2 header: " + path.string());
    }
    std::vector<int16_t> out;
    out.reserve(static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE);
    for (int32_t i = 0; i < n_compressed; ++i) {
        int16_t value = 0;
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("broken egev2 body: " + path.string());
        }
        if (value >= N_ZEROS_PLUS) {
            out.insert(out.end(), static_cast<size_t>(value - N_ZEROS_PLUS), 0);
        } else {
            out.push_back(value);
        }
    }
    const size_t expected = static_cast<size_t>(N_PHASES) * N_PARAMS_PER_PHASE;
    if (out.size() != expected) {
        throw std::runtime_error("expanded egev2 size mismatch: " + std::to_string(out.size()) + " expected " + std::to_string(expected));
    }
    return out;
}

fs::path data_path(const fs::path &root, int phase, int file_id) {
    return root / std::to_string(phase) / (std::to_string(file_id) + ".dat");
}

uint64_t count_file_records(const fs::path &path, int max_records_per_file) {
    const uint64_t file_records = static_cast<uint64_t>(fs::file_size(path) / RECORD_BYTES);
    if (max_records_per_file > 0) {
        return std::min<uint64_t>(file_records, static_cast<uint64_t>(max_records_per_file));
    }
    return file_records;
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
    for (int i = 0; i < N_FEATURES; ++i) {
        uint16_t feature = 0;
        std::memcpy(&feature, raw.data() + 4 + i * 2, sizeof(feature));
        record.active_ids[i] = static_cast<uint32_t>(FEATURE_OFFSETS[i] + static_cast<int>(feature));
    }
    int16_t score = 0;
    std::memcpy(&score, raw.data() + 4 + N_FEATURES * 2, sizeof(score));
    record.score = static_cast<float>(score);
    return true;
}

void initialize_param(
    uint32_t param_id,
    int phase,
    const Config &cfg,
    uint64_t bank_salt,
    std::vector<float> &vectors,
    std::vector<float> &m,
    std::vector<float> &v,
    std::vector<uint8_t> &initialized
) {
    if (initialized[param_id] != 0) {
        return;
    }
    const size_t base = static_cast<size_t>(param_id) * cfg.dim;
    for (int d = 0; d < cfg.dim; ++d) {
        vectors[base + d] = deterministic_init(cfg.seed, phase, param_id, d, bank_salt, cfg.init_scale);
        m[base + d] = 0.0f;
        v[base + d] = 0.0f;
    }
    initialized[param_id] = 1;
}

struct InteractionScratch {
    std::vector<double> sums;
    std::vector<double> sums_sq;
    std::vector<double> group_sums;
    std::vector<double> group_sums_sq;

    explicit InteractionScratch(int dim)
        : sums(static_cast<size_t>(dim), 0.0),
          sums_sq(static_cast<size_t>(dim), 0.0),
          group_sums(static_cast<size_t>(N_PATTERN_TYPES * dim), 0.0),
          group_sums_sq(static_cast<size_t>(N_PATTERN_TYPES * dim), 0.0) {}

    void clear() {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(sums_sq.begin(), sums_sq.end(), 0.0);
        std::fill(group_sums.begin(), group_sums.end(), 0.0);
        std::fill(group_sums_sq.begin(), group_sums_sq.end(), 0.0);
    }
};

double predict_record(
    const std::vector<float> &linear,
    const std::vector<float> &cross_vectors,
    const std::vector<float> &same_vectors,
    const Record &record,
    int dim,
    double cross_weight,
    double same_weight,
    InteractionScratch &cross_scratch,
    InteractionScratch &same_scratch
) {
    cross_scratch.clear();
    same_scratch.clear();
    double value = 0.0;
    for (int feature_idx = 0; feature_idx < N_FEATURES; ++feature_idx) {
        const uint32_t param_id = record.active_ids[feature_idx];
        value += linear[param_id];
        const int pattern_type = pattern_type_from_feature(feature_idx);
        if (pattern_type < 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(param_id) * dim;
        const size_t group_base = static_cast<size_t>(pattern_type) * dim;
        for (int d = 0; d < dim; ++d) {
            const double cross_vec = cross_vectors[base + d];
            cross_scratch.sums[d] += cross_vec;
            cross_scratch.sums_sq[d] += cross_vec * cross_vec;
            cross_scratch.group_sums[group_base + d] += cross_vec;
            cross_scratch.group_sums_sq[group_base + d] += cross_vec * cross_vec;

            const double same_vec = same_vectors[base + d];
            same_scratch.group_sums[group_base + d] += same_vec;
            same_scratch.group_sums_sq[group_base + d] += same_vec * same_vec;
        }
    }

    double cross_interaction = 0.0;
    double same_interaction = 0.0;
    for (int d = 0; d < dim; ++d) {
        double cross_same_type_interaction = 0.0;
        for (int pattern_type = 0; pattern_type < N_PATTERN_TYPES; ++pattern_type) {
            const size_t group_idx = static_cast<size_t>(pattern_type) * dim + d;
            cross_same_type_interaction +=
                cross_scratch.group_sums[group_idx] * cross_scratch.group_sums[group_idx] -
                cross_scratch.group_sums_sq[group_idx];
            same_interaction +=
                same_scratch.group_sums[group_idx] * same_scratch.group_sums[group_idx] -
                same_scratch.group_sums_sq[group_idx];
        }
        cross_interaction +=
            cross_scratch.sums[d] * cross_scratch.sums[d] -
            cross_scratch.sums_sq[d] -
            cross_same_type_interaction;
    }
    return value + 0.5 * cross_weight * cross_interaction + 0.5 * same_weight * same_interaction;
}

Metrics compute_metrics(
    const fs::path &root,
    int phase,
    const std::vector<int> &file_ids,
    int max_records_per_file,
    const std::vector<float> &linear,
    const std::vector<float> &cross_vectors,
    const std::vector<float> &same_vectors,
    const Config &cfg
) {
    Metrics metrics;
    InteractionScratch cross_scratch(cfg.dim);
    InteractionScratch same_scratch(cfg.dim);
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
            const double pred = predict_record(
                linear, cross_vectors, same_vectors, record, cfg.dim,
                cfg.cross_weight, cfg.same_weight, cross_scratch, same_scratch
            );
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
    const std::vector<float> &linear,
    std::vector<float> &cross_vectors,
    std::vector<float> &cross_m,
    std::vector<float> &cross_v,
    std::vector<uint8_t> &cross_initialized,
    std::vector<float> &same_vectors,
    std::vector<float> &same_m,
    std::vector<float> &same_v,
    std::vector<uint8_t> &same_initialized,
    uint64_t &step,
    double &beta1_power,
    double &beta2_power,
    InteractionScratch &cross_scratch,
    InteractionScratch &same_scratch
) {
    for (int feature_idx = 0; feature_idx < N_PATTERN_FEATURES; ++feature_idx) {
        const uint32_t param_id = record.active_ids[feature_idx];
        const int pattern_type = pattern_type_from_feature(feature_idx);
        if (pattern_type < 0) {
            continue;
        }
        initialize_param(param_id, phase, cfg, FM_BANK_SALT_CROSS, cross_vectors, cross_m, cross_v, cross_initialized);
        initialize_param(param_id, phase, cfg, FM_BANK_SALT_SAME, same_vectors, same_m, same_v, same_initialized);
    }

    const double pred = predict_record(
        linear, cross_vectors, same_vectors, record, cfg.dim,
        cfg.cross_weight, cfg.same_weight, cross_scratch, same_scratch
    );
    double residual = static_cast<double>(record.score) - pred;
    residual = residual_for_loss(residual, cfg);

    ++step;
    beta1_power *= 0.9;
    beta2_power *= 0.999;
    const double lr_t = cfg.lr * std::sqrt(1.0 - beta2_power) / (1.0 - beta1_power);
    const double decay = cfg.weight_decay > 0.0 ? std::max(0.0, 1.0 - cfg.lr * cfg.weight_decay) : 1.0;

    for (int feature_idx = 0; feature_idx < N_PATTERN_FEATURES; ++feature_idx) {
        const uint32_t param_id = record.active_ids[feature_idx];
        const int pattern_type = pattern_type_from_feature(feature_idx);
        if (pattern_type < 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(param_id) * cfg.dim;
        const size_t group_base = static_cast<size_t>(pattern_type) * cfg.dim;
        for (int d = 0; d < cfg.dim; ++d) {
            const double cross_vec = cross_vectors[base + d];
            const double cross_grad =
                -residual * cfg.cross_weight * (cross_scratch.sums[d] - cross_scratch.group_sums[group_base + d]) +
                cfg.l2 * cross_vec;
            const double next_cross_m = 0.9 * cross_m[base + d] + 0.1 * cross_grad;
            const double next_cross_v = 0.999 * cross_v[base + d] + 0.001 * cross_grad * cross_grad;
            double next_cross_vec = cross_vec - lr_t * next_cross_m / (std::sqrt(next_cross_v) + 1.0e-8);
            next_cross_vec *= decay;
            cross_m[base + d] = static_cast<float>(next_cross_m);
            cross_v[base + d] = static_cast<float>(next_cross_v);
            cross_vectors[base + d] = static_cast<float>(next_cross_vec);

            const double same_vec = same_vectors[base + d];
            const double same_grad =
                -residual * cfg.same_weight * (same_scratch.group_sums[group_base + d] - same_vec) +
                cfg.l2 * same_vec;
            const double next_same_m = 0.9 * same_m[base + d] + 0.1 * same_grad;
            const double next_same_v = 0.999 * same_v[base + d] + 0.001 * same_grad * same_grad;
            double next_same_vec = same_vec - lr_t * next_same_m / (std::sqrt(next_same_v) + 1.0e-8);
            next_same_vec *= decay;
            same_m[base + d] = static_cast<float>(next_same_m);
            same_v[base + d] = static_cast<float>(next_same_v);
            same_vectors[base + d] = static_cast<float>(next_same_vec);
        }
    }
}

uint64_t train_one_epoch(
    const fs::path &root,
    int phase,
    const std::vector<int> &train_ids,
    int epoch,
    const Config &cfg,
    const std::vector<float> &linear,
    std::vector<float> &cross_vectors,
    std::vector<float> &cross_m,
    std::vector<float> &cross_v,
    std::vector<uint8_t> &cross_initialized,
    std::vector<float> &same_vectors,
    std::vector<float> &same_m,
    std::vector<float> &same_v,
    std::vector<uint8_t> &same_initialized,
    uint64_t &step,
    double &beta1_power,
    double &beta2_power
) {
    std::vector<int> file_order = train_ids;
    if (cfg.shuffle_files) {
        std::mt19937_64 file_rng(cfg.seed ^ (static_cast<uint64_t>(phase + 1) << 32) ^ static_cast<uint64_t>(epoch));
        std::shuffle(file_order.begin(), file_order.end(), file_rng);
    }

    uint64_t updates = 0;
    std::vector<Record> chunk;
    chunk.reserve(static_cast<size_t>(cfg.chunk_records));
    InteractionScratch cross_scratch(cfg.dim);
    InteractionScratch same_scratch(cfg.dim);

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
                update_record(
                    record, phase, cfg, linear,
                    cross_vectors, cross_m, cross_v, cross_initialized,
                    same_vectors, same_m, same_v, same_initialized,
                    step, beta1_power, beta2_power, cross_scratch, same_scratch
                );
                ++updates;
            }
            ++chunk_idx;
        }
    }
    return updates;
}

double vector_scale_for(const std::vector<float> &vectors, const std::vector<uint8_t> *initialized, int dim, double &max_abs) {
    max_abs = 0.0;
    if (initialized == nullptr) {
        for (float value : vectors) {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        }
    } else {
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            if ((*initialized)[param_id] == 0) {
                continue;
            }
            const size_t base = static_cast<size_t>(param_id) * dim;
            for (int d = 0; d < dim; ++d) {
                max_abs = std::max(max_abs, std::abs(static_cast<double>(vectors[base + d])));
            }
        }
    }
    return max_abs > 0.0 ? max_abs / 127.0 : 1.0;
}

uint64_t count_quantized_nonzero(
    const std::vector<float> &vectors,
    const std::vector<uint8_t> &initialized,
    int dim,
    double vector_scale
) {
    uint64_t n = 0;
    if (vector_scale <= 0.0) {
        return 0;
    }
    for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
        if (initialized[param_id] == 0) {
            continue;
        }
        const size_t base = static_cast<size_t>(param_id) * dim;
        for (int d = 0; d < dim; ++d) {
            const int q = clamp_value(static_cast<int>(std::llround(vectors[base + d] / vector_scale)), -127, 127);
            if (q != 0) {
                ++n;
            }
        }
    }
    return n;
}

PhaseResult train_phase(
    const Config &cfg,
    const fs::path &data_root,
    int phase,
    const std::vector<int16_t> &linear_values,
    std::vector<float> &best_cross_vectors_out,
    std::vector<float> &best_same_vectors_out
) {
    const uint64_t start_ms = now_ms();
    std::vector<float> linear(static_cast<size_t>(N_PARAMS_PER_PHASE), 0.0f);
    const size_t phase_base = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
    for (int i = 0; i < N_PARAMS_PER_PHASE; ++i) {
        linear[i] = static_cast<float>(static_cast<double>(linear_values[phase_base + i]) / STEP);
    }

    std::vector<float> cross_vectors(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> cross_m(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> cross_v(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<uint8_t> cross_initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);
    std::vector<uint8_t> best_cross_initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);

    std::vector<float> same_vectors(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> same_m(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<float> same_v(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    std::vector<uint8_t> same_initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);
    std::vector<uint8_t> best_same_initialized(static_cast<size_t>(N_PARAMS_PER_PHASE), 0);

    PhaseResult result;
    result.phase = phase;
    result.n_train = count_records(data_root, phase, cfg.train_ids, cfg.max_records_per_file);
    result.n_holdout = count_records(data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file);

    uint64_t step = 0;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    EpochRow initial;
    initial.epoch = 0;
    initial.n_initialized = 0;
    initial.train = compute_metrics(
        data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file,
        linear, cross_vectors, same_vectors, cfg
    );
    initial.holdout = compute_metrics(
        data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file,
        linear, cross_vectors, same_vectors, cfg
    );
    initial.elapsed_ms = now_ms() - start_ms;
    result.history.push_back(initial);

    double best_holdout_mae = initial.holdout.mae;
    best_cross_vectors_out.assign(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    best_same_vectors_out.assign(static_cast<size_t>(N_PARAMS_PER_PHASE) * cfg.dim, 0.0f);
    result.best_epoch = 0;

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const uint64_t updates = train_one_epoch(
            data_root, phase, cfg.train_ids, epoch, cfg, linear,
            cross_vectors, cross_m, cross_v, cross_initialized,
            same_vectors, same_m, same_v, same_initialized,
            step, beta1_power, beta2_power
        );
        if (updates == 0) {
            throw std::runtime_error("no records trained for phase " + std::to_string(phase));
        }

        EpochRow row;
        row.epoch = epoch;
        row.n_initialized = count_initialized(cross_initialized);
        row.train = compute_metrics(
            data_root, phase, cfg.train_ids, cfg.train_metric_max_records_per_file,
            linear, cross_vectors, same_vectors, cfg
        );
        row.holdout = compute_metrics(
            data_root, phase, cfg.holdout_ids, cfg.holdout_max_records_per_file,
            linear, cross_vectors, same_vectors, cfg
        );
        row.elapsed_ms = now_ms() - start_ms;
        result.history.push_back(row);

        if (row.holdout.mae < best_holdout_mae) {
            best_holdout_mae = row.holdout.mae;
            result.best_epoch = epoch;
            best_cross_vectors_out = cross_vectors;
            best_cross_initialized = cross_initialized;
            best_same_vectors_out = same_vectors;
            best_same_initialized = same_initialized;
        } else if (cfg.early_stop_patience > 0 && epoch - result.best_epoch >= cfg.early_stop_patience) {
            break;
        }
    }

    result.cross_vector_scale = vector_scale_for(best_cross_vectors_out, &best_cross_initialized, cfg.dim, result.cross_vector_max_abs);
    result.same_vector_scale = vector_scale_for(best_same_vectors_out, &best_same_initialized, cfg.dim, result.same_vector_max_abs);
    if (result.cross_vector_scale <= 0.0) {
        result.cross_vector_scale = 1.0;
    }
    if (result.same_vector_scale <= 0.0) {
        result.same_vector_scale = 1.0;
    }
    result.cross_nonzero_vector_values = count_quantized_nonzero(
        best_cross_vectors_out, best_cross_initialized, cfg.dim, result.cross_vector_scale
    );
    result.same_nonzero_vector_values = count_quantized_nonzero(
        best_same_vectors_out, best_same_initialized, cfg.dim, result.same_vector_scale
    );
    return result;
}

std::string make_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d%H%M%S");
    return out.str();
}

void write_i32(std::ofstream &out, int32_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_i16(std::ofstream &out, int16_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_i8(std::ofstream &out, int8_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_f32(std::ofstream &out, float value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_egev4(
    const Config &cfg,
    const std::vector<int16_t> &linear_values,
    const std::map<int, std::vector<float>> &phase_vectors,
    const std::map<int, PhaseResult> &phase_results,
    const std::string &output_egev4,
    bool is_cross_bank
) {
    fs::path output_path(output_egev4);
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
    write_i32(out, VERSION_LINEAR_FM_INT16_INT8);
    write_i32(out, N_PHASES);
    write_i32(out, N_PARAMS_PER_PHASE);
    write_i32(out, cfg.dim);

    for (int phase = 0; phase < N_PHASES; ++phase) {
        write_f32(out, static_cast<float>(1.0 / STEP));
    }
    for (int phase = 0; phase < N_PHASES; ++phase) {
        auto it = phase_results.find(phase);
        double vector_scale = 1.0;
        if (it != phase_results.end()) {
            vector_scale = is_cross_bank ? it->second.cross_vector_scale : it->second.same_vector_scale;
        }
        write_f32(out, static_cast<float>(vector_scale));
    }

    for (int phase = 0; phase < N_PHASES; ++phase) {
        auto vec_it = phase_vectors.find(phase);
        const std::vector<float> *vectors = vec_it == phase_vectors.end() ? nullptr : &vec_it->second;
        double vector_scale = 1.0;
        auto result_it = phase_results.find(phase);
        if (result_it != phase_results.end()) {
            vector_scale = is_cross_bank ? result_it->second.cross_vector_scale : result_it->second.same_vector_scale;
        }
        const size_t phase_base = static_cast<size_t>(phase) * N_PARAMS_PER_PHASE;
        for (int param_id = 0; param_id < N_PARAMS_PER_PHASE; ++param_id) {
            write_i16(out, clamp_value<int16_t>(linear_values[phase_base + param_id], -32767, 32767));
            for (int d = 0; d < cfg.dim; ++d) {
                int q_vec = 0;
                if (vectors != nullptr && vector_scale > 0.0) {
                    const size_t base = static_cast<size_t>(param_id) * cfg.dim;
                    q_vec = static_cast<int>(std::llround((*vectors)[base + d] / vector_scale));
                    q_vec = clamp_value(q_vec, -127, 127);
                }
                write_i8(out, static_cast<int8_t>(q_vec));
            }
        }
    }
}

void write_summary(
    const Config &cfg,
    const std::map<int, PhaseResult> &phase_results,
    uint64_t total_elapsed_ms
) {
    fs::path summary_path(cfg.summary);
    if (!summary_path.parent_path().empty()) {
        fs::create_directories(summary_path.parent_path());
    }
    std::ofstream out(summary_path);
    if (!out) {
        throw std::runtime_error("cannot write " + summary_path.string());
    }

    out << std::setprecision(10);
    out << "optimizer=current-linear-fixed joint-dual-type-FM streaming Adam\n";
    out << "fm_interaction_features=board-pattern features only\n";
    out << "fm_definition=linear + cross_weight * cross_type_FM + same_weight * same_type_FM\n";
    out << "cross_type_FM=pairs between different pattern types among the 64 board-pattern features\n";
    out << "same_type_FM=pairs among the four orientations of each pattern type\n";
    out << "linear_features=all current-model features including stone count\n";
    out << "comparison_rule=Only compare strength against candidates trained with the same train_ids, holdout_ids, data_root, and record caps.\n";
    out << "input_egev2=" << cfg.input_egev2 << "\n";
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
    out << "residual_clip=" << cfg.residual_clip << "\n";
    out << "loss=" << cfg.loss << "\n";
    out << "huber_delta=" << cfg.huber_delta << "\n";
    out << "l2=" << cfg.l2 << "\n";
    out << "weight_decay=" << cfg.weight_decay << "\n";
    out << "cross_weight=" << cfg.cross_weight << "\n";
    out << "same_weight=" << cfg.same_weight << "\n";
    out << "early_stop_patience=" << cfg.early_stop_patience << "\n";
    out << "seed=" << cfg.seed << "\n";
    out << "output_cross_egev4=" << cfg.output_cross_egev4 << "\n";
    out << "output_same_egev4=" << cfg.output_same_egev4 << "\n";
    out << "eval_spec=" << cfg.output_cross_egev4 << '@' << cfg.output_same_egev4
        << '@' << cfg.cross_weight << '@' << cfg.same_weight << "\n";
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
        out << "cross_vector_scale=" << result.cross_vector_scale << "\n";
        out << "cross_vector_max_abs=" << result.cross_vector_max_abs << "\n";
        out << "cross_nonzero_vector_values=" << result.cross_nonzero_vector_values << "\n";
        out << "same_vector_scale=" << result.same_vector_scale << "\n";
        out << "same_vector_max_abs=" << result.same_vector_max_abs << "\n";
        out << "same_nonzero_vector_values=" << result.same_nonzero_vector_values << "\n";
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
        const fs::path data_root(cfg.data_root);
        const std::vector<int16_t> linear_values = load_unzip_egev2(cfg.input_egev2);

        std::map<int, std::vector<float>> phase_cross_vectors;
        std::map<int, std::vector<float>> phase_same_vectors;
        std::map<int, PhaseResult> phase_results;
        for (int phase : cfg.phases) {
            std::vector<float> best_cross_vectors;
            std::vector<float> best_same_vectors;
            PhaseResult result = train_phase(cfg, data_root, phase, linear_values, best_cross_vectors, best_same_vectors);
            phase_cross_vectors[phase] = std::move(best_cross_vectors);
            phase_same_vectors[phase] = std::move(best_same_vectors);
            phase_results[phase] = result;
            const EpochRow &best_row = result.history[static_cast<size_t>(result.best_epoch)];
            std::cout << "phase " << phase
                      << " n_train " << result.n_train
                      << " n_holdout " << result.n_holdout
                      << " best_epoch " << result.best_epoch
                      << " best_holdout_MAE " << std::setprecision(8) << best_row.holdout.mae
                      << " cross_nonzero_vector_values " << result.cross_nonzero_vector_values
                      << " same_nonzero_vector_values " << result.same_nonzero_vector_values
                      << std::endl;
        }

        write_egev4(cfg, linear_values, phase_cross_vectors, phase_results, cfg.output_cross_egev4, true);
        write_egev4(cfg, linear_values, phase_same_vectors, phase_results, cfg.output_same_egev4, false);
        const uint64_t total_elapsed_ms = now_ms() - start_ms;
        write_summary(cfg, phase_results, total_elapsed_ms);
        std::cout << "elapsed_ms " << total_elapsed_ms << std::endl;
        std::cout << "wrote_cross " << cfg.output_cross_egev4 << std::endl;
        std::cout << "wrote_same " << cfg.output_same_egev4 << std::endl;
        std::cout << "summary " << cfg.summary << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
