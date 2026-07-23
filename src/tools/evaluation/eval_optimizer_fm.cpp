/*
    Egaroucid Project

    @file eval_optimizer_fm.cpp
        CPU Factorization Machine optimizer for the current evaluation layout
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Keep this tool tied to the evaluation layout used by src/engine/evaluate*.hpp.
#include "evaluation_definition_20241125_1_7_5.hpp"

constexpr char FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr uint32_t FM_FILE_VERSION = 1;
constexpr int N_ZEROS_PLUS_LOCAL = 1 << 12;
constexpr int FM_N_PATTERN_FEATURES = ADJ_N_FEATURES - 1;

struct RawRecord {
    Board board;
    int8_t player;
    int8_t policy;
    int8_t score;
};

struct Sample {
    std::array<uint16_t, FM_N_PATTERN_FEATURES> features;
    float target;
    uint16_t phase;
    uint16_t fm_phase;
};

uint64_t fm_optimizer_tim() {
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

std::vector<int16_t> load_unzip_egev2_local(const std::string &file) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        std::cerr << "[ERROR] can't open base eval " << file << std::endl;
        return {};
    }
    int32_t n_zipped = 0;
    if (!read_scalar(fp, &n_zipped) || n_zipped <= 0) {
        std::cerr << "[ERROR] base eval header broken " << file << std::endl;
        fclose(fp);
        return {};
    }
    std::vector<int16_t> zipped((size_t)n_zipped);
    if (fread(zipped.data(), sizeof(int16_t), zipped.size(), fp) < zipped.size()) {
        std::cerr << "[ERROR] base eval payload broken " << file << std::endl;
        fclose(fp);
        return {};
    }
    fclose(fp);

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
        starts[i] = start;
    }
    return starts;
}

std::array<int, FM_N_PATTERN_FEATURES> make_fm_feature_offsets(uint64_t *total_vectors) {
    std::array<int, FM_N_PATTERN_FEATURES> offsets = {};
    int offset = 0;
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        offsets[i] = offset;
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

float predict_linear(
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    const uint16_t features[ADJ_N_FEATURES],
    int phase
) {
    const int per_phase = linear_params_per_phase();
    const int phase_offset = phase * per_phase;
    int res = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        res += linear[(size_t)phase_offset + starts[i] + features[i]];
    }
    return (float)res;
}

uint16_t fm_phase_from_phase(int phase, int n_fm_phases) {
    if (n_fm_phases <= 1) {
        return 0;
    }
    return (uint16_t)std::min(n_fm_phases - 1, phase * n_fm_phases / ADJ_N_PHASES);
}

bool fm_feature_active(int feature_idx, uint32_t active_pattern_mask) {
    return active_pattern_mask == 0 || (active_pattern_mask & (1U << (feature_idx >> 2))) != 0;
}

void append_samples_from_raw_file(
    const std::string &file,
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    int n_fm_phases,
    size_t max_records,
    std::vector<Sample> *samples
) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        std::cerr << "[WARN] can't open data " << file << std::endl;
        return;
    }
    RawRecord rec;
    uint16_t features[ADJ_N_FEATURES];
    while (max_records == 0 || samples->size() < max_records) {
        if (fread(&rec.board.player, 8, 1, fp) < 1) {
            break;
        }
        if (fread(&rec.board.opponent, 8, 1, fp) < 1 ||
            fread(&rec.player, 1, 1, fp) < 1 ||
            fread(&rec.policy, 1, 1, fp) < 1 ||
            fread(&rec.score, 1, 1, fp) < 1) {
            break;
        }
        const int phase = calc_phase(&rec.board, rec.player);
        if (phase < 0 || phase >= ADJ_N_PHASES) {
            continue;
        }
        adj_calc_features(&rec.board, features);
        Sample sample;
        for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
            sample.features[i] = features[i];
        }
        sample.phase = (uint16_t)phase;
        sample.fm_phase = fm_phase_from_phase(phase, n_fm_phases);
        sample.target = (float)rec.score * ADJ_STEP - predict_linear(linear, starts, features, phase);
        samples->emplace_back(sample);
    }
    fclose(fp);
}

void append_samples_from_index_file(
    const std::string &file,
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    int phase,
    int n_fm_phases,
    size_t max_records,
    std::vector<Sample> *samples
) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        std::cerr << "[WARN] can't open indexed data " << file << std::endl;
        return;
    }
    int16_t n_discs = 0;
    int16_t player = 0;
    int16_t score = 0;
    uint16_t features[ADJ_N_FEATURES];
    while (max_records == 0 || samples->size() < max_records) {
        if (fread(&n_discs, sizeof(int16_t), 1, fp) < 1) {
            break;
        }
        if (fread(&player, sizeof(int16_t), 1, fp) < 1 ||
            fread(features, sizeof(uint16_t), ADJ_N_FEATURES, fp) < ADJ_N_FEATURES ||
            fread(&score, sizeof(int16_t), 1, fp) < 1) {
            break;
        }
        if (phase < 0 || phase >= ADJ_N_PHASES || n_discs < 0 || n_discs > HW2 || player < 0 || player > 1) {
            continue;
        }
        Sample sample;
        for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
            sample.features[i] = features[i];
        }
        sample.phase = (uint16_t)phase;
        sample.fm_phase = fm_phase_from_phase(phase, n_fm_phases);
        sample.target = (float)score * ADJ_STEP - predict_linear(linear, starts, features, phase);
        samples->emplace_back(sample);
    }
    fclose(fp);
}

std::vector<Sample> load_samples(
    const std::string &data_dir,
    int start_file,
    int n_files,
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    int n_fm_phases,
    size_t max_records
) {
    std::vector<Sample> samples;
    if (max_records > 0) {
        samples.reserve(max_records);
    }
    for (int i = start_file; i < start_file + n_files; ++i) {
        const std::string file = data_dir + "/" + std::to_string(i) + ".dat";
        const std::string indexed_file = data_dir + "/" + std::to_string(i) + "/teacher_0.dat";
        const bool raw_exists = std::filesystem::exists(file);
        const bool indexed_exists = std::filesystem::exists(indexed_file);
        const size_t before = samples.size();
        if (raw_exists) {
            append_samples_from_raw_file(file, linear, starts, n_fm_phases, max_records, &samples);
        } else if (indexed_exists) {
            append_samples_from_index_file(indexed_file, linear, starts, i, n_fm_phases, max_records, &samples);
        } else {
            std::cerr << "[WARN] can't find data " << file << " or " << indexed_file << std::endl;
        }
        if (samples.size() > before) {
            std::cerr << (raw_exists ? file : indexed_file) << " samples " << samples.size() << std::endl;
        }
        if (max_records > 0 && samples.size() >= max_records) {
            break;
        }
    }
    return samples;
}

float predict_fm(
    const std::vector<float> &vec,
    const std::array<int, FM_N_PATTERN_FEATURES> &offsets,
    uint64_t total_vectors,
    const Sample &sample,
    int dim,
    uint32_t active_pattern_mask
) {
    float sum[64] = {};
    float square_sum[64] = {};
    const uint64_t phase_offset = (uint64_t)sample.fm_phase * total_vectors * dim;
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        if (!fm_feature_active(i, active_pattern_mask)) {
            continue;
        }
        const uint64_t row = phase_offset + (uint64_t)(offsets[i] + sample.features[i]) * dim;
        for (int d = 0; d < dim; ++d) {
            const float x = vec[(size_t)row + d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }
    float res = 0.0f;
    for (int d = 0; d < dim; ++d) {
        res += 0.5f * (sum[d] * sum[d] - square_sum[d]);
    }
    return res;
}

float calc_mae(
    const std::vector<float> &vec,
    const std::array<int, FM_N_PATTERN_FEATURES> &offsets,
    uint64_t total_vectors,
    const std::vector<Sample> &samples,
    const std::vector<size_t> &indices,
    int dim,
    size_t limit,
    uint32_t active_pattern_mask
) {
    if (indices.empty()) {
        return 0.0f;
    }
    const size_t n = std::min(limit == 0 ? indices.size() : limit, indices.size());
    double mae = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const Sample &sample = samples[indices[i]];
        mae += std::fabs(sample.target - predict_fm(vec, offsets, total_vectors, sample, dim, active_pattern_mask)) / ADJ_STEP;
    }
    return (float)(mae / n);
}

void train_one_sample(
    std::vector<float> *vec,
    const std::array<int, FM_N_PATTERN_FEATURES> &offsets,
    uint64_t total_vectors,
    const Sample &sample,
    int dim,
    float lr,
    float l2,
    float error_clip,
    float vector_clip,
    uint32_t active_pattern_mask
) {
    float sum[64] = {};
    float square_sum[64] = {};
    const uint64_t phase_offset = (uint64_t)sample.fm_phase * total_vectors * dim;
    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        if (!fm_feature_active(i, active_pattern_mask)) {
            continue;
        }
        const uint64_t row = phase_offset + (uint64_t)(offsets[i] + sample.features[i]) * dim;
        for (int d = 0; d < dim; ++d) {
            const float x = (*vec)[(size_t)row + d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }

    float pred = 0.0f;
    for (int d = 0; d < dim; ++d) {
        pred += 0.5f * (sum[d] * sum[d] - square_sum[d]);
    }
    float err = sample.target - pred;
    err = std::clamp(err, -error_clip, error_clip);

    for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
        if (!fm_feature_active(i, active_pattern_mask)) {
            continue;
        }
        const uint64_t row = phase_offset + (uint64_t)(offsets[i] + sample.features[i]) * dim;
        for (int d = 0; d < dim; ++d) {
            float &x = (*vec)[(size_t)row + d];
            const float grad = sum[d] - x;
            x += lr * (err * grad - l2 * x);
            x = std::clamp(x, -vector_clip, vector_clip);
        }
    }
}

bool write_fm_file(
    const std::string &out_file,
    const std::vector<int16_t> &linear,
    const std::vector<float> &vec,
    uint64_t total_vectors,
    int n_fm_phases,
    int dim,
    int scale,
    int best_epoch,
    float best_val_mae,
    uint32_t active_pattern_mask
) {
    std::filesystem::path out_path(out_file);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << std::endl;
        return false;
    }
    const uint32_t version = FM_FILE_VERSION;
    const uint32_t n_phases = ADJ_N_PHASES;
    const uint32_t linear_per_phase = (uint32_t)linear_params_per_phase();
    const uint32_t fm_phases_u32 = (uint32_t)n_fm_phases;
    const uint32_t n_features = FM_N_PATTERN_FEATURES;
    const uint32_t dim_u32 = (uint32_t)dim;
    const int32_t scale_i32 = scale;
    const uint32_t flags = active_pattern_mask & 0xFFFFU;
    const uint64_t linear_count = (uint64_t)linear.size();
    const uint64_t fm_count = (uint64_t)n_fm_phases * total_vectors * dim;

    out.write(FM_FILE_MAGIC, sizeof(FM_FILE_MAGIC));
    write_scalar(out, version);
    write_scalar(out, n_phases);
    write_scalar(out, linear_per_phase);
    write_scalar(out, fm_phases_u32);
    write_scalar(out, n_features);
    write_scalar(out, dim_u32);
    write_scalar(out, scale_i32);
    write_scalar(out, flags);
    write_scalar(out, linear_count);
    write_scalar(out, fm_count);
    out.write((const char*)linear.data(), sizeof(int16_t) * linear.size());

    int nonzero = 0;
    int max_abs = 0;
    for (const float x: vec) {
        int q = (int)std::lrint(x * scale);
        q = std::clamp(q, -127, 127);
        nonzero += q != 0;
        max_abs = std::max(max_abs, std::abs(q));
        const int8_t q8 = (int8_t)q;
        out.write((const char*)&q8, sizeof(int8_t));
    }
    out.close();

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "created_at_ms " << fm_optimizer_tim() << "\n";
        summary << "definition " << EVAL_DEFINITION_NAME << "\n";
        summary << "linear_params " << linear.size() << "\n";
        summary << "fm_phases " << n_fm_phases << "\n";
        summary << "dim " << dim << "\n";
        summary << "scale " << scale << "\n";
        summary << "total_vectors_per_fm_phase " << total_vectors << "\n";
        summary << "fm_values " << fm_count << "\n";
        summary << "active_pattern_mask 0x" << std::hex << flags << std::dec << "\n";
        summary << "nonzero_quantized " << nonzero << "\n";
        summary << "max_abs_quantized " << max_abs << "\n";
        summary << "best_epoch " << best_epoch << "\n";
        summary << "best_val_mae_disc " << best_val_mae << "\n";
    }
    std::cerr << "wrote " << out_file << " nonzero_quantized " << nonzero << " max_abs_quantized " << max_abs << std::endl;
    return true;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr
            << "usage: eval_optimizer_fm [base_eval.egev2] [board_data_dir] [start_file] [n_files] [out_file] "
            << "[dim=8] [fm_phases=1] [epochs=3] [lr=0.0002] [max_records=100000] [scale=16] [seed=20260723] [active_pattern_mask=0]\n";
        return 1;
    }
    const std::string base_eval = argv[1];
    const std::string data_dir = argv[2];
    const int start_file = std::atoi(argv[3]);
    const int n_files = std::atoi(argv[4]);
    const std::string out_file = argv[5];
    const int dim = argc >= 7 ? std::atoi(argv[6]) : 8;
    const int n_fm_phases = argc >= 8 ? std::atoi(argv[7]) : 1;
    const int epochs = argc >= 9 ? std::atoi(argv[8]) : 3;
    const float lr = argc >= 10 ? (float)std::atof(argv[9]) : 0.0002f;
    const size_t max_records = argc >= 11 ? (size_t)std::strtoull(argv[10], nullptr, 10) : 100000;
    const int scale = argc >= 12 ? std::atoi(argv[11]) : 16;
    const uint32_t seed = argc >= 13 ? (uint32_t)std::strtoul(argv[12], nullptr, 10) : 20260723U;
    const uint32_t active_pattern_mask = argc >= 14 ? (uint32_t)std::strtoul(argv[13], nullptr, 0) : 0U;
    const float l2 = 0.00001f;
    const float error_clip = 4096.0f;
    const float vector_clip = 7.5f;

    if (dim <= 0 || dim > 64 || n_fm_phases <= 0 || n_fm_phases > ADJ_N_PHASES || epochs < 0 || scale <= 0 || (active_pattern_mask & 0xFFFF0000U) != 0) {
        std::cerr << "[ERROR] invalid dim/fm_phases/epochs/scale/active_pattern_mask" << std::endl;
        return 1;
    }

    evaluation_definition_init();
    const auto linear = load_unzip_egev2_local(base_eval);
    const int expected_linear = ADJ_N_PHASES * linear_params_per_phase();
    if ((int)linear.size() != expected_linear) {
        std::cerr << "[ERROR] invalid base eval element count found " << linear.size()
                  << " expected " << expected_linear << std::endl;
        return 1;
    }
    const auto starts = make_linear_starts();
    uint64_t total_vectors = 0;
    const auto offsets = make_fm_feature_offsets(&total_vectors);
    if (total_vectors == 0) {
        std::cerr << "[ERROR] invalid FM vector layout" << std::endl;
        return 1;
    }

    auto samples = load_samples(data_dir, start_file, n_files, linear, starts, n_fm_phases, max_records);
    if (samples.empty()) {
        std::cerr << "[ERROR] no samples loaded" << std::endl;
        return 1;
    }
    std::cerr << "samples " << samples.size() << " dim " << dim
              << " fm_phases " << n_fm_phases << " epochs " << epochs
              << " lr " << lr << " scale " << scale
              << " active_pattern_mask 0x" << std::hex << active_pattern_mask << std::dec << std::endl;

    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    const size_t n_val = std::max<size_t>(1, samples.size() / 10);
    std::vector<size_t> val_indices(indices.begin(), indices.begin() + n_val);
    std::vector<size_t> train_indices(indices.begin() + n_val, indices.end());
    if (train_indices.empty()) {
        train_indices = val_indices;
    }

    std::normal_distribution<float> init_dist(0.0f, 0.02f);
    std::vector<float> vec((size_t)n_fm_phases * total_vectors * dim, 0.0f);
    for (int fm_phase = 0; fm_phase < n_fm_phases; ++fm_phase) {
        const uint64_t phase_offset = (uint64_t)fm_phase * total_vectors * dim;
        for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
            if (!fm_feature_active(i, active_pattern_mask)) {
                continue;
            }
            const uint64_t n_rows = (uint64_t)adj_eval_sizes[i >> 2];
            const uint64_t row_offset = phase_offset + (uint64_t)offsets[i] * dim;
            for (uint64_t row = 0; row < n_rows; ++row) {
                for (int d = 0; d < dim; ++d) {
                    vec[(size_t)(row_offset + row * dim + d)] = init_dist(rng);
                }
            }
        }
    }

    float best_val_mae = calc_mae(vec, offsets, total_vectors, samples, val_indices, dim, 20000, active_pattern_mask);
    int best_epoch = 0;
    std::vector<float> best_vec = vec;
    std::cerr << "initial train_mae "
              << calc_mae(vec, offsets, total_vectors, samples, train_indices, dim, 20000, active_pattern_mask)
              << " val_mae " << best_val_mae << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        const uint64_t start_ms = fm_optimizer_tim();
        for (const size_t idx: train_indices) {
            train_one_sample(&vec, offsets, total_vectors, samples[idx], dim, lr, l2, error_clip, vector_clip, active_pattern_mask);
        }
        const uint64_t elapsed = fm_optimizer_tim() - start_ms;
        const float train_mae = calc_mae(vec, offsets, total_vectors, samples, train_indices, dim, 20000, active_pattern_mask);
        const float val_mae = calc_mae(vec, offsets, total_vectors, samples, val_indices, dim, 20000, active_pattern_mask);
        if (val_mae < best_val_mae) {
            best_val_mae = val_mae;
            best_epoch = epoch + 1;
            best_vec = vec;
        }
        std::cerr << "epoch " << (epoch + 1)
                  << " elapsed_ms " << elapsed
                  << " train_mae " << train_mae
                  << " val_mae " << val_mae
                  << " best_epoch " << best_epoch
                  << " best_val_mae " << best_val_mae
                  << std::endl;
    }

    if (!write_fm_file(out_file, linear, best_vec, total_vectors, n_fm_phases, dim, scale, best_epoch, best_val_mae, active_pattern_mask)) {
        return 1;
    }
    return 0;
}
