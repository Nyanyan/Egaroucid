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
#include <limits>
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
constexpr uint64_t INDEXED_PHASE_RECORD_BYTES =
    sizeof(int16_t) + sizeof(int16_t) + sizeof(uint16_t) * ADJ_N_FEATURES + sizeof(int16_t);

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

struct TargetClipStats {
    uint64_t clipped = 0;
    float max_abs_before = 0.0f;
    float max_abs_after = 0.0f;
};

struct ScoreClipStats {
    uint64_t clipped = 0;
    int max_abs_before = 0;
    int max_abs_after = 0;
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

int stone_count_feature_start(const std::array<int, ADJ_N_FEATURES> &starts) {
    return starts[FM_N_PATTERN_FEATURES];
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

int apply_score_clip(int score, int score_clip, ScoreClipStats *stats) {
    stats->max_abs_before = std::max(stats->max_abs_before, std::abs(score));
    if (score_clip > 0) {
        const int clipped = std::clamp(score, -score_clip, score_clip);
        if (clipped != score) {
            ++stats->clipped;
            score = clipped;
        }
    }
    stats->max_abs_after = std::max(stats->max_abs_after, std::abs(score));
    return score;
}

void append_samples_from_raw_file(
    const std::string &file,
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    int n_fm_phases,
    size_t max_records,
    int score_clip,
    ScoreClipStats *score_clip_stats,
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
        const int teacher_score = apply_score_clip((int)rec.score, score_clip, score_clip_stats);
        sample.target = (float)teacher_score * ADJ_STEP - predict_linear(linear, starts, features, phase);
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
    int score_clip,
    ScoreClipStats *score_clip_stats,
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
        const int teacher_score = apply_score_clip((int)score, score_clip, score_clip_stats);
        sample.target = (float)teacher_score * ADJ_STEP - predict_linear(linear, starts, features, phase);
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
    size_t max_records,
    int score_clip,
    ScoreClipStats *score_clip_stats,
    int fixed_data_phase
) {
    std::vector<Sample> samples;
    if (fixed_data_phase >= 0) {
        uint64_t expected_records = 0;
        int bad_sized_files = 0;
        for (int i = start_file; i < start_file + n_files; ++i) {
            const std::string file = data_dir + "/" + std::to_string(i) + ".dat";
            const std::string indexed_file = data_dir + "/" + std::to_string(i) + "/teacher_0.dat";
            std::string target_file;
            if (std::filesystem::exists(file)) {
                target_file = file;
            } else if (std::filesystem::exists(indexed_file)) {
                target_file = indexed_file;
            } else {
                continue;
            }
            std::error_code ec;
            const uint64_t bytes = (uint64_t)std::filesystem::file_size(target_file, ec);
            if (ec) {
                continue;
            }
            if (bytes % INDEXED_PHASE_RECORD_BYTES != 0) {
                ++bad_sized_files;
            }
            expected_records += bytes / INDEXED_PHASE_RECORD_BYTES;
            if (max_records > 0 && expected_records >= max_records) {
                expected_records = max_records;
                break;
            }
        }
        if (expected_records > 0) {
            samples.reserve((size_t)expected_records);
            std::cerr << "reserve_samples " << expected_records
                      << " fixed_data_phase " << fixed_data_phase
                      << " bad_sized_files " << bad_sized_files << std::endl;
        }
    } else if (max_records > 0) {
        samples.reserve(max_records);
    }
    for (int i = start_file; i < start_file + n_files; ++i) {
        const std::string file = data_dir + "/" + std::to_string(i) + ".dat";
        const std::string indexed_file = data_dir + "/" + std::to_string(i) + "/teacher_0.dat";
        const bool raw_exists = std::filesystem::exists(file);
        const bool indexed_exists = std::filesystem::exists(indexed_file);
        const size_t before = samples.size();
        if (fixed_data_phase >= 0 && raw_exists) {
            append_samples_from_index_file(file, linear, starts, fixed_data_phase, n_fm_phases, max_records, score_clip, score_clip_stats, &samples);
        } else if (fixed_data_phase >= 0 && indexed_exists) {
            append_samples_from_index_file(indexed_file, linear, starts, fixed_data_phase, n_fm_phases, max_records, score_clip, score_clip_stats, &samples);
        } else if (raw_exists) {
            append_samples_from_raw_file(file, linear, starts, n_fm_phases, max_records, score_clip, score_clip_stats, &samples);
        } else if (indexed_exists) {
            append_samples_from_index_file(indexed_file, linear, starts, i, n_fm_phases, max_records, score_clip, score_clip_stats, &samples);
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
    size_t begin,
    size_t end,
    int dim,
    size_t limit,
    uint32_t active_pattern_mask
) {
    if (begin >= end || end > indices.size()) {
        return 0.0f;
    }
    const size_t n_available = end - begin;
    const size_t n = std::min(limit == 0 ? n_available : limit, n_available);
    double mae = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const Sample &sample = samples[indices[begin + i]];
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

uint64_t init_touched_fm_rows(
    std::vector<float> *vec,
    const std::array<int, FM_N_PATTERN_FEATURES> &offsets,
    uint64_t total_vectors,
    const std::vector<Sample> &samples,
    const std::vector<size_t> &indices,
    size_t begin,
    size_t end,
    int dim,
    uint32_t active_pattern_mask,
    float init_std,
    std::mt19937 *rng
) {
    std::normal_distribution<float> init_dist(0.0f, init_std);
    std::vector<uint8_t> touched((size_t)((uint64_t)vec->size() / dim), 0);
    uint64_t n_touched = 0;
    for (size_t k = begin; k < end; ++k) {
        const size_t idx = indices[k];
        const Sample &sample = samples[idx];
        const uint64_t phase_offset = (uint64_t)sample.fm_phase * total_vectors;
        for (int i = 0; i < FM_N_PATTERN_FEATURES; ++i) {
            if (!fm_feature_active(i, active_pattern_mask)) {
                continue;
            }
            if (sample.features[i] >= adj_eval_sizes[i >> 2]) {
                continue;
            }
            const uint64_t row = phase_offset + (uint64_t)offsets[i] + sample.features[i];
            if (touched[(size_t)row]) {
                continue;
            }
            touched[(size_t)row] = 1;
            ++n_touched;
            const uint64_t vec_offset = row * dim;
            for (int d = 0; d < dim; ++d) {
                (*vec)[(size_t)vec_offset + d] = init_dist(*rng);
            }
        }
    }
    return n_touched;
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
    uint32_t active_pattern_mask,
    uint64_t initialized_rows,
    bool center_phase_target,
    const std::array<int, ADJ_N_PHASES> &phase_target_corrections,
    float l2,
    float error_clip,
    float vector_clip,
    float init_std,
    float target_clip,
    const TargetClipStats &target_clip_stats,
    int score_clip,
    const ScoreClipStats &score_clip_stats,
    const std::string &data_dir,
    int start_file,
    int n_files,
    int fixed_data_phase,
    size_t n_samples
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
        summary << "center_phase_target " << (center_phase_target ? 1 : 0) << "\n";
        summary << "l2 " << l2 << "\n";
        summary << "error_clip " << error_clip << "\n";
        summary << "vector_clip " << vector_clip << "\n";
        summary << "init_std " << init_std << "\n";
        summary << "target_clip " << target_clip << "\n";
        summary << "target_clip_count " << target_clip_stats.clipped << "\n";
        summary << "target_max_abs_before_disc " << target_clip_stats.max_abs_before / ADJ_STEP << "\n";
        summary << "target_max_abs_after_disc " << target_clip_stats.max_abs_after / ADJ_STEP << "\n";
        summary << "score_clip " << score_clip << "\n";
        summary << "score_clip_count " << score_clip_stats.clipped << "\n";
        summary << "score_max_abs_before_disc " << score_clip_stats.max_abs_before << "\n";
        summary << "score_max_abs_after_disc " << score_clip_stats.max_abs_after << "\n";
        summary << "initialized_fm_rows " << initialized_rows << "\n";
        summary << "nonzero_quantized " << nonzero << "\n";
        summary << "max_abs_quantized " << max_abs << "\n";
        summary << "best_epoch " << best_epoch << "\n";
        summary << "best_val_mae_disc " << best_val_mae << "\n";
        summary << "data_dir " << data_dir << "\n";
        summary << "start_file " << start_file << "\n";
        summary << "n_files " << n_files << "\n";
        summary << "fixed_data_phase " << fixed_data_phase << "\n";
        summary << "samples " << n_samples << "\n";
        if (center_phase_target) {
            summary << "phase_target_corrections";
            for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
                summary << " " << phase_target_corrections[(size_t)phase];
            }
            summary << "\n";
        }
    }
    std::cerr << "wrote " << out_file << " nonzero_quantized " << nonzero << " max_abs_quantized " << max_abs << std::endl;
    return true;
}

void center_targets_by_train_phase_mean(
    std::vector<Sample> *samples,
    const std::vector<size_t> &indices,
    size_t begin,
    size_t end,
    const std::array<int, ADJ_N_FEATURES> &starts,
    std::vector<int16_t> *linear,
    std::array<int, ADJ_N_PHASES> *phase_target_corrections
) {
    std::array<double, ADJ_N_PHASES> phase_sum = {};
    std::array<uint64_t, ADJ_N_PHASES> phase_count = {};
    for (size_t k = begin; k < end; ++k) {
        const size_t idx = indices[k];
        const Sample &sample = (*samples)[idx];
        phase_sum[(size_t)sample.phase] += sample.target;
        ++phase_count[(size_t)sample.phase];
    }

    const int per_phase = linear_params_per_phase();
    const int stone_start = stone_count_feature_start(starts);
    for (int phase = 0; phase < ADJ_N_PHASES; ++phase) {
        if (phase_count[(size_t)phase] == 0) {
            continue;
        }
        const int correction = (int)std::lrint(phase_sum[(size_t)phase] / (double)phase_count[(size_t)phase]);
        (*phase_target_corrections)[(size_t)phase] = correction;
        if (correction == 0) {
            continue;
        }
        for (Sample &sample: *samples) {
            if (sample.phase == phase) {
                sample.target -= (float)correction;
            }
        }
        const size_t base = (size_t)phase * per_phase + stone_start;
        for (int n_player = 0; n_player < ADJ_MAX_STONE_NUM; ++n_player) {
            int value = (*linear)[base + (size_t)n_player] + correction;
            value = std::clamp(value, (int)std::numeric_limits<int16_t>::min(), (int)std::numeric_limits<int16_t>::max());
            (*linear)[base + (size_t)n_player] = (int16_t)value;
        }
        std::cerr << "center_phase_target phase " << phase
                  << " count " << phase_count[(size_t)phase]
                  << " correction " << correction << std::endl;
    }
}

TargetClipStats clip_sample_targets(std::vector<Sample> *samples, float target_clip) {
    TargetClipStats stats;
    for (Sample &sample: *samples) {
        const float before_abs = std::fabs(sample.target);
        stats.max_abs_before = std::max(stats.max_abs_before, before_abs);
        if (target_clip > 0.0f) {
            const float clipped = std::clamp(sample.target, -target_clip, target_clip);
            if (clipped != sample.target) {
                ++stats.clipped;
                sample.target = clipped;
            }
        }
        stats.max_abs_after = std::max(stats.max_abs_after, std::fabs(sample.target));
    }
    return stats;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr
            << "usage: eval_optimizer_fm [base_eval.egev2] [board_data_dir] [start_file] [n_files] [out_file] "
            << "[dim=8] [fm_phases=1] [epochs=3] [lr=0.0002] [max_records=100000] [scale=16] [seed=20260723] "
            << "[active_pattern_mask=0] [center_phase_target=0] [l2=0.00001] [error_clip=4096] [vector_clip=7.5] "
            << "[init_std=0.02] [target_clip=0] [score_clip=0] [fixed_data_phase=-1]\n";
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
    const bool center_phase_target = argc >= 15 ? std::atoi(argv[14]) != 0 : false;
    const float l2 = argc >= 16 ? (float)std::atof(argv[15]) : 0.00001f;
    const float error_clip = argc >= 17 ? (float)std::atof(argv[16]) : 4096.0f;
    const float vector_clip = argc >= 18 ? (float)std::atof(argv[17]) : 7.5f;
    const float init_std = argc >= 19 ? (float)std::atof(argv[18]) : 0.02f;
    const float target_clip = argc >= 20 ? (float)std::atof(argv[19]) : 0.0f;
    const int score_clip = argc >= 21 ? std::atoi(argv[20]) : 0;
    const int fixed_data_phase = argc >= 22 ? std::atoi(argv[21]) : -1;

    if (dim <= 0 || dim > 64 || n_fm_phases <= 0 || n_fm_phases > ADJ_N_PHASES || epochs < 0 || scale <= 0 ||
        (active_pattern_mask & 0xFFFF0000U) != 0 || l2 < 0.0f || error_clip <= 0.0f || vector_clip <= 0.0f ||
        init_std <= 0.0f || target_clip < 0.0f || score_clip < 0 || score_clip > HW2 ||
        fixed_data_phase < -1 || fixed_data_phase >= ADJ_N_PHASES) {
        std::cerr << "[ERROR] invalid dim/fm_phases/epochs/scale/active_pattern_mask/l2/error_clip/vector_clip/init_std/target_clip/score_clip/fixed_data_phase" << std::endl;
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
    std::vector<int16_t> output_linear = linear;
    const auto starts = make_linear_starts();
    uint64_t total_vectors = 0;
    const auto offsets = make_fm_feature_offsets(&total_vectors);
    if (total_vectors == 0) {
        std::cerr << "[ERROR] invalid FM vector layout" << std::endl;
        return 1;
    }

    ScoreClipStats score_clip_stats;
    auto samples = load_samples(data_dir, start_file, n_files, linear, starts, n_fm_phases, max_records, score_clip, &score_clip_stats, fixed_data_phase);
    if (samples.empty()) {
        std::cerr << "[ERROR] no samples loaded" << std::endl;
        return 1;
    }
    std::cerr << "samples " << samples.size() << " dim " << dim
              << " fm_phases " << n_fm_phases << " epochs " << epochs
              << " lr " << lr << " scale " << scale
              << " active_pattern_mask 0x" << std::hex << active_pattern_mask << std::dec
              << " l2 " << l2
              << " error_clip " << error_clip
              << " vector_clip " << vector_clip
              << " init_std " << init_std
              << " target_clip " << target_clip
              << " score_clip " << score_clip
              << " fixed_data_phase " << fixed_data_phase
              << std::endl;
    if (score_clip > 0) {
        std::cerr << "score_clip " << score_clip
                  << " clipped " << score_clip_stats.clipped
                  << " max_abs_before_disc " << score_clip_stats.max_abs_before
                  << " max_abs_after_disc " << score_clip_stats.max_abs_after
                  << std::endl;
    }

    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    const size_t n_val = std::min(samples.size(), std::max<size_t>(1, samples.size() / 10));
    const size_t val_begin = 0;
    const size_t val_end = n_val;
    size_t train_begin = n_val;
    size_t train_end = samples.size();
    if (train_begin >= train_end) {
        train_begin = val_begin;
        train_end = val_end;
    }

    std::array<int, ADJ_N_PHASES> phase_target_corrections = {};
    if (center_phase_target) {
        center_targets_by_train_phase_mean(&samples, indices, train_begin, train_end, starts, &output_linear, &phase_target_corrections);
    }
    const TargetClipStats target_clip_stats = clip_sample_targets(&samples, target_clip);
    if (target_clip > 0.0f) {
        std::cerr << "target_clip " << target_clip
                  << " clipped " << target_clip_stats.clipped
                  << " max_abs_before_disc " << target_clip_stats.max_abs_before / ADJ_STEP
                  << " max_abs_after_disc " << target_clip_stats.max_abs_after / ADJ_STEP
                  << std::endl;
    }

    std::vector<float> vec((size_t)n_fm_phases * total_vectors * dim, 0.0f);
    const uint64_t initialized_rows = init_touched_fm_rows(
        &vec, offsets, total_vectors, samples, indices, train_begin, train_end, dim, active_pattern_mask, init_std, &rng
    );
    std::cerr << "initialized_fm_rows " << initialized_rows << std::endl;

    float best_val_mae = calc_mae(vec, offsets, total_vectors, samples, indices, val_begin, val_end, dim, 20000, active_pattern_mask);
    int best_epoch = 0;
    std::vector<float> best_vec = vec;
    std::cerr << "initial train_mae "
              << calc_mae(vec, offsets, total_vectors, samples, indices, train_begin, train_end, dim, 20000, active_pattern_mask)
              << " val_mae " << best_val_mae << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin() + (std::ptrdiff_t)train_begin, indices.begin() + (std::ptrdiff_t)train_end, rng);
        const uint64_t start_ms = fm_optimizer_tim();
        for (size_t k = train_begin; k < train_end; ++k) {
            const size_t idx = indices[k];
            train_one_sample(&vec, offsets, total_vectors, samples[idx], dim, lr, l2, error_clip, vector_clip, active_pattern_mask);
        }
        const uint64_t elapsed = fm_optimizer_tim() - start_ms;
        const float train_mae = calc_mae(vec, offsets, total_vectors, samples, indices, train_begin, train_end, dim, 20000, active_pattern_mask);
        const float val_mae = calc_mae(vec, offsets, total_vectors, samples, indices, val_begin, val_end, dim, 20000, active_pattern_mask);
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

    if (!write_fm_file(out_file, output_linear, best_vec, total_vectors, n_fm_phases, dim, scale, best_epoch, best_val_mae, active_pattern_mask, initialized_rows, center_phase_target, phase_target_corrections, l2, error_clip, vector_clip, init_std, target_clip, target_clip_stats, score_clip, score_clip_stats, data_dir, start_file, n_files, fixed_data_phase, samples.size())) {
        return 1;
    }
    return 0;
}
