/*
    Egaroucid Project

    @file center_egevfm_phase_mean.cpp
        Estimate FM mean contribution from indexed data and compensate phase bias
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "./../../../engine/evaluate.hpp"

constexpr uint32_t EVAL_FM_FILE_VERSION_LOCAL = 1;
constexpr size_t EVAL_FM_HEADER_SIZE = 56;

template<typename T>
bool read_scalar(const std::vector<char> &data, size_t *offset, T *value) {
    if (*offset + sizeof(T) > data.size()) {
        return false;
    }
    std::memcpy(value, data.data() + *offset, sizeof(T));
    *offset += sizeof(T);
    return true;
}

template<typename T>
bool write_scalar(std::vector<char> *data, size_t offset, const T value) {
    if (offset + sizeof(T) > data->size()) {
        return false;
    }
    std::memcpy(data->data() + offset, &value, sizeof(T));
    return true;
}

struct EgevfmFile {
    std::vector<char> bytes;
    uint32_t version = 0;
    uint32_t n_phases = 0;
    uint32_t linear_params_per_phase = 0;
    uint32_t n_fm_phases = 0;
    uint32_t n_features = 0;
    uint32_t dim = 0;
    int32_t scale = 0;
    uint32_t flags = 0;
    uint64_t linear_count = 0;
    uint64_t fm_count = 0;
    size_t linear_offset = EVAL_FM_HEADER_SIZE;
    size_t fm_offset = 0;
};

bool load_egevfm(const std::string &file, EgevfmFile *res) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << file << "\n";
        return false;
    }
    res->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (res->bytes.size() < EVAL_FM_HEADER_SIZE ||
        std::memcmp(res->bytes.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        std::cerr << "[ERROR] input is not egevfm v1: " << file << "\n";
        return false;
    }
    size_t offset = sizeof(EVAL_FM_FILE_MAGIC);
    bool ok =
        read_scalar(res->bytes, &offset, &res->version) &&
        read_scalar(res->bytes, &offset, &res->n_phases) &&
        read_scalar(res->bytes, &offset, &res->linear_params_per_phase) &&
        read_scalar(res->bytes, &offset, &res->n_fm_phases) &&
        read_scalar(res->bytes, &offset, &res->n_features) &&
        read_scalar(res->bytes, &offset, &res->dim) &&
        read_scalar(res->bytes, &offset, &res->scale) &&
        read_scalar(res->bytes, &offset, &res->flags) &&
        read_scalar(res->bytes, &offset, &res->linear_count) &&
        read_scalar(res->bytes, &offset, &res->fm_count);
    if (!ok || offset != EVAL_FM_HEADER_SIZE) {
        std::cerr << "[ERROR] broken egevfm header " << file << "\n";
        return false;
    }
    res->linear_offset = EVAL_FM_HEADER_SIZE;
    res->fm_offset = res->linear_offset + (size_t)res->linear_count * sizeof(int16_t);
    const size_t expected_size = res->fm_offset + (size_t)res->fm_count * sizeof(int8_t);
    if (res->version != EVAL_FM_FILE_VERSION_LOCAL ||
        res->n_phases != N_PHASES ||
        res->linear_params_per_phase != N_PATTERN_PARAMS_RAW + MAX_STONE_NUM ||
        res->n_features != N_PATTERN_FEATURES ||
        res->dim == 0 ||
        res->scale <= 0 ||
        expected_size != res->bytes.size()) {
        std::cerr << "[ERROR] unsupported egevfm layout " << file << "\n";
        return false;
    }
    return true;
}

std::array<uint64_t, N_PATTERN_FEATURES> make_feature_offsets(uint64_t *total_vectors) {
    std::array<uint64_t, N_PATTERN_FEATURES> offsets = {};
    uint64_t offset = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        offsets[i] = offset;
        offset += (uint64_t)pow3[pattern_sizes[i >> 2]];
    }
    *total_vectors = offset;
    return offsets;
}

bool fm_feature_active(const int feature_idx, const uint32_t active_pattern_mask) {
    return active_pattern_mask == 0 || (active_pattern_mask & (1U << (feature_idx >> 2))) != 0;
}

uint32_t fm_phase_from_phase(const int phase, const int n_fm_phases) {
    if (n_fm_phases <= 1) {
        return 0;
    }
    return (uint32_t)std::min(n_fm_phases - 1, phase * n_fm_phases / N_PHASES);
}

int finalize_fm_score(const int64_t diff, const int scale) {
    const int64_t denom = 2LL * scale * scale;
    if (diff >= 0) {
        return (int)((diff + denom / 2) / denom);
    }
    return -(int)((-diff + denom / 2) / denom);
}

int calc_fm_score(
    const EgevfmFile &fm,
    const std::array<uint64_t, N_PATTERN_FEATURES> &feature_offsets,
    const uint64_t total_vectors,
    const int phase,
    const std::vector<uint16_t> &features
) {
    std::array<int32_t, EVAL_FM_MAX_DIM> sum = {};
    std::array<int32_t, EVAL_FM_MAX_DIM> square_sum = {};
    const uint32_t fm_phase = fm_phase_from_phase(phase, (int)fm.n_fm_phases);
    const uint64_t phase_offset = (uint64_t)fm_phase * total_vectors * fm.dim;
    const int8_t *vectors = (const int8_t*)(fm.bytes.data() + fm.fm_offset);
    const uint32_t active_pattern_mask = fm.flags & 0xFFFFU;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        if (!fm_feature_active(i, active_pattern_mask)) {
            continue;
        }
        const uint16_t feature = features[(size_t)i];
        if (feature >= pow3[pattern_sizes[i >> 2]]) {
            return 0;
        }
        const uint64_t row = phase_offset + (feature_offsets[i] + feature) * fm.dim;
        for (uint32_t d = 0; d < fm.dim; ++d) {
            const int32_t x = vectors[(size_t)row + d];
            sum[d] += x;
            square_sum[d] += x * x;
        }
    }
    int64_t diff = 0;
    for (uint32_t d = 0; d < fm.dim; ++d) {
        diff += (int64_t)sum[d] * sum[d] - square_sum[d];
    }
    return finalize_fm_score(diff, fm.scale);
}

bool estimate_phase_means(
    const EgevfmFile &fm,
    const std::string &data_dir,
    const int start_phase,
    const int n_phase_files,
    const uint64_t max_records,
    const int indexed_feature_count,
    std::array<int64_t, N_PHASES> *phase_sum,
    std::array<uint64_t, N_PHASES> *phase_count
) {
    uint64_t total_vectors = 0;
    const auto feature_offsets = make_feature_offsets(&total_vectors);
    if (total_vectors * fm.n_fm_phases * fm.dim != fm.fm_count) {
        std::cerr << "[ERROR] FM vector count mismatch\n";
        return false;
    }
    uint64_t total_records = 0;
    for (int phase = start_phase; phase < start_phase + n_phase_files; ++phase) {
        const std::string file = data_dir + "/" + std::to_string(phase) + "/teacher_0.dat";
        FILE *fp = nullptr;
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "[WARN] can't open indexed data " << file << "\n";
            continue;
        }
        int16_t n_discs = 0;
        int16_t player = 0;
        int16_t score = 0;
        std::vector<uint16_t> features((size_t)indexed_feature_count);
        uint64_t file_records = 0;
        while (max_records == 0 || total_records < max_records) {
            if (fread(&n_discs, sizeof(int16_t), 1, fp) < 1) {
                break;
            }
            if (fread(&player, sizeof(int16_t), 1, fp) < 1 ||
                fread(features.data(), sizeof(uint16_t), (size_t)indexed_feature_count, fp) < (size_t)indexed_feature_count ||
                fread(&score, sizeof(int16_t), 1, fp) < 1) {
                break;
            }
            if (phase < 0 || phase >= N_PHASES || n_discs < 0 || n_discs > HW2 || player < 0 || player > 1) {
                continue;
            }
            const int fm_score = calc_fm_score(fm, feature_offsets, total_vectors, phase, features);
            (*phase_sum)[(size_t)phase] += fm_score;
            ++(*phase_count)[(size_t)phase];
            ++file_records;
            ++total_records;
        }
        fclose(fp);
        std::cerr << file << " records " << file_records << " total " << total_records << "\n";
        if (max_records > 0 && total_records >= max_records) {
            break;
        }
    }
    return total_records > 0;
}

bool apply_phase_correction(
    EgevfmFile *fm,
    const std::array<int64_t, N_PHASES> &phase_sum,
    const std::array<uint64_t, N_PHASES> &phase_count,
    const double shrink,
    std::array<int, N_PHASES> *corrections
) {
    int16_t *linear = (int16_t*)(fm->bytes.data() + fm->linear_offset);
    bool changed = false;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (phase_count[(size_t)phase] == 0) {
            continue;
        }
        const double mean = (double)phase_sum[(size_t)phase] / (double)phase_count[(size_t)phase];
        const int correction = (int)std::lrint(-mean * shrink);
        (*corrections)[(size_t)phase] = correction;
        if (correction == 0) {
            continue;
        }
        const size_t eval_num_offset = (size_t)phase * fm->linear_params_per_phase + N_PATTERN_PARAMS_RAW;
        for (int n_player = 0; n_player < MAX_STONE_NUM; ++n_player) {
            int value = linear[eval_num_offset + (size_t)n_player] + correction;
            value = std::clamp(value, (int)std::numeric_limits<int16_t>::min(), (int)std::numeric_limits<int16_t>::max());
            linear[eval_num_offset + (size_t)n_player] = (int16_t)value;
        }
        changed = true;
    }
    return changed;
}

bool write_file(const std::string &file, const std::vector<char> &data) {
    std::filesystem::path path(file);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << file << "\n";
        return false;
    }
    out.write(data.data(), (std::streamsize)data.size());
    return (bool)out;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr
            << "usage: center_egevfm_phase_mean [in.egevfm] [data_dir] [start_phase] [n_phase_files] [out.egevfm]"
            << " [max_records=0] [shrink=1.0] [indexed_feature_count=65]\n";
        return 1;
    }
    const std::string in_file = argv[1];
    const std::string data_dir = argv[2];
    const int start_phase = std::atoi(argv[3]);
    const int n_phase_files = std::atoi(argv[4]);
    const std::string out_file = argv[5];
    const uint64_t max_records = argc >= 7 ? std::strtoull(argv[6], nullptr, 10) : 0;
    const double shrink = argc >= 8 ? std::stod(argv[7]) : 1.0;
    const int indexed_feature_count = argc >= 9 ? std::atoi(argv[8]) : N_PATTERN_FEATURES + 1;
    if (start_phase < 0 || start_phase >= N_PHASES || n_phase_files <= 0 ||
        indexed_feature_count < N_PATTERN_FEATURES || shrink < 0.0 || shrink > 2.0) {
        std::cerr << "[ERROR] invalid arguments\n";
        return 1;
    }

    EgevfmFile fm;
    if (!load_egevfm(in_file, &fm)) {
        return 1;
    }
    std::array<int64_t, N_PHASES> phase_sum = {};
    std::array<uint64_t, N_PHASES> phase_count = {};
    if (!estimate_phase_means(fm, data_dir, start_phase, n_phase_files, max_records, indexed_feature_count, &phase_sum, &phase_count)) {
        return 1;
    }
    std::array<int, N_PHASES> corrections = {};
    apply_phase_correction(&fm, phase_sum, phase_count, shrink, &corrections);
    if (!write_file(out_file, fm.bytes)) {
        return 1;
    }

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "source " << in_file << "\n";
        summary << "data_dir " << data_dir << "\n";
        summary << "start_phase " << start_phase << "\n";
        summary << "n_phase_files " << n_phase_files << "\n";
        summary << "max_records " << max_records << "\n";
        summary << "shrink " << shrink << "\n";
        summary << "indexed_feature_count " << indexed_feature_count << "\n";
        summary << "phase count mean_fm_raw correction\n";
        for (int phase = 0; phase < N_PHASES; ++phase) {
            if (phase_count[(size_t)phase] == 0) {
                continue;
            }
            const double mean = (double)phase_sum[(size_t)phase] / (double)phase_count[(size_t)phase];
            summary << phase << " " << phase_count[(size_t)phase] << " " << mean << " " << corrections[(size_t)phase] << "\n";
        }
    }
    std::cout << "wrote " << out_file << "\n";
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (phase_count[(size_t)phase] == 0) {
            continue;
        }
        const double mean = (double)phase_sum[(size_t)phase] / (double)phase_count[(size_t)phase];
        std::cout << "phase " << phase << " count " << phase_count[(size_t)phase]
                  << " mean_fm_raw " << mean
                  << " correction " << corrections[(size_t)phase] << "\n";
    }
    return 0;
}
