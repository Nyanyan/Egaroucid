/*
    Egaroucid Project

    @file merge_egevfm_phases.cpp
        Merge phase-wise egevfm files into one egevfm file
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr char EVAL_FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
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

struct EgevfmFile {
    std::string path;
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

bool load_egevfm(const std::string &path, EgevfmFile *res) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << path << "\n";
        return false;
    }
    res->path = path;
    res->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (res->bytes.size() < EVAL_FM_HEADER_SIZE ||
        std::memcmp(res->bytes.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        std::cerr << "[ERROR] input is not egevfm v1: " << path << "\n";
        return false;
    }

    size_t offset = sizeof(EVAL_FM_FILE_MAGIC);
    const bool ok =
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
    if (!ok || offset != EVAL_FM_HEADER_SIZE || res->version != EVAL_FM_FILE_VERSION_LOCAL ||
        res->dim == 0 || res->scale <= 0 || res->n_phases == 0 ||
        res->n_fm_phases != res->n_phases || res->fm_count % res->n_fm_phases != 0) {
        std::cerr << "[ERROR] unsupported egevfm layout " << path << "\n";
        return false;
    }

    res->linear_offset = EVAL_FM_HEADER_SIZE;
    res->fm_offset = res->linear_offset + (size_t)res->linear_count * sizeof(int16_t);
    const size_t expected_size = res->fm_offset + (size_t)res->fm_count * sizeof(int8_t);
    if (expected_size != res->bytes.size()) {
        std::cerr << "[ERROR] broken egevfm payload size " << path << "\n";
        return false;
    }
    return true;
}

bool compatible_with_base(const EgevfmFile &base, const EgevfmFile &target) {
    if (base.n_phases != target.n_phases ||
        base.linear_params_per_phase != target.linear_params_per_phase ||
        base.n_fm_phases != target.n_fm_phases ||
        base.n_features != target.n_features ||
        base.dim != target.dim ||
        base.scale != target.scale ||
        base.flags != target.flags ||
        base.linear_count != target.linear_count ||
        base.fm_count != target.fm_count ||
        base.linear_offset != target.linear_offset ||
        base.fm_offset != target.fm_offset) {
        return false;
    }
    return std::memcmp(
        base.bytes.data() + base.linear_offset,
        target.bytes.data() + target.linear_offset,
        (size_t)base.linear_count * sizeof(int16_t)
    ) == 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || ((argc - 2) % 2) != 0) {
        std::cerr << "usage: merge_egevfm_phases [out.egevfm] [phase0] [phase0.egevfm] [phase1] [phase1.egevfm] ...\n";
        return 1;
    }

    const std::string out_file = argv[1];
    EgevfmFile base;
    if (!load_egevfm(argv[3], &base)) {
        return 1;
    }
    const uint64_t phase_slice = base.fm_count / base.n_fm_phases;

    std::vector<char> output = base.bytes;
    std::memset(output.data() + base.fm_offset, 0, (size_t)base.fm_count * sizeof(int8_t));
    std::vector<uint8_t> merged((size_t)base.n_phases, 0);
    std::vector<std::string> merged_paths((size_t)base.n_phases);

    for (int arg = 2; arg < argc; arg += 2) {
        int phase = -1;
        try {
            phase = std::stoi(argv[arg]);
        } catch (...) {
            std::cerr << "[ERROR] invalid phase: " << argv[arg] << "\n";
            return 1;
        }
        if (phase < 0 || phase >= (int)base.n_phases) {
            std::cerr << "[ERROR] phase out of range: " << phase << "\n";
            return 1;
        }
        if (merged[(size_t)phase]) {
            std::cerr << "[ERROR] duplicated phase: " << phase << "\n";
            return 1;
        }

        EgevfmFile input;
        if (!load_egevfm(argv[arg + 1], &input)) {
            return 1;
        }
        if (!compatible_with_base(base, input)) {
            std::cerr << "[ERROR] incompatible egevfm input " << argv[arg + 1] << "\n";
            return 1;
        }

        const size_t offset = base.fm_offset + (size_t)((uint64_t)phase * phase_slice);
        std::memcpy(output.data() + offset, input.bytes.data() + offset, (size_t)phase_slice);
        merged[(size_t)phase] = 1;
        merged_paths[(size_t)phase] = input.path;
    }

    int merged_count = 0;
    int missing_count = 0;
    int nonzero = 0;
    int max_abs = 0;
    for (const uint8_t value: merged) {
        if (value) {
            ++merged_count;
        } else {
            ++missing_count;
        }
    }
    for (uint64_t i = 0; i < base.fm_count; ++i) {
        int8_t q = 0;
        std::memcpy(&q, output.data() + base.fm_offset + (size_t)i, sizeof(int8_t));
        if (q != 0) {
            ++nonzero;
            max_abs = std::max(max_abs, std::abs((int)q));
        }
    }

    std::filesystem::path out_path(out_file);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << "\n";
        return 1;
    }
    out.write(output.data(), (std::streamsize)output.size());
    if (!out) {
        std::cerr << "[ERROR] failed to write output " << out_file << "\n";
        return 1;
    }

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "n_phases " << base.n_phases << "\n";
        summary << "n_fm_phases " << base.n_fm_phases << "\n";
        summary << "dim " << base.dim << "\n";
        summary << "scale " << base.scale << "\n";
        summary << "flags " << base.flags << "\n";
        summary << "active_pattern_mask 0x" << std::hex << (base.flags & 0xFFFFU) << std::dec << "\n";
        summary << "fm_count " << base.fm_count << "\n";
        summary << "phase_slice_values " << phase_slice << "\n";
        summary << "merged_phases " << merged_count << "\n";
        summary << "missing_phases " << missing_count << "\n";
        summary << "nonzero_quantized " << nonzero << "\n";
        summary << "max_abs_quantized " << max_abs << "\n";
        for (uint32_t phase = 0; phase < base.n_phases; ++phase) {
            summary << "phase " << phase << " merged " << (int)merged[(size_t)phase];
            if (merged[(size_t)phase]) {
                summary << " source " << merged_paths[(size_t)phase];
            }
            summary << "\n";
        }
    }

    std::cout << "wrote " << out_file
              << " merged_phases " << merged_count
              << " missing_phases " << missing_count
              << " nonzero_quantized " << nonzero
              << " max_abs_quantized " << max_abs << "\n";
    return missing_count == 0 ? 0 : 2;
}
