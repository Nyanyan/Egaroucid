/*
    Egaroucid Project

    @file prune_egevfm.cpp
        Zero small quantized FM values in an egevfm file
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

struct EgevfmHeader {
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

bool parse_header(const std::vector<char> &data, EgevfmHeader *header) {
    if (data.size() < EVAL_FM_HEADER_SIZE ||
        std::memcmp(data.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        return false;
    }
    size_t offset = sizeof(EVAL_FM_FILE_MAGIC);
    const bool ok =
        read_scalar(data, &offset, &header->version) &&
        read_scalar(data, &offset, &header->n_phases) &&
        read_scalar(data, &offset, &header->linear_params_per_phase) &&
        read_scalar(data, &offset, &header->n_fm_phases) &&
        read_scalar(data, &offset, &header->n_features) &&
        read_scalar(data, &offset, &header->dim) &&
        read_scalar(data, &offset, &header->scale) &&
        read_scalar(data, &offset, &header->flags) &&
        read_scalar(data, &offset, &header->linear_count) &&
        read_scalar(data, &offset, &header->fm_count);
    if (!ok || offset != EVAL_FM_HEADER_SIZE || header->version != EVAL_FM_FILE_VERSION_LOCAL ||
        header->dim == 0 || header->scale <= 0) {
        return false;
    }
    header->linear_offset = EVAL_FM_HEADER_SIZE;
    header->fm_offset = header->linear_offset + (size_t)header->linear_count * sizeof(int16_t);
    const size_t expected_size = header->fm_offset + (size_t)header->fm_count * sizeof(int8_t);
    return expected_size == data.size();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: prune_egevfm [in.egevfm] [out.egevfm] [abs_threshold]\n";
        return 1;
    }
    const std::string in_file = argv[1];
    const std::string out_file = argv[2];
    int threshold = 0;
    try {
        threshold = std::stoi(argv[3]);
    } catch (...) {
        std::cerr << "[ERROR] invalid abs_threshold\n";
        return 1;
    }
    if (threshold < 0 || threshold > 127) {
        std::cerr << "[ERROR] abs_threshold must be in [0, 127]\n";
        return 1;
    }

    std::ifstream in(in_file, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << in_file << "\n";
        return 1;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EgevfmHeader header;
    if (!parse_header(data, &header)) {
        std::cerr << "[ERROR] input is not supported egevfm v1: " << in_file << "\n";
        return 1;
    }

    int old_nonzero = 0;
    int new_nonzero = 0;
    int pruned = 0;
    int old_max_abs = 0;
    int new_max_abs = 0;
    for (uint64_t i = 0; i < header.fm_count; ++i) {
        const size_t offset = header.fm_offset + (size_t)i;
        int8_t q = 0;
        std::memcpy(&q, data.data() + offset, sizeof(int8_t));
        const int abs_q = std::abs((int)q);
        if (q != 0) {
            ++old_nonzero;
            old_max_abs = std::max(old_max_abs, abs_q);
        }
        if (abs_q <= threshold) {
            if (q != 0) {
                ++pruned;
                q = 0;
                std::memcpy(data.data() + offset, &q, sizeof(int8_t));
            }
        }
        if (q != 0) {
            ++new_nonzero;
            new_max_abs = std::max(new_max_abs, std::abs((int)q));
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
    out.write(data.data(), (std::streamsize)data.size());
    if (!out) {
        std::cerr << "[ERROR] failed to write output " << out_file << "\n";
        return 1;
    }

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "source " << in_file << "\n";
        summary << "abs_threshold " << threshold << "\n";
        summary << "n_fm_phases " << header.n_fm_phases << "\n";
        summary << "dim " << header.dim << "\n";
        summary << "scale " << header.scale << "\n";
        summary << "flags " << header.flags << "\n";
        summary << "active_pattern_mask 0x" << std::hex << (header.flags & 0xFFFFU) << std::dec << "\n";
        summary << "fm_count " << header.fm_count << "\n";
        summary << "old_nonzero_quantized " << old_nonzero << "\n";
        summary << "new_nonzero_quantized " << new_nonzero << "\n";
        summary << "pruned_quantized " << pruned << "\n";
        summary << "old_max_abs_quantized " << old_max_abs << "\n";
        summary << "new_max_abs_quantized " << new_max_abs << "\n";
    }

    std::cout << "wrote " << out_file
              << " threshold " << threshold
              << " old_nonzero " << old_nonzero
              << " new_nonzero " << new_nonzero
              << " pruned " << pruned << "\n";
    return 0;
}
