/*
    Egaroucid Project

    @file eval77_grouped_semantic_compare.cpp
        Compare grouped-FM EGEV10 files by their per-phase evaluation
        semantics, ignoring different but equivalent FM group layouts.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int MAX_DIM = 16;
constexpr int N_PATTERN_PARAMS_RAW = 944784;
constexpr int MAX_STONE_NUM = 65;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int GROUPED_VERSION = 10;
constexpr size_t GROUPED_ALIGNMENT = 64;
constexpr size_t CHUNK_PARAMS = 8192;

struct GroupedInfo {
    int dim = 0;
    int group_count = 0;
    std::array<uint8_t, N_PHASES> phase_to_group{};
    std::array<float, N_PHASES> linear_scales{};
    std::array<float, N_PHASES> vector_scales_by_phase{};
    size_t linear_offset = 0;
    size_t vector_offset = 0;
    size_t linear_phase_stride = 0;
    size_t vector_param_stride = 0;
    size_t vector_phase_stride = 0;
};

struct CompareStats {
    int linear_scale_mismatches = 0;
    int vector_scale_mismatches = 0;
    uint64_t linear_mismatch_params = 0;
    uint64_t vector_mismatch_params = 0;
    uint64_t vector_mismatch_bytes = 0;
    int first_linear_mismatch_phase = -1;
    int first_linear_mismatch_param = -1;
    int first_vector_mismatch_phase = -1;
    int first_vector_mismatch_param = -1;
    int first_vector_mismatch_dim = -1;
};

int32_t read_i32_le(const unsigned char *p) {
    int32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

size_t align_up(const size_t value, const size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void read_exact_at(
    std::ifstream &in,
    const size_t offset,
    void *dst,
    const size_t bytes,
    const std::string &what
) {
    in.clear();
    in.seekg((std::streamoff)offset, std::ios::beg);
    in.read((char*)dst, (std::streamsize)bytes);
    if ((size_t)in.gcount() != bytes) {
        throw std::runtime_error("failed to read " + what);
    }
}

GroupedInfo read_grouped_info(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const size_t file_size = (size_t)in.tellg();
    constexpr size_t header_base_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 5 + N_PHASES;
    if (file_size < header_base_size + sizeof(float) * N_PHASES) {
        throw std::runtime_error("input is too short for grouped header: " + path);
    }

    std::vector<unsigned char> base_header(header_base_size);
    read_exact_at(in, 0, base_header.data(), base_header.size(), "grouped base header");
    if (std::memcmp(base_header.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("input does not have EGEV magic: " + path);
    }

    const int version = read_i32_le(base_header.data() + 18);
    const int n_phase = read_i32_le(base_header.data() + 22);
    const int n_param = read_i32_le(base_header.data() + 26);
    const int dim = read_i32_le(base_header.data() + 30);
    const int group_count = read_i32_le(base_header.data() + 34);
    if (version != GROUPED_VERSION || n_phase != N_PHASES || n_param != N_PARAMS_PER_PHASE ||
        dim <= 0 || dim > MAX_DIM || group_count <= 0 || group_count > N_PHASES) {
        throw std::runtime_error("unsupported grouped EGEV10 shape: " + path);
    }

    GroupedInfo info;
    info.dim = dim;
    info.group_count = group_count;
    std::memcpy(info.phase_to_group.data(), base_header.data() + 38, N_PHASES);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (info.phase_to_group[phase] >= group_count) {
            throw std::runtime_error("phase_to_group contains an out-of-range group: " + path);
        }
    }

    const size_t header_size = header_base_size + sizeof(float) * N_PHASES +
        sizeof(float) * (size_t)group_count;
    std::vector<unsigned char> scales(header_size - header_base_size);
    read_exact_at(in, header_base_size, scales.data(), scales.size(), "grouped scales");
    std::memcpy(info.linear_scales.data(), scales.data(), sizeof(float) * N_PHASES);
    std::vector<float> group_vector_scales((size_t)group_count);
    std::memcpy(group_vector_scales.data(), scales.data() + sizeof(float) * N_PHASES,
                sizeof(float) * (size_t)group_count);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        info.vector_scales_by_phase[phase] = group_vector_scales[info.phase_to_group[phase]];
    }

    info.linear_offset = align_up(header_size, GROUPED_ALIGNMENT);
    info.linear_phase_stride = (size_t)N_PARAMS_PER_PHASE * sizeof(int16_t);
    info.vector_param_stride = align_up((size_t)dim, 16);
    info.vector_phase_stride = (size_t)N_PARAMS_PER_PHASE * info.vector_param_stride;
    info.vector_offset = align_up(
        info.linear_offset + (size_t)N_PHASES * info.linear_phase_stride,
        GROUPED_ALIGNMENT
    );
    const size_t expected_size = info.vector_offset + (size_t)group_count * info.vector_phase_stride;
    if (file_size < expected_size) {
        throw std::runtime_error("grouped payload is truncated: " + path);
    }
    return info;
}

void compare_scales(const GroupedInfo &left, const GroupedInfo &right, CompareStats *stats) {
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (left.linear_scales[phase] != right.linear_scales[phase]) {
            ++stats->linear_scale_mismatches;
        }
        if (left.vector_scales_by_phase[phase] != right.vector_scales_by_phase[phase]) {
            ++stats->vector_scale_mismatches;
        }
    }
}

void compare_linear_tables(
    std::ifstream &left_file,
    std::ifstream &right_file,
    const GroupedInfo &left,
    const GroupedInfo &right,
    CompareStats *stats
) {
    std::vector<unsigned char> left_chunk;
    std::vector<unsigned char> right_chunk;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            left_chunk.resize(n_param * sizeof(int16_t));
            right_chunk.resize(n_param * sizeof(int16_t));
            read_exact_at(
                left_file,
                left.linear_offset + (size_t)phase * left.linear_phase_stride + start * sizeof(int16_t),
                left_chunk.data(),
                left_chunk.size(),
                "left linear chunk"
            );
            read_exact_at(
                right_file,
                right.linear_offset + (size_t)phase * right.linear_phase_stride + start * sizeof(int16_t),
                right_chunk.data(),
                right_chunk.size(),
                "right linear chunk"
            );
            for (size_t param = 0; param < n_param; ++param) {
                const size_t offset = param * sizeof(int16_t);
                if (std::memcmp(left_chunk.data() + offset, right_chunk.data() + offset, sizeof(int16_t)) != 0) {
                    ++stats->linear_mismatch_params;
                    if (stats->first_linear_mismatch_phase < 0) {
                        stats->first_linear_mismatch_phase = phase;
                        stats->first_linear_mismatch_param = (int)(start + param);
                    }
                }
            }
        }
    }
}

void compare_vector_tables(
    std::ifstream &left_file,
    std::ifstream &right_file,
    const GroupedInfo &left,
    const GroupedInfo &right,
    CompareStats *stats
) {
    std::vector<unsigned char> left_chunk;
    std::vector<unsigned char> right_chunk;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const int left_group = left.phase_to_group[phase];
        const int right_group = right.phase_to_group[phase];
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            left_chunk.resize(n_param * left.vector_param_stride);
            right_chunk.resize(n_param * right.vector_param_stride);
            read_exact_at(
                left_file,
                left.vector_offset + (size_t)left_group * left.vector_phase_stride +
                    start * left.vector_param_stride,
                left_chunk.data(),
                left_chunk.size(),
                "left vector chunk"
            );
            read_exact_at(
                right_file,
                right.vector_offset + (size_t)right_group * right.vector_phase_stride +
                    start * right.vector_param_stride,
                right_chunk.data(),
                right_chunk.size(),
                "right vector chunk"
            );
            for (size_t param = 0; param < n_param; ++param) {
                bool param_mismatch = false;
                const size_t left_offset = param * left.vector_param_stride;
                const size_t right_offset = param * right.vector_param_stride;
                for (int dim = 0; dim < left.dim; ++dim) {
                    if (left_chunk[left_offset + (size_t)dim] != right_chunk[right_offset + (size_t)dim]) {
                        ++stats->vector_mismatch_bytes;
                        param_mismatch = true;
                        if (stats->first_vector_mismatch_phase < 0) {
                            stats->first_vector_mismatch_phase = phase;
                            stats->first_vector_mismatch_param = (int)(start + param);
                            stats->first_vector_mismatch_dim = dim;
                        }
                    }
                }
                if (param_mismatch) {
                    ++stats->vector_mismatch_params;
                }
            }
        }
    }
}

CompareStats compare_grouped_semantics(const std::string &left_path, const std::string &right_path) {
    const GroupedInfo left = read_grouped_info(left_path);
    const GroupedInfo right = read_grouped_info(right_path);
    if (left.dim != right.dim) {
        throw std::runtime_error("grouped files have different dimensions");
    }

    std::ifstream left_file(left_path, std::ios::binary);
    std::ifstream right_file(right_path, std::ios::binary);
    if (!left_file || !right_file) {
        throw std::runtime_error("failed to reopen grouped files");
    }

    CompareStats stats;
    compare_scales(left, right, &stats);
    compare_linear_tables(left_file, right_file, left, right, &stats);
    compare_vector_tables(left_file, right_file, left, right, &stats);

    std::cout << "left=" << left_path
              << " left_groups=" << left.group_count
              << " right=" << right_path
              << " right_groups=" << right.group_count
              << " phases=" << N_PHASES
              << " params/phase=" << N_PARAMS_PER_PHASE
              << " dim=" << left.dim
              << std::endl;
    return stats;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: eval77_grouped_semantic_compare <left.egev10> <right.egev10>" << std::endl;
        return 1;
    }
    try {
        const CompareStats stats = compare_grouped_semantics(argv[1], argv[2]);
        const bool semantic_equal =
            stats.linear_scale_mismatches == 0 &&
            stats.vector_scale_mismatches == 0 &&
            stats.linear_mismatch_params == 0 &&
            stats.vector_mismatch_params == 0;
        std::cout << "linear_scale_mismatches=" << stats.linear_scale_mismatches
                  << " vector_scale_mismatches=" << stats.vector_scale_mismatches
                  << " linear_mismatch_params=" << stats.linear_mismatch_params
                  << " vector_mismatch_params=" << stats.vector_mismatch_params
                  << " vector_mismatch_bytes=" << stats.vector_mismatch_bytes
                  << " first_linear_mismatch_phase=" << stats.first_linear_mismatch_phase
                  << " first_linear_mismatch_param=" << stats.first_linear_mismatch_param
                  << " first_vector_mismatch_phase=" << stats.first_vector_mismatch_phase
                  << " first_vector_mismatch_param=" << stats.first_vector_mismatch_param
                  << " first_vector_mismatch_dim=" << stats.first_vector_mismatch_dim
                  << " semantic_equal=" << semantic_equal
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
