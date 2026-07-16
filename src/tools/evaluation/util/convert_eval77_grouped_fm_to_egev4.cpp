/*
    Egaroucid Project

    @file convert_eval77_grouped_fm_to_egev4.cpp
        Convert grouped-FM EGEV version 10 back to an EGEV4 interleaved file
        by duplicating each FM group vector table into its mapped phases.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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
constexpr int EGEV4_OUTPUT_VERSION = 8;
constexpr size_t GROUPED_ALIGNMENT = 64;
constexpr size_t EGEV4_FIXED_HEADER_SIZE = 34;
constexpr size_t EGEV4_SCALES_SIZE = sizeof(float) * N_PHASES * 2;
constexpr size_t EGEV4_PAYLOAD_OFFSET = EGEV4_FIXED_HEADER_SIZE + EGEV4_SCALES_SIZE;
constexpr size_t CHUNK_PARAMS = 8192;

struct GroupedInfo {
    std::string timestamp;
    int dim = 0;
    int group_count = 0;
    std::array<uint8_t, N_PHASES> phase_to_group{};
    std::array<float, N_PHASES> linear_scales{};
    std::array<float, N_PHASES> vector_scales_by_phase{};
    size_t header_size = 0;
    size_t linear_offset = 0;
    size_t vector_offset = 0;
    size_t linear_phase_stride = 0;
    size_t vector_param_stride = 0;
    size_t vector_phase_stride = 0;
};

int32_t read_i32_le(const unsigned char *p) {
    int32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

void write_i32_le(unsigned char *p, const int32_t value) {
    std::memcpy(p, &value, sizeof(value));
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

std::string timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return oss.str();
}

bool valid_timestamp(const std::string &timestamp) {
    return timestamp.size() == TIMESTAMP_SIZE &&
        std::all_of(timestamp.begin(), timestamp.end(), [](const char c) {
            return '0' <= c && c <= '9';
        });
}

GroupedInfo read_grouped_info(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open input: " + path);
    }
    const size_t file_size = (size_t)in.tellg();
    constexpr size_t header_base_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 5 + N_PHASES;
    if (file_size < header_base_size + sizeof(float) * N_PHASES) {
        throw std::runtime_error("input is too short for grouped header");
    }

    std::vector<unsigned char> base_header(header_base_size);
    read_exact_at(in, 0, base_header.data(), base_header.size(), "grouped base header");
    if (std::memcmp(base_header.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("input does not have EGEV magic");
    }

    const int version = read_i32_le(base_header.data() + 18);
    const int n_phase = read_i32_le(base_header.data() + 22);
    const int n_param = read_i32_le(base_header.data() + 26);
    const int dim = read_i32_le(base_header.data() + 30);
    const int group_count = read_i32_le(base_header.data() + 34);
    if (version != GROUPED_VERSION) {
        throw std::runtime_error("input must be grouped EGEV version 10");
    }
    if (n_phase != N_PHASES || n_param != N_PARAMS_PER_PHASE ||
        dim <= 0 || dim > MAX_DIM || group_count <= 0 || group_count > N_PHASES) {
        throw std::runtime_error("unsupported grouped 7.7 beta + FM shape");
    }

    GroupedInfo info;
    info.timestamp.assign((const char*)base_header.data(), TIMESTAMP_SIZE);
    info.dim = dim;
    info.group_count = group_count;
    std::memcpy(info.phase_to_group.data(), base_header.data() + 38, N_PHASES);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        if (info.phase_to_group[phase] >= group_count) {
            throw std::runtime_error("phase_to_group contains an out-of-range group");
        }
    }

    info.header_size = header_base_size + sizeof(float) * N_PHASES + sizeof(float) * (size_t)group_count;
    std::vector<unsigned char> scales(info.header_size - header_base_size);
    read_exact_at(in, header_base_size, scales.data(), scales.size(), "grouped scales");
    std::memcpy(info.linear_scales.data(), scales.data(), sizeof(float) * N_PHASES);
    std::vector<float> group_vector_scales((size_t)group_count);
    std::memcpy(group_vector_scales.data(), scales.data() + sizeof(float) * N_PHASES,
                sizeof(float) * (size_t)group_count);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        info.vector_scales_by_phase[phase] = group_vector_scales[info.phase_to_group[phase]];
    }

    info.linear_offset = align_up(info.header_size, GROUPED_ALIGNMENT);
    info.linear_phase_stride = (size_t)N_PARAMS_PER_PHASE * sizeof(int16_t);
    info.vector_param_stride = align_up((size_t)dim, 16);
    info.vector_phase_stride = (size_t)N_PARAMS_PER_PHASE * info.vector_param_stride;
    info.vector_offset = align_up(info.linear_offset + (size_t)N_PHASES * info.linear_phase_stride,
                                  GROUPED_ALIGNMENT);
    const size_t expected_size = info.vector_offset + (size_t)group_count * info.vector_phase_stride;
    if (file_size < expected_size) {
        throw std::runtime_error("grouped payload is truncated");
    }
    return info;
}

void write_egev4_header(
    std::ofstream &out,
    const GroupedInfo &info,
    const std::string &timestamp
) {
    std::array<unsigned char, EGEV4_PAYLOAD_OFFSET> header{};
    std::memcpy(header.data(), timestamp.data(), TIMESTAMP_SIZE);
    std::memcpy(header.data() + TIMESTAMP_SIZE, "EGEV", 4);
    write_i32_le(header.data() + 18, EGEV4_OUTPUT_VERSION);
    write_i32_le(header.data() + 22, N_PHASES);
    write_i32_le(header.data() + 26, N_PARAMS_PER_PHASE);
    write_i32_le(header.data() + 30, info.dim);
    std::memcpy(header.data() + EGEV4_FIXED_HEADER_SIZE, info.linear_scales.data(), sizeof(float) * N_PHASES);
    std::memcpy(header.data() + EGEV4_FIXED_HEADER_SIZE + sizeof(float) * N_PHASES,
                info.vector_scales_by_phase.data(), sizeof(float) * N_PHASES);
    out.write((const char*)header.data(), (std::streamsize)header.size());
}

void convert(
    const std::string &input_path,
    const std::string &output_path,
    const std::string &timestamp_arg
) {
    const GroupedInfo info = read_grouped_info(input_path);
    std::string timestamp = info.timestamp;
    if (!timestamp_arg.empty() && timestamp_arg != "preserve") {
        timestamp = timestamp_arg == "now" ? timestamp_now() : timestamp_arg;
    }
    if (!valid_timestamp(timestamp)) {
        throw std::runtime_error("timestamp must be 14 digits, 'now', or 'preserve'");
    }

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot reopen input: " + input_path);
    }
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open output: " + output_path);
    }

    write_egev4_header(out, info, timestamp);

    std::vector<unsigned char> linear_chunk;
    std::vector<unsigned char> vector_chunk;
    std::vector<unsigned char> output_chunk;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const int group = info.phase_to_group[phase];
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            linear_chunk.resize(n_param * sizeof(int16_t));
            vector_chunk.resize(n_param * info.vector_param_stride);
            output_chunk.resize(n_param * (sizeof(int16_t) + (size_t)info.dim));
            read_exact_at(
                in,
                info.linear_offset + (size_t)phase * info.linear_phase_stride + start * sizeof(int16_t),
                linear_chunk.data(),
                linear_chunk.size(),
                "grouped linear chunk"
            );
            read_exact_at(
                in,
                info.vector_offset + (size_t)group * info.vector_phase_stride + start * info.vector_param_stride,
                vector_chunk.data(),
                vector_chunk.size(),
                "grouped vector chunk"
            );

            const size_t output_stride = sizeof(int16_t) + (size_t)info.dim;
            for (size_t param = 0; param < n_param; ++param) {
                std::memcpy(output_chunk.data() + param * output_stride,
                            linear_chunk.data() + param * sizeof(int16_t),
                            sizeof(int16_t));
                std::memcpy(output_chunk.data() + param * output_stride + sizeof(int16_t),
                            vector_chunk.data() + param * info.vector_param_stride,
                            (size_t)info.dim);
            }
            out.write((const char*)output_chunk.data(), (std::streamsize)output_chunk.size());
        }
    }
    if (!out) {
        throw std::runtime_error("failed while writing output");
    }

    const size_t output_size = EGEV4_PAYLOAD_OFFSET +
        (size_t)N_PHASES * (size_t)N_PARAMS_PER_PHASE * (sizeof(int16_t) + (size_t)info.dim);
    std::cerr << "converted " << input_path << " -> " << output_path << "\n"
              << "input_version " << GROUPED_VERSION
              << " output_version " << EGEV4_OUTPUT_VERSION
              << " phases " << N_PHASES
              << " params/phase " << N_PARAMS_PER_PHASE
              << " dim " << info.dim
              << " fm_groups " << info.group_count
              << " output_bytes " << output_size << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: convert_eval77_grouped_fm_to_egev4 <input.egev10> <output.egev4> [preserve|now|YYYYMMDDHHMMSS]" << std::endl;
        return 1;
    }
    try {
        const std::string timestamp = argc == 4 ? argv[3] : "preserve";
        convert(argv[1], argv[2], timestamp);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
