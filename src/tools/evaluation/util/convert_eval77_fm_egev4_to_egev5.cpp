/*
    Egaroucid Project

    @file convert_eval77_fm_egev4_to_egev5.cpp
        Convert 7.7 beta + FM EGEV4 interleaved layout to EGEV5 split aligned layout.
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

constexpr int N_PHASES = 60;
constexpr int MAX_DIM = 16;
constexpr int N_PATTERN_PARAMS_RAW = 944784;
constexpr int MAX_STONE_NUM = 65;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int EGEV5_VERSION = 9;
constexpr size_t FIXED_HEADER_SIZE = 34;
constexpr size_t SCALES_SIZE = sizeof(float) * N_PHASES * 2;
constexpr size_t PAYLOAD_OFFSET = FIXED_HEADER_SIZE + SCALES_SIZE;
constexpr size_t EGEV5_ALIGNMENT = 64;

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

void write_padding(std::ofstream &out, const size_t target_offset) {
    const size_t current = (size_t)out.tellp();
    if (current > target_offset) {
        throw std::runtime_error("internal offset error while padding");
    }
    std::array<unsigned char, EGEV5_ALIGNMENT> zeros{};
    size_t remain = target_offset - current;
    while (remain > 0) {
        const size_t chunk = std::min(remain, zeros.size());
        out.write((const char*)zeros.data(), (std::streamsize)chunk);
        remain -= chunk;
    }
}

void read_exact(std::ifstream &in, void *dst, const size_t bytes, const std::string &what) {
    in.read((char*)dst, (std::streamsize)bytes);
    if ((size_t)in.gcount() != bytes) {
        throw std::runtime_error("failed to read " + what);
    }
}

void convert(const std::string &input_path, const std::string &output_path) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input: " + input_path);
    }

    std::vector<unsigned char> header(PAYLOAD_OFFSET);
    read_exact(in, header.data(), header.size(), "header");
    if (std::memcmp(header.data() + 14, "EGEV", 4) != 0) {
        throw std::runtime_error("input does not have EGEV magic");
    }

    const int32_t version = read_i32_le(header.data() + 18);
    const int32_t n_phase = read_i32_le(header.data() + 22);
    const int32_t n_param = read_i32_le(header.data() + 26);
    const int32_t dim = read_i32_le(header.data() + 30);
    if (version != 7 && version != 8) {
        throw std::runtime_error("input must be EGEV4 version 7 or 8");
    }
    if (n_phase != N_PHASES || n_param != N_PARAMS_PER_PHASE || dim <= 0 || dim > MAX_DIM) {
        throw std::runtime_error("unsupported 7.7 FM shape");
    }

    const size_t input_param_stride = sizeof(int16_t) + (size_t)dim;
    const size_t input_phase_stride = (size_t)n_param * input_param_stride;
    const size_t linear_offset = align_up(PAYLOAD_OFFSET, EGEV5_ALIGNMENT);
    const size_t linear_param_stride = sizeof(int16_t);
    const size_t linear_phase_stride = (size_t)n_param * linear_param_stride;
    const size_t vector_param_stride = align_up((size_t)dim, 16);
    const size_t vector_phase_stride = (size_t)n_param * vector_param_stride;
    const size_t vector_offset = align_up(linear_offset + (size_t)n_phase * linear_phase_stride, EGEV5_ALIGNMENT);
    const size_t output_size = vector_offset + (size_t)n_phase * vector_phase_stride;

    write_i32_le(header.data() + 18, EGEV5_VERSION);

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open output: " + output_path);
    }
    out.write((const char*)header.data(), (std::streamsize)header.size());
    write_padding(out, linear_offset);

    std::vector<unsigned char> phase(input_phase_stride);
    for (int phase_idx = 0; phase_idx < n_phase; ++phase_idx) {
        in.seekg((std::streamoff)(PAYLOAD_OFFSET + (size_t)phase_idx * input_phase_stride), std::ios::beg);
        read_exact(in, phase.data(), phase.size(), "input phase payload for linear table");
        for (int id = 0; id < n_param; ++id) {
            out.write((const char*)(phase.data() + (size_t)id * input_param_stride), sizeof(int16_t));
        }
    }

    write_padding(out, vector_offset);

    const std::array<unsigned char, 16> zeros{};
    for (int phase_idx = 0; phase_idx < n_phase; ++phase_idx) {
        in.seekg((std::streamoff)(PAYLOAD_OFFSET + (size_t)phase_idx * input_phase_stride), std::ios::beg);
        read_exact(in, phase.data(), phase.size(), "input phase payload for vector table");
        for (int id = 0; id < n_param; ++id) {
            const unsigned char *src = phase.data() + (size_t)id * input_param_stride + sizeof(int16_t);
            out.write((const char*)src, dim);
            if ((size_t)dim < vector_param_stride) {
                out.write((const char*)zeros.data(), (std::streamsize)(vector_param_stride - (size_t)dim));
            }
        }
    }

    if (!out) {
        throw std::runtime_error("failed while writing output");
    }

    std::cerr << "converted " << input_path << " -> " << output_path << "\n"
              << "version " << version << " -> " << EGEV5_VERSION
              << " phases " << n_phase
              << " params/phase " << n_param
              << " dim " << dim << "\n"
              << "linear_offset " << linear_offset
              << " vector_offset " << vector_offset
              << " output_bytes " << output_size << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: convert_eval77_fm_egev4_to_egev5 <input.egev4> <output.egev5>" << std::endl;
        return 1;
    }
    try {
        convert(argv[1], argv[2]);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
