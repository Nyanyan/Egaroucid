/*
    Egaroucid Project

    @file eval77_fm_phase_vector_compare.cpp
        Compare 7.7-FM EGEV4 phase vector tables for candidate grouped-FM
        phase pairs.
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int MAX_DIM = 16;
constexpr int N_PATTERN_PARAMS_RAW = 944784;
constexpr int MAX_STONE_NUM = 65;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr size_t EGEV4_FIXED_HEADER_SIZE = 34;
constexpr size_t EGEV4_SCALES_SIZE = sizeof(float) * N_PHASES * 2;
constexpr size_t EGEV4_PAYLOAD_OFFSET = EGEV4_FIXED_HEADER_SIZE + EGEV4_SCALES_SIZE;
constexpr size_t CHUNK_PARAMS = 8192;

struct InputInfo {
    int version = 0;
    int dim = 0;
    std::array<float, N_PHASES> linear_scales{};
    std::array<float, N_PHASES> vector_scales{};
    size_t record_stride = 0;
    size_t phase_stride = 0;
};

struct PairStats {
    int phase_a = 0;
    int phase_b = 0;
    uint64_t linear_mismatch_params = 0;
    uint64_t vector_mismatch_params = 0;
    uint64_t vector_mismatch_bytes = 0;
    int first_vector_mismatch_param = -1;
    int first_vector_mismatch_dim = -1;
};

int32_t read_i32_le(const unsigned char *p) {
    int32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
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

InputInfo read_input_info(const std::string &input_path) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input: " + input_path);
    }

    std::array<unsigned char, EGEV4_PAYLOAD_OFFSET> header{};
    read_exact_at(in, 0, header.data(), header.size(), "EGEV4 header");
    if (std::memcmp(header.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("input does not have EGEV magic");
    }

    InputInfo info;
    info.version = read_i32_le(header.data() + 18);
    const int32_t n_phase = read_i32_le(header.data() + 22);
    const int32_t n_param = read_i32_le(header.data() + 26);
    info.dim = read_i32_le(header.data() + 30);
    if (info.version != 7 && info.version != 8) {
        throw std::runtime_error("input must be EGEV4 version 7 or 8");
    }
    if (n_phase != N_PHASES || n_param != N_PARAMS_PER_PHASE ||
        info.dim <= 0 || info.dim > MAX_DIM) {
        throw std::runtime_error("unsupported 7.7 beta + FM EGEV4 shape");
    }

    std::memcpy(info.linear_scales.data(), header.data() + EGEV4_FIXED_HEADER_SIZE,
                sizeof(float) * N_PHASES);
    std::memcpy(info.vector_scales.data(), header.data() + EGEV4_FIXED_HEADER_SIZE + sizeof(float) * N_PHASES,
                sizeof(float) * N_PHASES);
    info.record_stride = sizeof(int16_t) + (size_t)info.dim;
    info.phase_stride = (size_t)N_PARAMS_PER_PHASE * info.record_stride;
    return info;
}

std::vector<std::pair<int, int>> parse_pairs(const std::string &arg) {
    std::vector<std::pair<int, int>> pairs;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const size_t dash = token.find('-');
        if (dash == std::string::npos) {
            throw std::runtime_error("pair must be A-B: " + token);
        }
        size_t parsed_a = 0;
        size_t parsed_b = 0;
        const int a = std::stoi(token.substr(0, dash), &parsed_a);
        const std::string b_text = token.substr(dash + 1);
        const int b = std::stoi(b_text, &parsed_b);
        if (parsed_a != dash || parsed_b != b_text.size()) {
            throw std::runtime_error("invalid pair: " + token);
        }
        if (a < 0 || N_PHASES <= a || b < 0 || N_PHASES <= b) {
            throw std::runtime_error("phase is out of range in pair: " + token);
        }
        pairs.emplace_back(a, b);
    }
    if (pairs.empty()) {
        throw std::runtime_error("no pairs were specified");
    }
    return pairs;
}

PairStats compare_pair(
    std::ifstream &in,
    const InputInfo &info,
    const int phase_a,
    const int phase_b
) {
    PairStats stats;
    stats.phase_a = phase_a;
    stats.phase_b = phase_b;
    std::vector<unsigned char> buffer_a;
    std::vector<unsigned char> buffer_b;

    for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
        const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
        buffer_a.resize(n_param * info.record_stride);
        buffer_b.resize(n_param * info.record_stride);
        read_exact_at(
            in,
            EGEV4_PAYLOAD_OFFSET + (size_t)phase_a * info.phase_stride + start * info.record_stride,
            buffer_a.data(),
            buffer_a.size(),
            "phase A chunk"
        );
        read_exact_at(
            in,
            EGEV4_PAYLOAD_OFFSET + (size_t)phase_b * info.phase_stride + start * info.record_stride,
            buffer_b.data(),
            buffer_b.size(),
            "phase B chunk"
        );
        for (size_t param = 0; param < n_param; ++param) {
            const size_t offset = param * info.record_stride;
            if (std::memcmp(buffer_a.data() + offset, buffer_b.data() + offset, sizeof(int16_t)) != 0) {
                ++stats.linear_mismatch_params;
            }
            bool vector_param_mismatch = false;
            for (int dim = 0; dim < info.dim; ++dim) {
                const size_t vector_offset = offset + sizeof(int16_t) + (size_t)dim;
                if (buffer_a[vector_offset] != buffer_b[vector_offset]) {
                    ++stats.vector_mismatch_bytes;
                    vector_param_mismatch = true;
                    if (stats.first_vector_mismatch_param < 0) {
                        stats.first_vector_mismatch_param = (int)(start + param);
                        stats.first_vector_mismatch_dim = dim;
                    }
                }
            }
            if (vector_param_mismatch) {
                ++stats.vector_mismatch_params;
            }
        }
    }
    return stats;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: eval77_fm_phase_vector_compare <eval.egev4> <pair_csv_like_0-1,2-3>" << std::endl;
        return 1;
    }
    try {
        const std::string input_path = argv[1];
        const std::vector<std::pair<int, int>> pairs = parse_pairs(argv[2]);
        const InputInfo info = read_input_info(input_path);
        std::ifstream in(input_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot reopen input: " + input_path);
        }
        std::cout << "input=" << input_path
                  << " version=" << info.version
                  << " phases=" << N_PHASES
                  << " params/phase=" << N_PARAMS_PER_PHASE
                  << " dim=" << info.dim
                  << std::endl;
        for (const auto [a, b] : pairs) {
            const PairStats stats = compare_pair(in, info, a, b);
            const bool linear_scale_equal = info.linear_scales[(size_t)a] == info.linear_scales[(size_t)b];
            const bool vector_scale_equal = info.vector_scales[(size_t)a] == info.vector_scales[(size_t)b];
            const bool vector_lossless = vector_scale_equal && stats.vector_mismatch_params == 0;
            std::cout << "pair " << a << '-' << b
                      << " linear_scale_equal=" << linear_scale_equal
                      << " linear_scale_a=" << info.linear_scales[(size_t)a]
                      << " linear_scale_b=" << info.linear_scales[(size_t)b]
                      << " vector_scale_equal=" << vector_scale_equal
                      << " vector_scale_a=" << info.vector_scales[(size_t)a]
                      << " vector_scale_b=" << info.vector_scales[(size_t)b]
                      << " linear_mismatch_params=" << stats.linear_mismatch_params
                      << " vector_mismatch_params=" << stats.vector_mismatch_params
                      << " vector_mismatch_bytes=" << stats.vector_mismatch_bytes
                      << " first_vector_mismatch_param=" << stats.first_vector_mismatch_param
                      << " first_vector_mismatch_dim=" << stats.first_vector_mismatch_dim
                      << " vector_lossless=" << vector_lossless
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
