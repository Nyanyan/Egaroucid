/*
    Egaroucid Project

    @file analyze_eval77_fm_vector_distribution.cpp
        Analyze quantized FM-vector values in an EGEV10 evaluation file.
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
#include <string>
#include <vector>

namespace {

constexpr int N_PHASES = 60;
constexpr size_t ALIGNMENT = 64;

int32_t read_i32_le(const unsigned char *ptr) {
    return static_cast<int32_t>(
        static_cast<uint32_t>(ptr[0]) |
        (static_cast<uint32_t>(ptr[1]) << 8) |
        (static_cast<uint32_t>(ptr[2]) << 16) |
        (static_cast<uint32_t>(ptr[3]) << 24)
    );
}

size_t align_up(const size_t value, const size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: analyze_eval77_fm_vector_distribution <eval.egev10>" << std::endl;
        return 1;
    }

    const std::string path = argv[1];
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "cannot open " << path << std::endl;
        return 1;
    }

    std::array<unsigned char, 128> fixed_header{};
    input.read(
        reinterpret_cast<char *>(fixed_header.data()),
        static_cast<std::streamsize>(fixed_header.size())
    );
    if (!input || std::memcmp(fixed_header.data() + 14, "EGEV", 4) != 0) {
        std::cerr << "invalid EGEV header" << std::endl;
        return 1;
    }

    const int version = read_i32_le(fixed_header.data() + 18);
    const int n_phase = read_i32_le(fixed_header.data() + 22);
    const int n_param = read_i32_le(fixed_header.data() + 26);
    const int dim = read_i32_le(fixed_header.data() + 30);
    const int n_group = read_i32_le(fixed_header.data() + 34);
    if (version != 10 || n_phase != N_PHASES || n_param <= 0 ||
        dim <= 0 || dim > 64 || n_group <= 0 || n_group > N_PHASES) {
        std::cerr << "unsupported EGEV10 shape" << std::endl;
        return 1;
    }

    const size_t header_base_size =
        14 + 4 + sizeof(int32_t) * 5 + N_PHASES;
    const size_t header_size =
        header_base_size +
        sizeof(float) * N_PHASES +
        sizeof(float) * static_cast<size_t>(n_group);
    const size_t linear_offset = align_up(header_size, ALIGNMENT);
    const size_t linear_bytes =
        static_cast<size_t>(n_phase) *
        static_cast<size_t>(n_param) *
        sizeof(int16_t);
    const size_t vector_offset =
        align_up(linear_offset + linear_bytes, ALIGNMENT);
    const size_t vector_stride = align_up(static_cast<size_t>(dim), 16);
    const uint64_t vector_count =
        static_cast<uint64_t>(n_group) * static_cast<uint64_t>(n_param);

    input.clear();
    input.seekg(static_cast<std::streamoff>(linear_offset));
    if (!input) {
        std::cerr << "cannot seek to linear payload" << std::endl;
        return 1;
    }

    constexpr std::array<int, 5> linear_limits = {127, 255, 511, 1023, 32767};
    std::array<uint64_t, linear_limits.size()> linear_limit_counts{};
    constexpr size_t linear_values_per_block = 65536;
    std::vector<int16_t> linear_buffer(linear_values_per_block);
    const uint64_t linear_value_count =
        static_cast<uint64_t>(n_phase) * static_cast<uint64_t>(n_param);
    uint64_t linear_values_read = 0;
    int maximum_linear_absolute_value = 0;
    while (linear_values_read < linear_value_count) {
        const size_t values = static_cast<size_t>(
            std::min<uint64_t>(
                linear_values_per_block,
                linear_value_count - linear_values_read
            )
        );
        const size_t bytes = values * sizeof(int16_t);
        input.read(
            reinterpret_cast<char *>(linear_buffer.data()),
            static_cast<std::streamsize>(bytes)
        );
        if (input.gcount() != static_cast<std::streamsize>(bytes)) {
            std::cerr << "truncated linear payload" << std::endl;
            return 1;
        }
        for (size_t value_idx = 0; value_idx < values; ++value_idx) {
            const int value = static_cast<int>(linear_buffer[value_idx]);
            const int absolute_value = value < 0 ? -value : value;
            maximum_linear_absolute_value =
                std::max(maximum_linear_absolute_value, absolute_value);
            for (size_t limit_idx = 0;
                 limit_idx < linear_limits.size();
                 ++limit_idx) {
                linear_limit_counts[limit_idx] +=
                    absolute_value <= linear_limits[limit_idx];
            }
        }
        linear_values_read += values;
    }

    input.clear();
    input.seekg(static_cast<std::streamoff>(vector_offset));
    if (!input) {
        std::cerr << "cannot seek to vector payload" << std::endl;
        return 1;
    }

    constexpr size_t records_per_block = 65536;
    std::vector<int8_t> buffer(records_per_block * vector_stride);
    std::array<uint64_t, 5> entry_limit_counts{};
    std::array<uint64_t, 5> vector_limit_counts{};
    constexpr std::array<int, 5> limits = {7, 15, 31, 63, 127};
    uint64_t vectors_read = 0;
    int maximum_absolute_value = 0;

    while (vectors_read < vector_count) {
        const size_t records = static_cast<size_t>(
            std::min<uint64_t>(records_per_block, vector_count - vectors_read)
        );
        const size_t bytes = records * vector_stride;
        input.read(
            reinterpret_cast<char *>(buffer.data()),
            static_cast<std::streamsize>(bytes)
        );
        if (input.gcount() != static_cast<std::streamsize>(bytes)) {
            std::cerr << "truncated vector payload" << std::endl;
            return 1;
        }

        for (size_t record = 0; record < records; ++record) {
            int vector_maximum = 0;
            const int8_t *vector = buffer.data() + record * vector_stride;
            for (int vector_idx = 0; vector_idx < dim; ++vector_idx) {
                const int value = static_cast<int>(vector[vector_idx]);
                const int absolute_value = value < 0 ? -value : value;
                vector_maximum = std::max(vector_maximum, absolute_value);
                maximum_absolute_value =
                    std::max(maximum_absolute_value, absolute_value);
                for (size_t limit_idx = 0; limit_idx < limits.size(); ++limit_idx) {
                    entry_limit_counts[limit_idx] +=
                        absolute_value <= limits[limit_idx];
                }
            }
            for (size_t limit_idx = 0; limit_idx < limits.size(); ++limit_idx) {
                vector_limit_counts[limit_idx] +=
                    vector_maximum <= limits[limit_idx];
            }
        }
        vectors_read += records;
    }

    const uint64_t entry_count = vector_count * static_cast<uint64_t>(dim);
    std::cout << "version=" << version
              << " phases=" << n_phase
              << " parameters_per_phase=" << n_param
              << " dimension=" << dim
              << " groups=" << n_group
              << " vectors=" << vector_count
              << " entries=" << entry_count
              << " maximum_absolute_value=" << maximum_absolute_value
              << std::endl;
    std::cout << "linear_values=" << linear_value_count
              << " maximum_linear_absolute_value="
              << maximum_linear_absolute_value
              << std::endl;
    for (size_t limit_idx = 0;
         limit_idx < linear_limits.size();
         ++limit_idx) {
        std::cout << "linear_absolute_value_limit="
                  << linear_limits[limit_idx]
                  << " linear_value_count="
                  << linear_limit_counts[limit_idx]
                  << std::endl;
    }
    for (size_t limit_idx = 0; limit_idx < limits.size(); ++limit_idx) {
        std::cout << "absolute_value_limit=" << limits[limit_idx]
                  << " entry_count=" << entry_limit_counts[limit_idx]
                  << " vector_count=" << vector_limit_counts[limit_idx]
                  << std::endl;
    }
    return 0;
}
