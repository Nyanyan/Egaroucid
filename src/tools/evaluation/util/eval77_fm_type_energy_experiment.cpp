/*
    Egaroucid Project

    @file eval77_fm_type_energy_experiment.cpp
        Summarize 7.7beta EGEV4 FM vector energy by pattern type.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int N_PATTERN_TYPES = 16;
constexpr int N_ORIENTATIONS_PER_TYPE = 4;
constexpr int N_PATTERN_FEATURES = N_PATTERN_TYPES * N_ORIENTATIONS_PER_TYPE;
constexpr int N_PATTERN_PARAMS_RAW = 944784;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + 65;
constexpr int PATTERN_PARAMS_PER_TYPE = 59049;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;

template<typename T>
T read_le(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("read past end of file");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

std::vector<uint8_t> read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open input: " + path.string());
    }
    const std::streamsize size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("empty input: " + path.string());
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) {
        throw std::runtime_error("cannot read input: " + path.string());
    }
    return data;
}

struct Egev4Info {
    std::vector<uint8_t> data;
    int n_phases = 0;
    int params_per_phase = 0;
    int dim = 0;
    size_t payload_offset = 0;
    size_t record_stride = 0;
    size_t phase_stride = 0;
};

Egev4Info read_egev4(const std::string &path) {
    Egev4Info info;
    info.data = read_file(path);
    if (info.data.size() < TIMESTAMP_SIZE + 4 + 4 * sizeof(int32_t) + N_PHASES * sizeof(float) * 2) {
        throw std::runtime_error("input too short: " + path);
    }
    if (std::memcmp(info.data.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("unexpected magic in " + path);
    }
    const size_t header_offset = TIMESTAMP_SIZE + 4;
    const int version = read_le<int32_t>(info.data, header_offset);
    info.n_phases = read_le<int32_t>(info.data, header_offset + 4);
    info.params_per_phase = read_le<int32_t>(info.data, header_offset + 8);
    info.dim = read_le<int32_t>(info.data, header_offset + 12);
    if (version != VERSION_LINEAR_FM_INT16_INT8 || info.n_phases != N_PHASES ||
        info.params_per_phase != N_PARAMS_PER_PHASE || info.dim <= 0) {
        throw std::runtime_error("unsupported 7.7beta EGEV4 shape");
    }
    const size_t linear_scales_offset = header_offset + 16;
    const size_t vector_scales_offset = linear_scales_offset + N_PHASES * sizeof(float);
    info.payload_offset = vector_scales_offset + N_PHASES * sizeof(float);
    info.record_stride = static_cast<size_t>(2 + info.dim);
    info.phase_stride = static_cast<size_t>(info.params_per_phase) * info.record_stride;
    const size_t expected_size = info.payload_offset + static_cast<size_t>(N_PHASES) * info.phase_stride;
    if (info.data.size() != expected_size) {
        throw std::runtime_error("input size mismatch");
    }
    return info;
}

uint64_t vector_energy_at(const Egev4Info &info, int phase, int param_id) {
    const size_t offset = info.payload_offset +
        static_cast<size_t>(phase) * info.phase_stride +
        static_cast<size_t>(param_id) * info.record_stride + 2;
    uint64_t energy = 0;
    for (int dim = 0; dim < info.dim; ++dim) {
        const int value = static_cast<int>(static_cast<int8_t>(info.data[offset + dim]));
        energy += static_cast<uint64_t>(value * value);
    }
    return energy;
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: eval77_fm_type_energy_experiment <eval.egev4> [phase_min] [phase_max]" << std::endl;
            return 1;
        }
        const std::string path = argv[1];
        const int phase_min = argc >= 3 ? std::stoi(argv[2]) : 0;
        const int phase_max = argc >= 4 ? std::stoi(argv[3]) : N_PHASES - 1;
        if (phase_min < 0 || phase_max >= N_PHASES || phase_max < phase_min) {
            throw std::runtime_error("invalid phase range");
        }

        const Egev4Info info = read_egev4(path);
        std::array<uint64_t, N_PATTERN_TYPES> type_energy{};
        std::array<uint64_t, N_PATTERN_TYPES> type_nonzero{};
        uint64_t count_energy = 0;
        uint64_t count_nonzero = 0;

        for (int phase = phase_min; phase <= phase_max; ++phase) {
            for (int type = 0; type < N_PATTERN_TYPES; ++type) {
                const int begin = type * PATTERN_PARAMS_PER_TYPE;
                const int end = begin + PATTERN_PARAMS_PER_TYPE;
                for (int param_id = begin; param_id < end; ++param_id) {
                    const uint64_t energy = vector_energy_at(info, phase, param_id);
                    type_energy[type] += energy;
                    type_nonzero[type] += energy != 0;
                }
            }
            for (int param_id = N_PATTERN_PARAMS_RAW; param_id < N_PARAMS_PER_PHASE; ++param_id) {
                const uint64_t energy = vector_energy_at(info, phase, param_id);
                count_energy += energy;
                count_nonzero += energy != 0;
            }
        }

        uint64_t total_pattern_energy = 0;
        uint64_t total_pattern_nonzero = 0;
        for (int type = 0; type < N_PATTERN_TYPES; ++type) {
            total_pattern_energy += type_energy[type];
            total_pattern_nonzero += type_nonzero[type];
        }

        std::cout << std::setprecision(10);
        std::cout << "input=" << path << "\n";
        std::cout << "phase_min=" << phase_min << "\n";
        std::cout << "phase_max=" << phase_max << "\n";
        std::cout << "dim=" << info.dim << "\n";
        std::cout << "total_pattern_energy=" << total_pattern_energy << "\n";
        std::cout << "total_pattern_nonzero=" << total_pattern_nonzero << "\n";
        std::cout << "count_energy=" << count_energy << "\n";
        std::cout << "count_nonzero=" << count_nonzero << "\n";
        std::cout << "type\tenergy\tenergy_ratio\tnonzero\tnonzero_ratio\n";
        for (int type = 0; type < N_PATTERN_TYPES; ++type) {
            const double energy_ratio = total_pattern_energy == 0 ? 0.0 :
                static_cast<double>(type_energy[type]) / static_cast<double>(total_pattern_energy);
            const double nonzero_ratio = total_pattern_nonzero == 0 ? 0.0 :
                static_cast<double>(type_nonzero[type]) / static_cast<double>(total_pattern_nonzero);
            std::cout << type << '\t'
                      << type_energy[type] << '\t'
                      << energy_ratio << '\t'
                      << type_nonzero[type] << '\t'
                      << nonzero_ratio << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
