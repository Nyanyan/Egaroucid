/*
    Egaroucid Project

    @file prune_egev4_fm_dims.cpp
        Prune quantized EGEV4 FM latent dimensions by per-phase vector energy.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int N_PARAMS_PER_PHASE = 612425;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;

struct Config {
    std::string input;
    std::string output;
    std::string summary;
    std::string timestamp;
    int output_dim = 8;
};

struct Egev4Info {
    std::vector<uint8_t> data;
    std::string timestamp;
    int version = 0;
    int n_phases = 0;
    int params_per_phase = 0;
    int dim = 0;
    size_t linear_scales_offset = 0;
    size_t vector_scales_offset = 0;
    size_t payload_offset = 0;
    size_t phase_stride = 0;
    size_t record_stride = 0;
};

struct PhaseSummary {
    int phase = 0;
    std::vector<int> selected_dims;
    uint64_t total_energy = 0;
    uint64_t kept_energy = 0;
    uint64_t total_nonzero = 0;
    uint64_t kept_nonzero = 0;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message + "\n"
        "usage: prune_egev4_fm_dims --input FILE --output FILE --output-dim 8 "
        "[--summary FILE] [--timestamp now|YYYYMMDDhhmmss]");
}

std::string require_value(int &idx, int argc, char **argv) {
    if (idx + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[idx]);
    }
    ++idx;
    return argv[idx];
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            cfg.input = require_value(i, argc, argv);
        } else if (arg == "--output") {
            cfg.output = require_value(i, argc, argv);
        } else if (arg == "--summary") {
            cfg.summary = require_value(i, argc, argv);
        } else if (arg == "--output-dim") {
            cfg.output_dim = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--timestamp") {
            cfg.timestamp = require_value(i, argc, argv);
        } else if (arg == "--help" || arg == "-h") {
            usage_error("");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }
    if (cfg.input.empty()) {
        usage_error("--input is required");
    }
    if (cfg.output.empty()) {
        usage_error("--output is required");
    }
    if (cfg.output_dim <= 0) {
        usage_error("--output-dim must be positive");
    }
    if (!cfg.timestamp.empty() && cfg.timestamp != "now") {
        if (cfg.timestamp.size() != TIMESTAMP_SIZE ||
            !std::all_of(cfg.timestamp.begin(), cfg.timestamp.end(), [](unsigned char c) { return std::isdigit(c); })) {
            usage_error("--timestamp must be 'now' or 14 digits");
        }
    }
    return cfg;
}

template<typename T>
T read_le(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("read past end of file");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

void write_le(std::vector<uint8_t> &data, size_t offset, const void *src, size_t bytes) {
    if (offset + bytes > data.size()) {
        throw std::runtime_error("write past end of output buffer");
    }
    std::memcpy(data.data() + offset, src, bytes);
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

std::string make_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d%H%M%S");
    return out.str();
}

Egev4Info read_egev4(const std::string &path) {
    Egev4Info info;
    info.data = read_file(path);
    const size_t min_size = TIMESTAMP_SIZE + 4 + 4 * sizeof(int32_t) + N_PHASES * sizeof(float) * 2;
    if (info.data.size() < min_size) {
        throw std::runtime_error("input too short for EGEV4 header: " + path);
    }
    info.timestamp.assign(reinterpret_cast<const char*>(info.data.data()), TIMESTAMP_SIZE);
    if (std::memcmp(info.data.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("unexpected magic in " + path);
    }
    const size_t header_offset = TIMESTAMP_SIZE + 4;
    info.version = read_le<int32_t>(info.data, header_offset);
    info.n_phases = read_le<int32_t>(info.data, header_offset + 4);
    info.params_per_phase = read_le<int32_t>(info.data, header_offset + 8);
    info.dim = read_le<int32_t>(info.data, header_offset + 12);
    if (info.version != VERSION_LINEAR_FM_INT16_INT8) {
        throw std::runtime_error("unsupported EGEV4 version: " + std::to_string(info.version));
    }
    if (info.n_phases != N_PHASES || info.params_per_phase != N_PARAMS_PER_PHASE) {
        throw std::runtime_error("unsupported EGEV4 shape");
    }
    if (info.dim <= 0) {
        throw std::runtime_error("invalid input dimension: " + std::to_string(info.dim));
    }
    info.linear_scales_offset = header_offset + 16;
    info.vector_scales_offset = info.linear_scales_offset + N_PHASES * sizeof(float);
    info.payload_offset = info.vector_scales_offset + N_PHASES * sizeof(float);
    info.record_stride = static_cast<size_t>(2 + info.dim);
    info.phase_stride = static_cast<size_t>(N_PARAMS_PER_PHASE) * info.record_stride;
    const size_t expected_size = info.payload_offset + static_cast<size_t>(N_PHASES) * info.phase_stride;
    if (info.data.size() != expected_size) {
        throw std::runtime_error("input size mismatch: got " + std::to_string(info.data.size()) +
            " expected " + std::to_string(expected_size));
    }
    return info;
}

std::vector<int> select_top_energy_dims(
    const Egev4Info &info,
    int phase,
    int output_dim,
    PhaseSummary &summary
) {
    std::vector<uint64_t> energy(static_cast<size_t>(info.dim), 0);
    std::vector<uint64_t> nonzero(static_cast<size_t>(info.dim), 0);
    const size_t phase_offset = info.payload_offset + static_cast<size_t>(phase) * info.phase_stride;
    for (int param = 0; param < N_PARAMS_PER_PHASE; ++param) {
        const size_t vector_offset = phase_offset + static_cast<size_t>(param) * info.record_stride + 2;
        for (int dim = 0; dim < info.dim; ++dim) {
            const int value = static_cast<int>(static_cast<int8_t>(info.data[vector_offset + dim]));
            energy[static_cast<size_t>(dim)] += static_cast<uint64_t>(value * value);
            nonzero[static_cast<size_t>(dim)] += value != 0;
        }
    }

    std::vector<int> dims(static_cast<size_t>(info.dim));
    std::iota(dims.begin(), dims.end(), 0);
    std::stable_sort(dims.begin(), dims.end(), [&](int lhs, int rhs) {
        const uint64_t le = energy[static_cast<size_t>(lhs)];
        const uint64_t re = energy[static_cast<size_t>(rhs)];
        if (le != re) {
            return le > re;
        }
        return lhs < rhs;
    });
    dims.resize(static_cast<size_t>(output_dim));
    std::sort(dims.begin(), dims.end());

    summary.phase = phase;
    summary.selected_dims = dims;
    summary.total_energy = 0;
    summary.kept_energy = 0;
    summary.total_nonzero = 0;
    summary.kept_nonzero = 0;
    for (int dim = 0; dim < info.dim; ++dim) {
        summary.total_energy += energy[static_cast<size_t>(dim)];
        summary.total_nonzero += nonzero[static_cast<size_t>(dim)];
    }
    for (int dim : dims) {
        summary.kept_energy += energy[static_cast<size_t>(dim)];
        summary.kept_nonzero += nonzero[static_cast<size_t>(dim)];
    }
    return dims;
}

std::vector<uint8_t> prune_egev4(
    const Egev4Info &info,
    const Config &cfg,
    std::vector<PhaseSummary> &summaries
) {
    if (cfg.output_dim >= info.dim) {
        throw std::runtime_error("--output-dim must be smaller than the input dim");
    }
    const size_t output_record_stride = static_cast<size_t>(2 + cfg.output_dim);
    const size_t output_phase_stride = static_cast<size_t>(N_PARAMS_PER_PHASE) * output_record_stride;
    const size_t output_size = info.payload_offset + static_cast<size_t>(N_PHASES) * output_phase_stride;
    std::vector<uint8_t> out(output_size, 0);

    std::memcpy(out.data(), info.data.data(), info.payload_offset);
    const std::string out_timestamp =
        cfg.timestamp == "now" ? make_timestamp_now() :
        (!cfg.timestamp.empty() ? cfg.timestamp : info.timestamp);
    std::memcpy(out.data(), out_timestamp.data(), TIMESTAMP_SIZE);
    const int32_t output_dim_i32 = cfg.output_dim;
    write_le(out, TIMESTAMP_SIZE + 4 + 12, &output_dim_i32, sizeof(output_dim_i32));

    summaries.clear();
    summaries.reserve(N_PHASES);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        PhaseSummary summary;
        const std::vector<int> dims = select_top_energy_dims(info, phase, cfg.output_dim, summary);
        summaries.push_back(summary);

        const size_t input_phase_offset = info.payload_offset + static_cast<size_t>(phase) * info.phase_stride;
        const size_t output_phase_offset = info.payload_offset + static_cast<size_t>(phase) * output_phase_stride;
        for (int param = 0; param < N_PARAMS_PER_PHASE; ++param) {
            const size_t input_record = input_phase_offset + static_cast<size_t>(param) * info.record_stride;
            const size_t output_record = output_phase_offset + static_cast<size_t>(param) * output_record_stride;
            out[output_record] = info.data[input_record];
            out[output_record + 1] = info.data[input_record + 1];
            for (int out_dim = 0; out_dim < cfg.output_dim; ++out_dim) {
                out[output_record + 2 + static_cast<size_t>(out_dim)] =
                    info.data[input_record + 2 + static_cast<size_t>(dims[static_cast<size_t>(out_dim)])];
            }
        }
    }
    return out;
}

void write_file(const fs::path &path, const std::vector<uint8_t> &data) {
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write output: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("failed while writing output: " + path.string());
    }
}

void write_summary(const Config &cfg, const Egev4Info &info, const std::vector<PhaseSummary> &summaries) {
    if (cfg.summary.empty()) {
        return;
    }
    const fs::path path(cfg.summary);
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write summary: " + path.string());
    }
    out << std::setprecision(10);
    out << "EGEV4 FM dimension prune experiment\n";
    out << "input=" << cfg.input << "\n";
    out << "output=" << cfg.output << "\n";
    out << "input_timestamp=" << info.timestamp << "\n";
    out << "input_dim=" << info.dim << "\n";
    out << "output_dim=" << cfg.output_dim << "\n";
    out << "method=per-phase top vector energy dimensions\n";
    out << "note=Linear weights, linear scales, and FM vector scales are copied unchanged. Only selected FM vector dimensions are retained.\n";
    out << "phase\tselected_dims\tkept_energy\ttotal_energy\tkept_energy_ratio\tkept_nonzero\ttotal_nonzero\tkept_nonzero_ratio\n";
    for (const PhaseSummary &s : summaries) {
        out << s.phase << '\t';
        for (size_t i = 0; i < s.selected_dims.size(); ++i) {
            if (i != 0) {
                out << ',';
            }
            out << s.selected_dims[i];
        }
        const double energy_ratio = s.total_energy == 0 ? 0.0 : static_cast<double>(s.kept_energy) / static_cast<double>(s.total_energy);
        const double nonzero_ratio = s.total_nonzero == 0 ? 0.0 : static_cast<double>(s.kept_nonzero) / static_cast<double>(s.total_nonzero);
        out << '\t' << s.kept_energy
            << '\t' << s.total_energy
            << '\t' << energy_ratio
            << '\t' << s.kept_nonzero
            << '\t' << s.total_nonzero
            << '\t' << nonzero_ratio
            << "\n";
    }
}

int main(int argc, char **argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const Egev4Info info = read_egev4(cfg.input);
        std::vector<PhaseSummary> summaries;
        const std::vector<uint8_t> output = prune_egev4(info, cfg, summaries);
        write_file(cfg.output, output);
        write_summary(cfg, info, summaries);

        uint64_t total_energy = 0;
        uint64_t kept_energy = 0;
        for (const PhaseSummary &s : summaries) {
            total_energy += s.total_energy;
            kept_energy += s.kept_energy;
        }
        const double kept_ratio = total_energy == 0 ? 0.0 : static_cast<double>(kept_energy) / static_cast<double>(total_energy);
        std::cout << "wrote " << cfg.output << std::endl;
        std::cout << "input_dim " << info.dim << " output_dim " << cfg.output_dim << std::endl;
        std::cout << "kept_energy_ratio " << std::setprecision(10) << kept_ratio << std::endl;
        if (!cfg.summary.empty()) {
            std::cout << "summary " << cfg.summary << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
