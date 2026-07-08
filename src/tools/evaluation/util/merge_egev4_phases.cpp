/*
    Egaroucid Project

    @file merge_egev4_phases.cpp
        Merge selected phase payloads from EGEV4 FM files with identical shape.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;

struct Config {
    std::string base;
    std::string output;
    std::string summary;
    std::string timestamp;
    std::vector<std::pair<std::vector<int>, std::string>> phase_sources;
};

struct Egev4Info {
    std::string path;
    int version = 0;
    int n_phases = 0;
    int params_per_phase = 0;
    int dim = 0;
    size_t linear_scales_offset = 0;
    size_t vector_scales_offset = 0;
    size_t payload_offset = 0;
    size_t record_stride = 0;
    size_t phase_stride = 0;
    size_t expected_size = 0;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(message + "\n"
        "usage: merge_egev4_phases --base FILE --output FILE "
        "--phase-source 11-13=FILE [--phase-source 20=FILE ...] "
        "[--summary FILE] [--timestamp now|YYYYMMDDhhmmss]");
}

std::string require_value(int &idx, int argc, char **argv) {
    if (idx + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[idx]);
    }
    ++idx;
    return argv[idx];
}

std::vector<int> parse_int_list(const std::string &text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string elem;
    while (std::getline(ss, elem, ',')) {
        if (elem.empty()) {
            continue;
        }
        const size_t dash = elem.find('-');
        if (dash == std::string::npos) {
            values.push_back(std::stoi(elem));
            continue;
        }
        const int start = std::stoi(elem.substr(0, dash));
        const int end = std::stoi(elem.substr(dash + 1));
        if (end < start) {
            usage_error("invalid phase range: " + elem);
        }
        for (int phase = start; phase <= end; ++phase) {
            values.push_back(phase);
        }
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::pair<std::vector<int>, std::string> parse_phase_source(const std::string &text) {
    const size_t eq = text.find('=');
    if (eq == std::string::npos) {
        usage_error("--phase-source must be PHASES=FILE");
    }
    std::vector<int> phases = parse_int_list(text.substr(0, eq));
    if (phases.empty()) {
        usage_error("--phase-source has no phases: " + text);
    }
    for (int phase : phases) {
        if (phase < 0 || N_PHASES <= phase) {
            usage_error("phase out of range: " + std::to_string(phase));
        }
    }
    std::string file = text.substr(eq + 1);
    if (file.empty()) {
        usage_error("--phase-source file is empty");
    }
    return {std::move(phases), std::move(file)};
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--base") {
            cfg.base = require_value(i, argc, argv);
        } else if (arg == "--output") {
            cfg.output = require_value(i, argc, argv);
        } else if (arg == "--summary") {
            cfg.summary = require_value(i, argc, argv);
        } else if (arg == "--timestamp") {
            cfg.timestamp = require_value(i, argc, argv);
        } else if (arg == "--phase-source") {
            cfg.phase_sources.push_back(parse_phase_source(require_value(i, argc, argv)));
        } else if (arg == "--help" || arg == "-h") {
            usage_error("");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }
    if (cfg.base.empty()) {
        usage_error("--base is required");
    }
    if (cfg.output.empty()) {
        usage_error("--output is required");
    }
    if (cfg.phase_sources.empty()) {
        usage_error("at least one --phase-source is required");
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
T read_le_at(std::ifstream &in, const size_t offset) {
    T value{};
    in.seekg(static_cast<std::streamoff>(offset));
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("cannot read EGEV4 header");
    }
    return value;
}

Egev4Info inspect_egev4(const std::string &path) {
    Egev4Info info;
    info.path = path;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input: " + path);
    }
    char magic[4]{};
    in.seekg(TIMESTAMP_SIZE);
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, "EGEV", 4) != 0) {
        throw std::runtime_error("unexpected EGEV4 magic: " + path);
    }
    const size_t header_offset = TIMESTAMP_SIZE + 4;
    info.version = read_le_at<int32_t>(in, header_offset);
    info.n_phases = read_le_at<int32_t>(in, header_offset + 4);
    info.params_per_phase = read_le_at<int32_t>(in, header_offset + 8);
    info.dim = read_le_at<int32_t>(in, header_offset + 12);
    if (info.version != VERSION_LINEAR_FM_INT16_INT8 || info.n_phases != N_PHASES ||
        info.params_per_phase <= 0 || info.dim <= 0) {
        throw std::runtime_error("unsupported EGEV4 shape: " + path);
    }
    info.linear_scales_offset = header_offset + 16;
    info.vector_scales_offset = info.linear_scales_offset + static_cast<size_t>(info.n_phases) * sizeof(float);
    info.payload_offset = info.vector_scales_offset + static_cast<size_t>(info.n_phases) * sizeof(float);
    info.record_stride = static_cast<size_t>(2 + info.dim);
    info.phase_stride = static_cast<size_t>(info.params_per_phase) * info.record_stride;
    info.expected_size = info.payload_offset + static_cast<size_t>(info.n_phases) * info.phase_stride;
    const size_t actual_size = static_cast<size_t>(fs::file_size(path));
    if (actual_size != info.expected_size) {
        throw std::runtime_error("EGEV4 file size mismatch for " + path +
            ": got " + std::to_string(actual_size) +
            " expected " + std::to_string(info.expected_size));
    }
    return info;
}

void require_same_shape(const Egev4Info &base, const Egev4Info &src) {
    if (base.version != src.version || base.n_phases != src.n_phases ||
        base.params_per_phase != src.params_per_phase || base.dim != src.dim ||
        base.expected_size != src.expected_size) {
        throw std::runtime_error("EGEV4 shape mismatch between " + base.path + " and " + src.path);
    }
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

void copy_exact(std::ifstream &src, std::fstream &dst, const size_t src_offset, const size_t dst_offset, std::vector<char> &buffer) {
    src.seekg(static_cast<std::streamoff>(src_offset));
    dst.seekp(static_cast<std::streamoff>(dst_offset));
    src.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (!src) {
        throw std::runtime_error("cannot read source phase payload");
    }
    dst.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (!dst) {
        throw std::runtime_error("cannot write output phase payload");
    }
}

void merge_phase_source(const Egev4Info &base, const std::vector<int> &phases, const std::string &source_path, std::fstream &out) {
    const Egev4Info src_info = inspect_egev4(source_path);
    require_same_shape(base, src_info);
    std::ifstream src(source_path, std::ios::binary);
    if (!src) {
        throw std::runtime_error("cannot reopen source: " + source_path);
    }
    std::vector<char> scale_buffer(sizeof(float));
    std::vector<char> payload_buffer(base.phase_stride);
    for (int phase : phases) {
        const size_t linear_scale_offset = base.linear_scales_offset + static_cast<size_t>(phase) * sizeof(float);
        const size_t vector_scale_offset = base.vector_scales_offset + static_cast<size_t>(phase) * sizeof(float);
        const size_t payload_offset = base.payload_offset + static_cast<size_t>(phase) * base.phase_stride;
        copy_exact(src, out, linear_scale_offset, linear_scale_offset, scale_buffer);
        copy_exact(src, out, vector_scale_offset, vector_scale_offset, scale_buffer);
        copy_exact(src, out, payload_offset, payload_offset, payload_buffer);
    }
}

void write_timestamp(std::fstream &out, const std::string &timestamp) {
    if (timestamp.empty()) {
        return;
    }
    const std::string actual = timestamp == "now" ? make_timestamp_now() : timestamp;
    out.seekp(0);
    out.write(actual.data(), TIMESTAMP_SIZE);
    if (!out) {
        throw std::runtime_error("cannot write output timestamp");
    }
}

void write_summary(const Config &cfg, const Egev4Info &base) {
    if (cfg.summary.empty()) {
        return;
    }
    fs::path summary_path(cfg.summary);
    if (!summary_path.parent_path().empty()) {
        fs::create_directories(summary_path.parent_path());
    }
    std::ofstream out(summary_path);
    if (!out) {
        throw std::runtime_error("cannot write summary: " + summary_path.string());
    }
    out << "EGEV4 phase merge\n";
    out << "base=" << cfg.base << "\n";
    out << "output=" << cfg.output << "\n";
    out << "version=" << base.version << "\n";
    out << "n_phases=" << base.n_phases << "\n";
    out << "params_per_phase=" << base.params_per_phase << "\n";
    out << "dim=" << base.dim << "\n";
    out << "phase_stride=" << base.phase_stride << "\n";
    out << "timestamp=" << (cfg.timestamp.empty() ? "copied_from_base" : cfg.timestamp) << "\n";
    for (const auto &[phases, source] : cfg.phase_sources) {
        out << "phase_source=";
        for (size_t i = 0; i < phases.size(); ++i) {
            if (i != 0) {
                out << ',';
            }
            out << phases[i];
        }
        out << '=' << source << "\n";
    }
}

int main(int argc, char **argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const Egev4Info base = inspect_egev4(cfg.base);
        if (!fs::path(cfg.output).parent_path().empty()) {
            fs::create_directories(fs::path(cfg.output).parent_path());
        }
        fs::copy_file(cfg.base, cfg.output, fs::copy_options::overwrite_existing);
        std::fstream out(cfg.output, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            throw std::runtime_error("cannot open output for merge: " + cfg.output);
        }
        for (const auto &[phases, source] : cfg.phase_sources) {
            merge_phase_source(base, phases, source, out);
        }
        write_timestamp(out, cfg.timestamp);
        out.close();
        write_summary(cfg, base);
        std::cout << "merged " << cfg.output << std::endl;
        std::cout << "dim " << base.dim << " params_per_phase " << base.params_per_phase << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
