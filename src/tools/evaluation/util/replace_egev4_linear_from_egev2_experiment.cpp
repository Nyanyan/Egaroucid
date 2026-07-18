/*
    Egaroucid Project

    @file replace_egev4_linear_from_egev2_experiment.cpp
        Replace EGEV4 linear weights with an EGEV2 linear model while retaining
        every FM vector and FM vector scale from the EGEV4 input.
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;
constexpr int N_ZEROS_PLUS = 1 << 12;
constexpr float EGEV2_LINEAR_SCALE = 1.0F / 32.0F;

struct Config {
    fs::path input_egev2;
    fs::path input_egev4;
    fs::path output;
    fs::path summary;
    std::string timestamp;
};

struct Egev4Header {
    std::vector<uint8_t> bytes;
    std::string input_timestamp;
    std::string output_timestamp;
    int version = 0;
    int n_phases = 0;
    int params_per_phase = 0;
    int dim = 0;
    size_t linear_scales_offset = 0;
    size_t payload_offset = 0;
    size_t record_stride = 0;
    size_t phase_stride = 0;
    uint64_t expected_file_size = 0;
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(
        message +
        "\nusage: replace_egev4_linear_from_egev2_experiment"
        " --input-egev2 FILE --input-egev4 FILE --output FILE"
        " [--summary FILE] [--timestamp now|YYYYMMDDhhmmss]"
    );
}

std::string require_value(int &index, int argc, char **argv) {
    if (index + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-egev2") {
            cfg.input_egev2 = require_value(i, argc, argv);
        } else if (arg == "--input-egev4") {
            cfg.input_egev4 = require_value(i, argc, argv);
        } else if (arg == "--output") {
            cfg.output = require_value(i, argc, argv);
        } else if (arg == "--summary") {
            cfg.summary = require_value(i, argc, argv);
        } else if (arg == "--timestamp") {
            cfg.timestamp = require_value(i, argc, argv);
        } else if (arg == "--help" || arg == "-h") {
            usage_error("");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }
    if (cfg.input_egev2.empty()) {
        usage_error("--input-egev2 is required");
    }
    if (cfg.input_egev4.empty()) {
        usage_error("--input-egev4 is required");
    }
    if (cfg.output.empty()) {
        usage_error("--output is required");
    }
    if (!cfg.timestamp.empty() && cfg.timestamp != "now") {
        if (cfg.timestamp.size() != TIMESTAMP_SIZE ||
            !std::all_of(cfg.timestamp.begin(), cfg.timestamp.end(), [](unsigned char c) {
                return std::isdigit(c);
            })) {
            usage_error("--timestamp must be 'now' or 14 digits");
        }
    }
    return cfg;
}

template<typename T>
T read_value(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("read past end of header");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

std::string make_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d%H%M%S");
    return out.str();
}

std::vector<int16_t> load_unzip_egev2(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open EGEV2 input: " + path.string());
    }

    int32_t n_compressed = 0;
    in.read(reinterpret_cast<char *>(&n_compressed), sizeof(n_compressed));
    if (!in || n_compressed <= 0) {
        throw std::runtime_error("invalid EGEV2 compressed-value count");
    }

    std::vector<int16_t> values;
    values.reserve(static_cast<size_t>(n_compressed));
    for (int32_t i = 0; i < n_compressed; ++i) {
        int16_t value = 0;
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in) {
            throw std::runtime_error("EGEV2 input ended before its declared value count");
        }
        if (value >= N_ZEROS_PLUS) {
            values.insert(values.end(), static_cast<size_t>(value - N_ZEROS_PLUS), 0);
        } else {
            values.push_back(value);
        }
    }
    if (in.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("EGEV2 input contains bytes after its declared value count");
    }
    return values;
}

Egev4Header read_egev4_header(std::ifstream &in, const fs::path &path, const std::string &timestamp) {
    constexpr size_t fixed_header_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 4;
    constexpr size_t scales_size = sizeof(float) * N_PHASES * 2;
    constexpr size_t header_size = fixed_header_size + scales_size;

    Egev4Header header;
    header.bytes.resize(header_size);
    in.read(reinterpret_cast<char *>(header.bytes.data()), static_cast<std::streamsize>(header.bytes.size()));
    if (!in) {
        throw std::runtime_error("EGEV4 input is too short for its header: " + path.string());
    }
    header.input_timestamp.assign(
        reinterpret_cast<const char *>(header.bytes.data()),
        TIMESTAMP_SIZE
    );
    if (std::memcmp(header.bytes.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("unexpected EGEV4 magic");
    }

    const size_t fields_offset = TIMESTAMP_SIZE + 4;
    header.version = read_value<int32_t>(header.bytes, fields_offset);
    header.n_phases = read_value<int32_t>(header.bytes, fields_offset + 4);
    header.params_per_phase = read_value<int32_t>(header.bytes, fields_offset + 8);
    header.dim = read_value<int32_t>(header.bytes, fields_offset + 12);
    if (header.version != VERSION_LINEAR_FM_INT16_INT8) {
        throw std::runtime_error("unsupported EGEV4 version: " + std::to_string(header.version));
    }
    if (header.n_phases != N_PHASES || header.params_per_phase <= 0 || header.dim <= 0) {
        throw std::runtime_error("unsupported EGEV4 shape");
    }

    header.linear_scales_offset = fixed_header_size;
    header.payload_offset = header_size;
    header.record_stride = static_cast<size_t>(2 + header.dim);
    header.phase_stride = static_cast<size_t>(header.params_per_phase) * header.record_stride;
    header.expected_file_size =
        static_cast<uint64_t>(header.payload_offset) +
        static_cast<uint64_t>(header.n_phases) * static_cast<uint64_t>(header.phase_stride);

    header.output_timestamp =
        timestamp == "now" ? make_timestamp_now() :
        (!timestamp.empty() ? timestamp : header.input_timestamp);
    std::memcpy(header.bytes.data(), header.output_timestamp.data(), TIMESTAMP_SIZE);
    const std::array<float, N_PHASES> linear_scales = [] {
        std::array<float, N_PHASES> scales{};
        scales.fill(EGEV2_LINEAR_SCALE);
        return scales;
    }();
    std::memcpy(
        header.bytes.data() + header.linear_scales_offset,
        linear_scales.data(),
        sizeof(linear_scales)
    );
    return header;
}

void write_summary(
    const Config &cfg,
    const Egev4Header &header,
    size_t linear_value_count,
    size_t changed_linear_value_count,
    size_t nonzero_linear_value_count,
    int16_t min_linear_value,
    int16_t max_linear_value
) {
    if (cfg.summary.empty()) {
        return;
    }
    if (!cfg.summary.parent_path().empty()) {
        fs::create_directories(cfg.summary.parent_path());
    }
    std::ofstream out(cfg.summary);
    if (!out) {
        throw std::runtime_error("cannot open summary output: " + cfg.summary.string());
    }
    out << "EGEV4 linear-weight replacement experiment\n";
    out << "input_egev2=" << cfg.input_egev2.string() << "\n";
    out << "input_egev4=" << cfg.input_egev4.string() << "\n";
    out << "output=" << cfg.output.string() << "\n";
    out << "input_timestamp=" << header.input_timestamp << "\n";
    out << "output_timestamp=" << header.output_timestamp << "\n";
    out << "version=" << header.version << "\n";
    out << "n_phases=" << header.n_phases << "\n";
    out << "params_per_phase=" << header.params_per_phase << "\n";
    out << "dim=" << header.dim << "\n";
    out << "linear_scale=" << std::setprecision(10) << EGEV2_LINEAR_SCALE << "\n";
    out << "linear_value_count=" << linear_value_count << "\n";
    out << "changed_linear_value_count=" << changed_linear_value_count << "\n";
    out << "nonzero_linear_value_count=" << nonzero_linear_value_count << "\n";
    out << "min_linear_value=" << min_linear_value << "\n";
    out << "max_linear_value=" << max_linear_value << "\n";
    out << "note=All FM vectors and all FM vector scales are copied byte-for-byte from input_egev4.\n";
}

void replace_linear_weights(const Config &cfg) {
    const std::vector<int16_t> linear_values = load_unzip_egev2(cfg.input_egev2);

    std::ifstream input(cfg.input_egev4, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open EGEV4 input: " + cfg.input_egev4.string());
    }
    const std::streamoff input_size = input.tellg();
    input.seekg(0);
    Egev4Header header = read_egev4_header(input, cfg.input_egev4, cfg.timestamp);
    if (input_size < 0 || static_cast<uint64_t>(input_size) != header.expected_file_size) {
        throw std::runtime_error(
            "EGEV4 input size mismatch: got " + std::to_string(input_size) +
            " expected " + std::to_string(header.expected_file_size)
        );
    }

    const size_t expected_linear_values =
        static_cast<size_t>(header.n_phases) * static_cast<size_t>(header.params_per_phase);
    if (linear_values.size() != expected_linear_values) {
        throw std::runtime_error(
            "EGEV2 expanded-value count mismatch: got " + std::to_string(linear_values.size()) +
            " expected " + std::to_string(expected_linear_values)
        );
    }

    if (!cfg.output.parent_path().empty()) {
        fs::create_directories(cfg.output.parent_path());
    }
    std::ofstream output(cfg.output, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot open EGEV4 output: " + cfg.output.string());
    }
    output.write(
        reinterpret_cast<const char *>(header.bytes.data()),
        static_cast<std::streamsize>(header.bytes.size())
    );

    size_t changed_linear_values = 0;
    size_t nonzero_linear_values = 0;
    int16_t min_linear_value = linear_values.front();
    int16_t max_linear_value = linear_values.front();
    std::vector<uint8_t> phase_data(header.phase_stride);
    size_t linear_index = 0;
    for (int phase = 0; phase < header.n_phases; ++phase) {
        input.read(
            reinterpret_cast<char *>(phase_data.data()),
            static_cast<std::streamsize>(phase_data.size())
        );
        if (!input) {
            throw std::runtime_error("EGEV4 input ended while reading phase " + std::to_string(phase));
        }
        for (int param = 0; param < header.params_per_phase; ++param, ++linear_index) {
            const size_t offset = static_cast<size_t>(param) * header.record_stride;
            int16_t old_value = 0;
            std::memcpy(&old_value, phase_data.data() + offset, sizeof(old_value));
            const int16_t new_value = linear_values[linear_index];
            changed_linear_values += old_value != new_value;
            nonzero_linear_values += new_value != 0;
            min_linear_value = std::min(min_linear_value, new_value);
            max_linear_value = std::max(max_linear_value, new_value);
            std::memcpy(phase_data.data() + offset, &new_value, sizeof(new_value));
        }
        output.write(
            reinterpret_cast<const char *>(phase_data.data()),
            static_cast<std::streamsize>(phase_data.size())
        );
        if (!output) {
            throw std::runtime_error("failed while writing output phase " + std::to_string(phase));
        }
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed while closing EGEV4 output");
    }

    write_summary(
        cfg,
        header,
        linear_values.size(),
        changed_linear_values,
        nonzero_linear_values,
        min_linear_value,
        max_linear_value
    );
    std::cout
        << "wrote " << cfg.output.string() << "\n"
        << "version=" << header.version
        << " phases=" << header.n_phases
        << " params_per_phase=" << header.params_per_phase
        << " dim=" << header.dim << "\n"
        << "linear_value_count=" << linear_values.size()
        << " changed_linear_value_count=" << changed_linear_values
        << " nonzero_linear_value_count=" << nonzero_linear_values
        << " min_linear_value=" << min_linear_value
        << " max_linear_value=" << max_linear_value << "\n";
}

int main(int argc, char **argv) {
    try {
        replace_linear_weights(parse_args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
