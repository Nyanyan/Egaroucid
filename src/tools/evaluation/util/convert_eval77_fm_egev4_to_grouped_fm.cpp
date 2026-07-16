/*
    Egaroucid Project

    @file convert_eval77_fm_egev4_to_grouped_fm.cpp
        Convert 7.7 beta + FM EGEV4 interleaved layout to grouped-FM EGEV version 10.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
constexpr size_t EGEV4_FIXED_HEADER_SIZE = 34;
constexpr size_t EGEV4_SCALES_SIZE = sizeof(float) * N_PHASES * 2;
constexpr size_t EGEV4_PAYLOAD_OFFSET = EGEV4_FIXED_HEADER_SIZE + EGEV4_SCALES_SIZE;
constexpr size_t GROUPED_ALIGNMENT = 64;
constexpr size_t CHUNK_PARAMS = 8192;

struct InputInfo {
    std::string timestamp;
    int version = 0;
    int dim = 0;
    std::array<float, N_PHASES> linear_scales{};
    std::array<float, N_PHASES> vector_scales{};
    size_t record_stride = 0;
    size_t phase_stride = 0;
};

enum class VectorGroupMode {
    Average,
    CopyFirst,
    CopyLast,
    CopyCustom
};

struct GroupSpec {
    std::array<uint8_t, N_PHASES> phase_to_group{};
    std::array<int, N_PHASES> representative_phase{};
    int group_count = 0;
    VectorGroupMode vector_mode = VectorGroupMode::Average;
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

int signed_byte(const unsigned char value) {
    return value < 128 ? (int)value : (int)value - 256;
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

void write_padding(std::ofstream &out, const size_t target_offset) {
    const size_t current = (size_t)out.tellp();
    if (current > target_offset) {
        throw std::runtime_error("internal offset error while padding");
    }
    std::array<unsigned char, GROUPED_ALIGNMENT> zeros{};
    size_t remain = target_offset - current;
    while (remain > 0) {
        const size_t chunk = std::min(remain, zeros.size());
        out.write((const char*)zeros.data(), (std::streamsize)chunk);
        remain -= chunk;
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

bool parse_grouped_count(const std::string &mode, int *group_count) {
    constexpr const char *prefix = "grouped";
    constexpr size_t prefix_len = 7;
    if (mode.size() <= prefix_len || mode.compare(0, prefix_len, prefix) != 0) {
        return false;
    }
    int value = 0;
    for (size_t i = prefix_len; i < mode.size(); ++i) {
        const char c = mode[i];
        if (c < '0' || '9' < c) {
            return false;
        }
        value = value * 10 + (c - '0');
    }
    *group_count = value;
    return true;
}

VectorGroupMode parse_vector_group_mode(std::string *mode) {
    constexpr const char *copy_first_suffix = "copyfirst";
    constexpr const char *copy_last_suffix = "copylast";
    constexpr const char *copy_custom_suffix = "copycustom";
    const auto strip_suffix = [&](const char *suffix) {
        const size_t suffix_len = std::strlen(suffix);
        if (mode->size() < suffix_len ||
            mode->compare(mode->size() - suffix_len, suffix_len, suffix) != 0) {
            return false;
        }
        mode->resize(mode->size() - suffix_len);
        return true;
    };
    if (strip_suffix(copy_first_suffix)) {
        return VectorGroupMode::CopyFirst;
    }
    if (strip_suffix(copy_last_suffix)) {
        return VectorGroupMode::CopyLast;
    }
    if (strip_suffix(copy_custom_suffix)) {
        return VectorGroupMode::CopyCustom;
    }
    return VectorGroupMode::Average;
}

void assign_default_group_representatives(GroupSpec *spec) {
    spec->representative_phase.fill(0);
    if (spec->vector_mode == VectorGroupMode::CopyCustom) {
        throw std::runtime_error("copycustom requires a representative phase CSV");
    }
    for (int group = 0; group < spec->group_count; ++group) {
        int first_phase = -1;
        int last_phase = -1;
        for (int phase = 0; phase < N_PHASES; ++phase) {
            if (spec->phase_to_group[phase] == group) {
                if (first_phase < 0) {
                    first_phase = phase;
                }
                last_phase = phase;
            }
        }
        if (first_phase < 0) {
            throw std::runtime_error("group " + std::to_string(group) + " has no phases");
        }
        spec->representative_phase[(size_t)group] =
            spec->vector_mode == VectorGroupMode::CopyFirst ? first_phase : last_phase;
    }
}

void assign_custom_group_representatives(GroupSpec *spec, const std::string &representative_arg) {
    spec->representative_phase.fill(0);
    std::stringstream ss(representative_arg);
    std::string token;
    int group = 0;
    while (std::getline(ss, token, ',')) {
        if (group >= spec->group_count) {
            throw std::runtime_error("too many representative phases for copycustom");
        }
        size_t parsed = 0;
        const int phase = std::stoi(token, &parsed);
        if (parsed != token.size()) {
            throw std::runtime_error("invalid representative phase: " + token);
        }
        if (phase < 0 || N_PHASES <= phase) {
            throw std::runtime_error("representative phase is out of range: " + token);
        }
        if (spec->phase_to_group[(size_t)phase] != group) {
            throw std::runtime_error(
                "representative phase " + std::to_string(phase) +
                " does not belong to group " + std::to_string(group)
            );
        }
        spec->representative_phase[(size_t)group] = phase;
        ++group;
    }
    if (group != spec->group_count) {
        throw std::runtime_error("copycustom representative CSV must have one phase per group");
    }
}

void assign_group_representatives(GroupSpec *spec, const std::string &representative_arg) {
    if (spec->vector_mode == VectorGroupMode::CopyCustom) {
        if (representative_arg.empty()) {
            throw std::runtime_error("copycustom requires a representative phase CSV");
        }
        assign_custom_group_representatives(spec, representative_arg);
        return;
    }
    if (!representative_arg.empty()) {
        throw std::runtime_error("representative phase CSV is only valid with copycustom");
    }
    assign_default_group_representatives(spec);
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
    info.timestamp.assign((const char*)header.data(), TIMESTAMP_SIZE);
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

GroupSpec make_group_spec(const std::string &mode, const std::string &representative_arg) {
    GroupSpec spec;
    std::string base_mode = mode;
    spec.vector_mode = parse_vector_group_mode(&base_mode);
    if (base_mode.empty()) {
        throw std::runtime_error("mode is missing before copyfirst/copylast/copycustom suffix");
    }
    if (base_mode == "shared") {
        spec.group_count = 1;
        spec.phase_to_group.fill(0);
        assign_group_representatives(&spec, representative_arg);
        return spec;
    }
    if (base_mode == "grouped7") {
        constexpr std::array<int, 8> starts = {0, 6, 14, 22, 31, 41, 50, 60};
        spec.group_count = 7;
        for (int group = 0; group < spec.group_count; ++group) {
            for (int phase = starts[group]; phase < starts[group + 1]; ++phase) {
                spec.phase_to_group[phase] = (uint8_t)group;
            }
        }
        assign_group_representatives(&spec, representative_arg);
        return spec;
    }
    int group_count = 0;
    if (parse_grouped_count(base_mode, &group_count)) {
        if (group_count < 2 || N_PHASES < group_count) {
            throw std::runtime_error("groupedN requires 2 <= N <= 60");
        }
        spec.group_count = group_count;
        for (int group = 0; group < spec.group_count; ++group) {
            const int start = group * N_PHASES / spec.group_count;
            const int end = (group + 1) * N_PHASES / spec.group_count;
            for (int phase = start; phase < end; ++phase) {
                spec.phase_to_group[phase] = (uint8_t)group;
            }
        }
        assign_group_representatives(&spec, representative_arg);
        return spec;
    }
    throw std::runtime_error("mode must be grouped7, groupedN, or shared, optionally suffixed by copyfirst/copylast/copycustom");
}

std::vector<std::vector<int>> phases_by_group(const GroupSpec &spec) {
    std::vector<std::vector<int>> groups((size_t)spec.group_count);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const int group = spec.phase_to_group[phase];
        if (group < 0 || spec.group_count <= group) {
            throw std::runtime_error("phase_to_group contains an out-of-range group");
        }
        groups[(size_t)group].push_back(phase);
    }
    for (int group = 0; group < spec.group_count; ++group) {
        if (groups[(size_t)group].empty()) {
            throw std::runtime_error("group " + std::to_string(group) + " has no phases");
        }
    }
    return groups;
}

void accumulate_group_chunk(
    std::ifstream &in,
    const InputInfo &info,
    const std::vector<int> &phases,
    const size_t start_param,
    const size_t n_param,
    std::vector<unsigned char> &phase_buffer,
    std::vector<float> &accum
) {
    phase_buffer.resize(n_param * info.record_stride);
    accum.assign(n_param * (size_t)info.dim, 0.0f);
    for (const int phase : phases) {
        const size_t offset = EGEV4_PAYLOAD_OFFSET + (size_t)phase * info.phase_stride +
            start_param * info.record_stride;
        read_exact_at(in, offset, phase_buffer.data(), phase_buffer.size(), "input phase vector chunk");
        const float scale = info.vector_scales[(size_t)phase];
        for (size_t param = 0; param < n_param; ++param) {
            const size_t record_offset = param * info.record_stride + sizeof(int16_t);
            const size_t accum_offset = param * (size_t)info.dim;
            for (int dim = 0; dim < info.dim; ++dim) {
                accum[accum_offset + (size_t)dim] +=
                    (float)signed_byte(phase_buffer[record_offset + (size_t)dim]) * scale;
            }
        }
    }
    const float inv_count = 1.0f / (float)phases.size();
    for (float &value : accum) {
        value *= inv_count;
    }
}

std::array<float, N_PHASES> compute_group_vector_scales(
    const std::string &input_path,
    const InputInfo &info,
    const GroupSpec &spec,
    const std::vector<std::vector<int>> &groups
) {
    std::array<float, N_PHASES> group_scales{};
    if (spec.vector_mode != VectorGroupMode::Average) {
        for (int group = 0; group < spec.group_count; ++group) {
            group_scales[(size_t)group] = info.vector_scales[(size_t)spec.representative_phase[(size_t)group]];
        }
        return group_scales;
    }

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot reopen input: " + input_path);
    }
    std::vector<unsigned char> phase_buffer;
    std::vector<float> accum;

    for (size_t group = 0; group < groups.size(); ++group) {
        double max_abs = 0.0;
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            accumulate_group_chunk(in, info, groups[group], start, n_param, phase_buffer, accum);
            for (const float value : accum) {
                max_abs = std::max(max_abs, std::abs((double)value));
            }
        }

        if (max_abs > 0.0) {
            group_scales[group] = (float)(max_abs / 127.0);
        } else {
            double scale_sum = 0.0;
            for (const int phase : groups[group]) {
                scale_sum += (double)info.vector_scales[(size_t)phase];
            }
            group_scales[group] = (float)(scale_sum / (double)groups[group].size());
        }
    }
    return group_scales;
}

void write_grouped_header(
    std::ofstream &out,
    const std::string &timestamp,
    const InputInfo &info,
    const GroupSpec &spec,
    const std::array<float, N_PHASES> &group_vector_scales
) {
    const size_t header_base_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 5 + N_PHASES;
    const size_t header_size = header_base_size + sizeof(float) * N_PHASES +
        sizeof(float) * (size_t)spec.group_count;
    std::vector<unsigned char> header(header_size, 0);
    std::memcpy(header.data(), timestamp.data(), TIMESTAMP_SIZE);
    std::memcpy(header.data() + TIMESTAMP_SIZE, "EGEV", 4);
    write_i32_le(header.data() + 18, GROUPED_VERSION);
    write_i32_le(header.data() + 22, N_PHASES);
    write_i32_le(header.data() + 26, N_PARAMS_PER_PHASE);
    write_i32_le(header.data() + 30, info.dim);
    write_i32_le(header.data() + 34, spec.group_count);
    std::memcpy(header.data() + 38, spec.phase_to_group.data(), N_PHASES);
    std::memcpy(header.data() + header_base_size, info.linear_scales.data(), sizeof(float) * N_PHASES);
    std::memcpy(header.data() + header_base_size + sizeof(float) * N_PHASES,
                group_vector_scales.data(), sizeof(float) * (size_t)spec.group_count);
    out.write((const char*)header.data(), (std::streamsize)header.size());
}

void write_linear_table(
    std::ifstream &in,
    std::ofstream &out,
    const InputInfo &info
) {
    std::vector<unsigned char> phase_buffer;
    std::vector<unsigned char> linear_buffer;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            phase_buffer.resize(n_param * info.record_stride);
            linear_buffer.resize(n_param * sizeof(int16_t));
            const size_t offset = EGEV4_PAYLOAD_OFFSET + (size_t)phase * info.phase_stride +
                start * info.record_stride;
            read_exact_at(in, offset, phase_buffer.data(), phase_buffer.size(), "input phase linear chunk");
            for (size_t param = 0; param < n_param; ++param) {
                std::memcpy(linear_buffer.data() + param * sizeof(int16_t),
                            phase_buffer.data() + param * info.record_stride,
                            sizeof(int16_t));
            }
            out.write((const char*)linear_buffer.data(), (std::streamsize)linear_buffer.size());
        }
    }
}

void write_vector_table(
    std::ifstream &in,
    std::ofstream &out,
    const InputInfo &info,
    const GroupSpec &spec,
    const std::vector<std::vector<int>> &groups,
    const std::array<float, N_PHASES> &group_vector_scales
) {
    const size_t vector_param_stride = align_up((size_t)info.dim, 16);
    std::vector<unsigned char> phase_buffer;
    std::vector<float> accum;
    std::vector<unsigned char> output_buffer;

    for (size_t group = 0; group < groups.size(); ++group) {
        const float group_scale = group_vector_scales[group];
        if (!(group_scale > 0.0f)) {
            throw std::runtime_error("group vector scale must be positive");
        }
        if (spec.vector_mode != VectorGroupMode::Average) {
            const int phase = spec.representative_phase[group];
            for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
                const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
                phase_buffer.resize(n_param * info.record_stride);
                output_buffer.assign(n_param * vector_param_stride, 0);
                const size_t offset = EGEV4_PAYLOAD_OFFSET + (size_t)phase * info.phase_stride +
                    start * info.record_stride;
                read_exact_at(in, offset, phase_buffer.data(), phase_buffer.size(),
                              "representative phase vector chunk");
                for (size_t param = 0; param < n_param; ++param) {
                    std::memcpy(output_buffer.data() + param * vector_param_stride,
                                phase_buffer.data() + param * info.record_stride + sizeof(int16_t),
                                (size_t)info.dim);
                }
                out.write((const char*)output_buffer.data(), (std::streamsize)output_buffer.size());
            }
            continue;
        }
        for (size_t start = 0; start < (size_t)N_PARAMS_PER_PHASE; start += CHUNK_PARAMS) {
            const size_t n_param = std::min(CHUNK_PARAMS, (size_t)N_PARAMS_PER_PHASE - start);
            accumulate_group_chunk(in, info, groups[group], start, n_param, phase_buffer, accum);
            output_buffer.assign(n_param * vector_param_stride, 0);
            for (size_t param = 0; param < n_param; ++param) {
                const size_t accum_offset = param * (size_t)info.dim;
                const size_t output_offset = param * vector_param_stride;
                for (int dim = 0; dim < info.dim; ++dim) {
                    int quant = (int)std::lrint((double)accum[accum_offset + (size_t)dim] / (double)group_scale);
                    quant = std::clamp(quant, -128, 127);
                    output_buffer[output_offset + (size_t)dim] = (unsigned char)(int8_t)quant;
                }
            }
            out.write((const char*)output_buffer.data(), (std::streamsize)output_buffer.size());
        }
    }
}

void convert(
    const std::string &input_path,
    const std::string &output_path,
    const std::string &mode,
    const std::string &timestamp_arg,
    const std::string &representative_arg
) {
    const InputInfo info = read_input_info(input_path);
    const GroupSpec spec = make_group_spec(mode, representative_arg);
    const std::vector<std::vector<int>> groups = phases_by_group(spec);
    const std::array<float, N_PHASES> group_vector_scales =
        compute_group_vector_scales(input_path, info, spec, groups);

    std::string timestamp = info.timestamp;
    if (!timestamp_arg.empty() && timestamp_arg != "preserve") {
        timestamp = timestamp_arg == "now" ? timestamp_now() : timestamp_arg;
    }
    if (!valid_timestamp(timestamp)) {
        throw std::runtime_error("timestamp must be 14 digits, 'now', or 'preserve'");
    }

    const size_t header_base_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 5 + N_PHASES;
    const size_t header_size = header_base_size + sizeof(float) * N_PHASES +
        sizeof(float) * (size_t)spec.group_count;
    const size_t linear_offset = align_up(header_size, GROUPED_ALIGNMENT);
    const size_t linear_phase_stride = (size_t)N_PARAMS_PER_PHASE * sizeof(int16_t);
    const size_t vector_param_stride = align_up((size_t)info.dim, 16);
    const size_t vector_phase_stride = (size_t)N_PARAMS_PER_PHASE * vector_param_stride;
    const size_t vector_offset = align_up(
        linear_offset + (size_t)N_PHASES * linear_phase_stride,
        GROUPED_ALIGNMENT
    );
    const size_t output_size = vector_offset + (size_t)spec.group_count * vector_phase_stride;

    std::ifstream in(input_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot reopen input: " + input_path);
    }
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open output: " + output_path);
    }

    write_grouped_header(out, timestamp, info, spec, group_vector_scales);
    write_padding(out, linear_offset);
    write_linear_table(in, out, info);
    write_padding(out, vector_offset);
    write_vector_table(in, out, info, spec, groups, group_vector_scales);
    if (!out) {
        throw std::runtime_error("failed while writing output");
    }

    std::cerr << "converted " << input_path << " -> " << output_path << "\n"
              << "input_version " << info.version
              << " output_version " << GROUPED_VERSION
              << " mode " << mode
              << " phases " << N_PHASES
              << " params/phase " << N_PARAMS_PER_PHASE
              << " dim " << info.dim
              << " fm_groups " << spec.group_count << "\n"
              << "linear_offset " << linear_offset
              << " vector_offset " << vector_offset
              << " output_bytes " << output_size << std::endl;
    for (int group = 0; group < spec.group_count; ++group) {
        std::cerr << "group " << group << " phases";
        for (const int phase : groups[(size_t)group]) {
            std::cerr << ' ' << phase;
        }
        std::cerr << " vector_scale " << group_vector_scales[(size_t)group];
        if (spec.vector_mode != VectorGroupMode::Average) {
            std::cerr << " representative_phase " << spec.representative_phase[(size_t)group];
        }
        std::cerr << std::endl;
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: convert_eval77_fm_egev4_to_grouped_fm <input.egev4> <output.egev10> <grouped7|groupedN|shared>[copyfirst|copylast|copycustom] [preserve|now|YYYYMMDDHHMMSS] [representative_phase_csv_for_copycustom]" << std::endl;
        return 1;
    }
    try {
        const std::string timestamp = argc >= 5 ? argv[4] : "preserve";
        const std::string representative_arg = argc >= 6 ? argv[5] : "";
        convert(argv[1], argv[2], argv[3], timestamp, representative_arg);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
