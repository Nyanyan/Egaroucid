/*
    Egaroucid Project

    @file blend_egevfm.cpp
        Average FM vectors from compatible egevfm files
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr char EVAL_FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr uint32_t EVAL_FM_FILE_VERSION_LOCAL = 1;
constexpr size_t EVAL_FM_HEADER_SIZE = 56;

template<typename T>
bool read_scalar(const std::vector<char> &data, size_t *offset, T *value) {
    if (*offset + sizeof(T) > data.size()) {
        return false;
    }
    std::memcpy(value, data.data() + *offset, sizeof(T));
    *offset += sizeof(T);
    return true;
}

struct EgevfmFile {
    std::string path;
    std::vector<char> bytes;
    uint32_t version = 0;
    uint32_t n_phases = 0;
    uint32_t linear_params_per_phase = 0;
    uint32_t n_fm_phases = 0;
    uint32_t n_features = 0;
    uint32_t dim = 0;
    int32_t scale = 0;
    uint32_t flags = 0;
    uint64_t linear_count = 0;
    uint64_t fm_count = 0;
    size_t linear_offset = EVAL_FM_HEADER_SIZE;
    size_t fm_offset = 0;
};

bool load_egevfm(const std::string &path, EgevfmFile *res) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << path << "\n";
        return false;
    }
    res->path = path;
    res->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (res->bytes.size() < EVAL_FM_HEADER_SIZE ||
        std::memcmp(res->bytes.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        std::cerr << "[ERROR] input is not egevfm v1: " << path << "\n";
        return false;
    }

    size_t offset = sizeof(EVAL_FM_FILE_MAGIC);
    const bool ok =
        read_scalar(res->bytes, &offset, &res->version) &&
        read_scalar(res->bytes, &offset, &res->n_phases) &&
        read_scalar(res->bytes, &offset, &res->linear_params_per_phase) &&
        read_scalar(res->bytes, &offset, &res->n_fm_phases) &&
        read_scalar(res->bytes, &offset, &res->n_features) &&
        read_scalar(res->bytes, &offset, &res->dim) &&
        read_scalar(res->bytes, &offset, &res->scale) &&
        read_scalar(res->bytes, &offset, &res->flags) &&
        read_scalar(res->bytes, &offset, &res->linear_count) &&
        read_scalar(res->bytes, &offset, &res->fm_count);
    if (!ok || offset != EVAL_FM_HEADER_SIZE || res->version != EVAL_FM_FILE_VERSION_LOCAL) {
        std::cerr << "[ERROR] broken egevfm header " << path << "\n";
        return false;
    }
    res->linear_offset = EVAL_FM_HEADER_SIZE;
    res->fm_offset = res->linear_offset + (size_t)res->linear_count * sizeof(int16_t);
    const size_t expected_size = res->fm_offset + (size_t)res->fm_count * sizeof(int8_t);
    if (expected_size != res->bytes.size() || res->dim == 0 || res->scale <= 0) {
        std::cerr << "[ERROR] unsupported egevfm layout " << path << "\n";
        return false;
    }
    return true;
}

bool compatible_with_first(const EgevfmFile &base, const EgevfmFile &target) {
    if (base.n_phases != target.n_phases ||
        base.linear_params_per_phase != target.linear_params_per_phase ||
        base.n_fm_phases != target.n_fm_phases ||
        base.n_features != target.n_features ||
        base.dim != target.dim ||
        base.scale != target.scale ||
        base.flags != target.flags ||
        base.linear_count != target.linear_count ||
        base.fm_count != target.fm_count ||
        base.linear_offset != target.linear_offset ||
        base.fm_offset != target.fm_offset) {
        return false;
    }
    return std::memcmp(
        base.bytes.data() + base.linear_offset,
        target.bytes.data() + target.linear_offset,
        (size_t)base.linear_count * sizeof(int16_t)
    ) == 0;
}

int round_div_nearest(const int sum, const int divisor) {
    if (sum >= 0) {
        return (sum + divisor / 2) / divisor;
    }
    return -((-sum + divisor / 2) / divisor);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: blend_egevfm [out.egevfm] [in1.egevfm] [in2.egevfm] ...\n";
        return 1;
    }

    const std::string out_file = argv[1];
    std::vector<EgevfmFile> inputs;
    inputs.reserve((size_t)argc - 2);
    for (int i = 2; i < argc; ++i) {
        EgevfmFile fm;
        if (!load_egevfm(argv[i], &fm)) {
            return 1;
        }
        if (!inputs.empty() && !compatible_with_first(inputs.front(), fm)) {
            std::cerr << "[ERROR] incompatible egevfm input " << argv[i] << "\n";
            return 1;
        }
        inputs.emplace_back(std::move(fm));
    }

    std::vector<char> output = inputs.front().bytes;
    int nonzero = 0;
    int max_abs = 0;
    int changed_from_first = 0;
    const size_t fm_offset = inputs.front().fm_offset;
    for (uint64_t i = 0; i < inputs.front().fm_count; ++i) {
        int sum = 0;
        for (const EgevfmFile &input: inputs) {
            int8_t value = 0;
            std::memcpy(&value, input.bytes.data() + fm_offset + (size_t)i, sizeof(int8_t));
            sum += value;
        }
        int blended = round_div_nearest(sum, (int)inputs.size());
        blended = std::clamp(blended, -127, 127);
        const int8_t q = (int8_t)blended;
        int8_t first = 0;
        std::memcpy(&first, inputs.front().bytes.data() + fm_offset + (size_t)i, sizeof(int8_t));
        if (q != first) {
            ++changed_from_first;
        }
        if (q != 0) {
            ++nonzero;
            max_abs = std::max(max_abs, std::abs((int)q));
        }
        std::memcpy(output.data() + fm_offset + (size_t)i, &q, sizeof(int8_t));
    }

    std::filesystem::path out_path(out_file);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << "\n";
        return 1;
    }
    out.write(output.data(), (std::streamsize)output.size());
    if (!out) {
        std::cerr << "[ERROR] failed to write output " << out_file << "\n";
        return 1;
    }

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "blend_count " << inputs.size() << "\n";
        summary << "n_phases " << inputs.front().n_phases << "\n";
        summary << "n_fm_phases " << inputs.front().n_fm_phases << "\n";
        summary << "dim " << inputs.front().dim << "\n";
        summary << "scale " << inputs.front().scale << "\n";
        summary << "flags " << inputs.front().flags << "\n";
        summary << "active_pattern_mask 0x" << std::hex << (inputs.front().flags & 0xFFFFU) << std::dec << "\n";
        summary << "linear_count " << inputs.front().linear_count << "\n";
        summary << "fm_count " << inputs.front().fm_count << "\n";
        summary << "nonzero_quantized " << nonzero << "\n";
        summary << "max_abs_quantized " << max_abs << "\n";
        summary << "changed_from_first " << changed_from_first << "\n";
        for (size_t i = 0; i < inputs.size(); ++i) {
            summary << "input" << i << " " << inputs[i].path << "\n";
        }
    }

    std::cout << "wrote " << out_file
              << " blend_count " << inputs.size()
              << " nonzero_quantized " << nonzero
              << " max_abs_quantized " << max_abs
              << " changed_from_first " << changed_from_first << "\n";
    return 0;
}
