/*
    Egaroucid Project

    @file blend_egev4_linear_weights.cpp
        Blend only the linear int16 weights of two compatible EGEV4 files.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;
constexpr size_t FIXED_HEADER_SIZE =
    TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 4;
constexpr size_t HEADER_SIZE =
    FIXED_HEADER_SIZE + sizeof(float) * N_PHASES * 2;
constexpr size_t CHUNK_RECORDS = 65536;

struct ModelInfo {
    std::array<unsigned char, HEADER_SIZE> header{};
    int n_params = 0;
    int dim = 0;
    size_t record_stride = 0;
    size_t phase_stride = 0;
    uint64_t expected_size = 0;
};

template<typename T>
T read_value(const unsigned char *data) {
    T value{};
    std::memcpy(&value, data, sizeof(value));
    return value;
}

ModelInfo inspect_model(const fs::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const std::streamoff file_size = in.tellg();
    in.seekg(0);

    ModelInfo info;
    in.read(
        reinterpret_cast<char *>(info.header.data()),
        static_cast<std::streamsize>(info.header.size())
    );
    if (!in ||
        std::memcmp(info.header.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("invalid EGEV4 header: " + path.string());
    }
    const int version = read_value<int32_t>(info.header.data() + 18);
    const int n_phases = read_value<int32_t>(info.header.data() + 22);
    info.n_params = read_value<int32_t>(info.header.data() + 26);
    info.dim = read_value<int32_t>(info.header.data() + 30);
    if (version != VERSION_LINEAR_FM_INT16_INT8 ||
        n_phases != N_PHASES ||
        info.n_params <= 0 ||
        info.dim <= 0) {
        throw std::runtime_error("unsupported EGEV4 shape: " + path.string());
    }
    info.record_stride = sizeof(int16_t) + static_cast<size_t>(info.dim);
    info.phase_stride =
        static_cast<size_t>(info.n_params) * info.record_stride;
    info.expected_size =
        HEADER_SIZE + static_cast<uint64_t>(N_PHASES) * info.phase_stride;
    if (file_size < 0 ||
        static_cast<uint64_t>(file_size) != info.expected_size) {
        throw std::runtime_error("EGEV4 size mismatch: " + path.string());
    }
    return info;
}

void read_exact(
    std::istream &in,
    char *data,
    size_t size,
    const std::string &label
) {
    in.read(data, static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("cannot read " + label);
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr
            << "usage: blend_egev4_linear_weights "
            << "<base.egev4> <candidate.egev4> <output.egev4> <alpha>"
            << std::endl;
        return 1;
    }

    try {
        const fs::path base_path = argv[1];
        const fs::path candidate_path = argv[2];
        const fs::path output_path = argv[3];
        const double alpha = std::stod(argv[4]);
        if (alpha < 0.0 || alpha > 1.0) {
            throw std::runtime_error("alpha must be in [0, 1]");
        }

        const ModelInfo base = inspect_model(base_path);
        const ModelInfo candidate = inspect_model(candidate_path);
        if (base.n_params != candidate.n_params ||
            base.dim != candidate.dim ||
            std::memcmp(
                base.header.data() + FIXED_HEADER_SIZE,
                candidate.header.data() + FIXED_HEADER_SIZE,
                HEADER_SIZE - FIXED_HEADER_SIZE
            ) != 0) {
            throw std::runtime_error(
                "input shapes or quantization scales do not match"
            );
        }

        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }
        fs::copy_file(
            base_path,
            output_path,
            fs::copy_options::overwrite_existing
        );
        std::ifstream candidate_in(candidate_path, std::ios::binary);
        std::fstream output(
            output_path,
            std::ios::binary | std::ios::in | std::ios::out
        );
        std::vector<char> base_buffer(
            CHUNK_RECORDS * base.record_stride
        );
        std::vector<char> candidate_buffer(
            CHUNK_RECORDS * base.record_stride
        );
        uint64_t changed = 0;
        int max_abs_quant_delta = 0;
        const uint64_t total_records =
            static_cast<uint64_t>(N_PHASES) *
            static_cast<uint64_t>(base.n_params);
        uint64_t processed = 0;
        while (processed < total_records) {
            const size_t records = static_cast<size_t>(
                std::min<uint64_t>(
                    CHUNK_RECORDS,
                    total_records - processed
                )
            );
            const size_t bytes = records * base.record_stride;
            const std::streamoff offset = static_cast<std::streamoff>(
                HEADER_SIZE + processed * base.record_stride
            );
            output.seekg(offset);
            candidate_in.seekg(offset);
            read_exact(
                output,
                base_buffer.data(),
                bytes,
                "base payload"
            );
            read_exact(
                candidate_in,
                candidate_buffer.data(),
                bytes,
                "candidate payload"
            );
            for (size_t i = 0; i < records; ++i) {
                const size_t offset = i * base.record_stride;
                int16_t base_value = 0;
                int16_t candidate_value = 0;
                std::memcpy(
                    &base_value,
                    base_buffer.data() + offset,
                    sizeof(base_value)
                );
                std::memcpy(
                    &candidate_value,
                    candidate_buffer.data() + offset,
                    sizeof(candidate_value)
                );
                const int blended = static_cast<int>(std::llround(
                    (1.0 - alpha) * static_cast<double>(base_value) +
                    alpha * static_cast<double>(candidate_value)
                ));
                const int16_t blended_value = static_cast<int16_t>(
                    std::clamp(blended, -32768, 32767)
                );
                changed += blended_value != base_value;
                max_abs_quant_delta = std::max(
                    max_abs_quant_delta,
                    std::abs(
                        static_cast<int>(blended_value) -
                        static_cast<int>(base_value)
                    )
                );
                std::memcpy(
                    base_buffer.data() + offset,
                    &blended_value,
                    sizeof(blended_value)
                );
            }
            output.seekp(offset);
            output.write(
                base_buffer.data(),
                static_cast<std::streamsize>(bytes)
            );
            if (!output) {
                throw std::runtime_error("cannot write blended payload");
            }
            processed += records;
        }

        std::cout << "alpha=" << alpha
                  << " changed_linear_values=" << changed
                  << " max_abs_quant_delta=" << max_abs_quant_delta
                  << " output=" << output_path.string()
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
