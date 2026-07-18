/*
    Egaroucid Project

    @file evaluate_experiment_7_7_fm_simdopt_mmap_common.hpp
        Memory-mapped loader and optimized scorer for the isolated 7.7 beta + FM experiment
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

constexpr int EVAL77_FM_N_PATTERN_PARAMS_RAW = 944784;
constexpr int EVAL77_FM_N_PARAMS_PER_PHASE = EVAL77_FM_N_PATTERN_PARAMS_RAW + MAX_STONE_NUM;
constexpr int EVAL77_FM_MAX_DIM = 16;
constexpr int EVAL77_FM_EGEV5_VERSION = 9;
constexpr size_t EVAL77_FM_EGEV5_ALIGNMENT = 64;

constexpr int eval77_fm_pattern_offsets[N_PATTERNS] = {
    0, 59049, 118098, 177147,
    236196, 295245, 354294, 413343,
    472392, 531441, 590490, 649539,
    708588, 767637, 826686, 885735
};

struct Eval77FmMappedFile {
    const unsigned char *data = nullptr;
    size_t size = 0;
#if defined(_WIN32)
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = nullptr;
#else
    int fd = -1;
#endif

    ~Eval77FmMappedFile() {
        reset();
    }

    void reset() {
#if defined(_WIN32)
        if (data != nullptr) {
            UnmapViewOfFile(data);
        }
        if (mapping_handle != nullptr) {
            CloseHandle(mapping_handle);
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
        file_handle = INVALID_HANDLE_VALUE;
        mapping_handle = nullptr;
#else
        if (data != nullptr) {
            munmap((void*)data, size);
        }
        if (fd >= 0) {
            close(fd);
        }
        fd = -1;
#endif
        data = nullptr;
        size = 0;
    }

    bool open_readonly(const std::string &path) {
        reset();
#if defined(_WIN32)
        file_handle = CreateFileA(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr
        );
        if (file_handle == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle, &file_size) || file_size.QuadPart <= 0 ||
            (uint64_t)file_size.QuadPart > (uint64_t)std::numeric_limits<size_t>::max()) {
            reset();
            return false;
        }
        mapping_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == nullptr) {
            reset();
            return false;
        }
        data = (const unsigned char*)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (data == nullptr) {
            reset();
            return false;
        }
        size = (size_t)file_size.QuadPart;
#else
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0 ||
            (uint64_t)st.st_size > (uint64_t)std::numeric_limits<size_t>::max()) {
            reset();
            return false;
        }
        size = (size_t)st.st_size;
        void *view = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (view == MAP_FAILED) {
            data = nullptr;
            reset();
            return false;
        }
        data = (const unsigned char*)view;
#endif
        return true;
    }
};

Eval77FmMappedFile eval77_fm_mapped_file;
const unsigned char *eval77_fm_payload = nullptr;
const unsigned char *eval77_fm_linear_payload = nullptr;
const unsigned char *eval77_fm_vector_payload = nullptr;
size_t eval77_fm_param_stride = 0;
size_t eval77_fm_phase_stride = 0;
size_t eval77_fm_linear_param_stride = 0;
size_t eval77_fm_linear_phase_stride = 0;
size_t eval77_fm_vector_param_stride = 0;
size_t eval77_fm_vector_phase_stride = 0;
std::array<float, N_PHASES> eval77_fm_linear_scale;
std::array<float, N_PHASES> eval77_fm_vector_scale;
std::array<double, N_PHASES> eval77_fm_linear_scale_double;
std::array<double, N_PHASES> eval77_fm_interaction_scale;
uint64_t eval77_fm_interaction_phase_mask = 0;
int eval77_fm_dim = 0;
int eval77_fm_file_version = 0;
bool eval77_fm_loaded = false;
bool eval77_fm_split_layout = false;
uint32_t eval77_fm_subset_pattern_mask = 0xFFFFu;
bool eval77_fm_subset_use_count = true;

#if defined(EVALUATE_EXPERIMENT_7_7_FM_SUBSET_SIMDOPT) || \
    defined(EVALUATE_EXPERIMENT_7_7_FM_FAST_SUBSET) || \
    defined(EVALUATE_EXPERIMENT_7_7_FM_GROUPED_SUBSET)
    #define EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET
#endif

inline bool eval77_fm_parse_subset_file_spec(const char *file, std::string &data_file) {
    data_file = file;
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
#if defined(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK)
    static_assert(
        (static_cast<uint32_t>(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK) &
         ~uint32_t{0xFFFF}) == 0
    );
    eval77_fm_subset_pattern_mask =
        static_cast<uint32_t>(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK);
    #if defined(EVAL77_FM_COMPILED_SUBSET_USE_COUNT)
    static_assert(
        EVAL77_FM_COMPILED_SUBSET_USE_COUNT == 0 ||
        EVAL77_FM_COMPILED_SUBSET_USE_COUNT == 1
    );
    eval77_fm_subset_use_count =
        EVAL77_FM_COMPILED_SUBSET_USE_COUNT != 0;
    #else
    eval77_fm_subset_use_count = true;
    #endif
#else
    eval77_fm_subset_pattern_mask = 0xFFFFu;
    eval77_fm_subset_use_count = true;
#endif
    const size_t count_sep = data_file.rfind('@');
    if (count_sep == std::string::npos || count_sep == 0) {
        return true;
    }
    const size_t mask_sep = data_file.rfind('@', count_sep - 1);
    if (mask_sep == std::string::npos) {
        return true;
    }

    const std::string path = data_file.substr(0, mask_sep);
    const std::string mask_text = data_file.substr(mask_sep + 1, count_sep - mask_sep - 1);
    const std::string count_text = data_file.substr(count_sep + 1);
    if (path.empty() || mask_text.empty() || count_text.empty()) {
        std::cerr << "[ERROR] [FATAL] subset eval spec must be <eval.egev4>@<pattern_mask>@<use_count>" << std::endl;
        return false;
    }

    try {
        size_t pos = 0;
        const unsigned long mask = std::stoul(mask_text, &pos, 0);
        if (pos != mask_text.size() || (mask & ~0xFFFFul) != 0) {
            std::cerr << "[ERROR] [FATAL] subset pattern mask must be a 16-bit integer" << std::endl;
            return false;
        }
        pos = 0;
        const unsigned long use_count = std::stoul(count_text, &pos, 0);
        if (pos != count_text.size() || use_count > 1) {
            std::cerr << "[ERROR] [FATAL] subset count flag must be 0 or 1" << std::endl;
            return false;
        }
        eval77_fm_subset_pattern_mask = (uint32_t)mask;
        eval77_fm_subset_use_count = use_count != 0;
#if defined(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK)
        constexpr uint32_t compiled_mask =
            static_cast<uint32_t>(EVAL77_FM_COMPILED_SUBSET_PATTERN_MASK);
        #if defined(EVAL77_FM_COMPILED_SUBSET_USE_COUNT)
        constexpr bool compiled_use_count =
            EVAL77_FM_COMPILED_SUBSET_USE_COUNT != 0;
        #else
        constexpr bool compiled_use_count = true;
        #endif
        if (eval77_fm_subset_pattern_mask != compiled_mask ||
            eval77_fm_subset_use_count != compiled_use_count) {
            std::cerr
                << "[ERROR] [FATAL] subset eval spec differs from the "
                << "compile-time subset" << std::endl;
            return false;
        }
#endif
        data_file = path;
    } catch (const std::exception&) {
        std::cerr << "[ERROR] [FATAL] failed to parse subset eval spec "
                  << file << std::endl;
        return false;
    }
#endif
    return true;
}

inline int32_t eval77_fm_read_i32_le(const unsigned char *p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline size_t eval77_fm_align_up_size(const size_t value, const size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline bool eval77_fm_load_egev4(const char *file, bool show_log) {
    std::string data_file;
    if (!eval77_fm_parse_subset_file_spec(file, data_file)) {
        return false;
    }
    if (show_log) {
        std::cerr << "7.7 beta + FM experiment evaluation file " << data_file << std::endl;
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
        std::cerr << "7.7-FM subset pattern_mask=0x" << std::hex << eval77_fm_subset_pattern_mask
                  << std::dec << " use_count=" << (eval77_fm_subset_use_count ? 1 : 0) << std::endl;
#endif
    }
    eval77_fm_loaded = false;
    eval77_fm_payload = nullptr;
    eval77_fm_linear_payload = nullptr;
    eval77_fm_vector_payload = nullptr;
    eval77_fm_param_stride = 0;
    eval77_fm_phase_stride = 0;
    eval77_fm_linear_param_stride = 0;
    eval77_fm_linear_phase_stride = 0;
    eval77_fm_vector_param_stride = 0;
    eval77_fm_vector_phase_stride = 0;
    eval77_fm_dim = 0;
    eval77_fm_file_version = 0;
    eval77_fm_split_layout = false;
    eval77_fm_interaction_phase_mask = 0;
    if (!eval77_fm_mapped_file.open_readonly(data_file)) {
        std::cerr << "[ERROR] [FATAL] can't memory-map eval " << data_file << std::endl;
        return false;
    }

    constexpr size_t fixed_header_size = 34;
    constexpr size_t scales_size = sizeof(float) * N_PHASES * 2;
    constexpr size_t payload_offset = fixed_header_size + scales_size;
    if (eval77_fm_mapped_file.size < payload_offset) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM egev4 file is too short" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }
    const unsigned char *fixed_header = eval77_fm_mapped_file.data;
    if (std::memcmp(fixed_header + 14, "EGEV", 4) != 0) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM experiment expects an EGEV4-style file" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    const int32_t version = eval77_fm_read_i32_le(fixed_header + 18);
    const int32_t n_phase = eval77_fm_read_i32_le(fixed_header + 22);
    const int32_t n_param = eval77_fm_read_i32_le(fixed_header + 26);
    const int32_t dim = eval77_fm_read_i32_le(fixed_header + 30);
    if (version != 7 && version != 8 && version != EVAL77_FM_EGEV5_VERSION) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM experiment supports linear+FM egev4 versions 7/8 and egev5 version 9, found version "
                  << version << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }
    if (n_phase != N_PHASES || n_param != EVAL77_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > EVAL77_FM_MAX_DIM) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM egev4 header mismatch: phases=" << n_phase
                  << " params_per_phase=" << n_param << " dim=" << dim << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }

    eval77_fm_dim = dim;
    eval77_fm_file_version = version;
    eval77_fm_param_stride = sizeof(int16_t) + (size_t)eval77_fm_dim;
    eval77_fm_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * eval77_fm_param_stride;
    eval77_fm_split_layout = version == EVAL77_FM_EGEV5_VERSION;

    size_t expected_size;
    if (eval77_fm_split_layout) {
        const size_t linear_offset = eval77_fm_align_up_size(payload_offset, EVAL77_FM_EGEV5_ALIGNMENT);
        eval77_fm_linear_param_stride = sizeof(int16_t);
        eval77_fm_linear_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * eval77_fm_linear_param_stride;
        eval77_fm_vector_param_stride = eval77_fm_align_up_size((size_t)eval77_fm_dim, 16);
        eval77_fm_vector_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * eval77_fm_vector_param_stride;
        const size_t vector_offset = eval77_fm_align_up_size(
            linear_offset + (size_t)N_PHASES * eval77_fm_linear_phase_stride,
            EVAL77_FM_EGEV5_ALIGNMENT
        );
        expected_size = vector_offset + (size_t)N_PHASES * eval77_fm_vector_phase_stride;
        eval77_fm_linear_payload = fixed_header + linear_offset;
        eval77_fm_vector_payload = fixed_header + vector_offset;
    } else {
        expected_size = payload_offset + (size_t)N_PHASES * eval77_fm_phase_stride;
        eval77_fm_payload = fixed_header + payload_offset;
        eval77_fm_linear_payload = eval77_fm_payload;
        eval77_fm_vector_payload = eval77_fm_payload + sizeof(int16_t);
        eval77_fm_linear_param_stride = eval77_fm_param_stride;
        eval77_fm_linear_phase_stride = eval77_fm_phase_stride;
        eval77_fm_vector_param_stride = eval77_fm_param_stride;
        eval77_fm_vector_phase_stride = eval77_fm_phase_stride;
    }
    if (eval77_fm_mapped_file.size < expected_size) {
        std::cerr << "[ERROR] [FATAL] 7.7-FM evaluation payload is truncated" << std::endl;
        eval77_fm_mapped_file.reset();
        return false;
    }
    std::memcpy(eval77_fm_linear_scale.data(), fixed_header + fixed_header_size, sizeof(float) * N_PHASES);
    std::memcpy(eval77_fm_vector_scale.data(), fixed_header + fixed_header_size + sizeof(float) * N_PHASES,
                sizeof(float) * N_PHASES);
    for (int phase = 0; phase < N_PHASES; ++phase) {
        eval77_fm_linear_scale_double[phase] = (double)eval77_fm_linear_scale[phase];
        const double vector_scale = (double)eval77_fm_vector_scale[phase];
        eval77_fm_interaction_scale[phase] = 0.5 * vector_scale * vector_scale;
        if (phase >= 6 && vector_scale != 0.0) {
            eval77_fm_interaction_phase_mask |= uint64_t{1} << phase;
        }
    }

    eval77_fm_loaded = true;
    if (show_log) {
        std::cerr << "7.7-FM evaluation loaded: version " << version
                  << " phases " << n_phase
                  << " params/phase " << n_param
                  << " dim " << eval77_fm_dim
                  << " layout " << (eval77_fm_split_layout ? "split-aligned" : "interleaved")
                  << " mapped_bytes " << eval77_fm_mapped_file.size << std::endl;
    }
    return true;
}

inline const unsigned char *eval77_fm_param_ptr(const int phase_idx, const int id) {
    return eval77_fm_linear_payload + (size_t)phase_idx * eval77_fm_linear_phase_stride +
        (size_t)id * eval77_fm_linear_param_stride;
}

inline int16_t eval77_fm_linear_quant_at(const int phase_idx, const int id) {
    int16_t value;
    std::memcpy(&value, eval77_fm_param_ptr(phase_idx, id), sizeof(value));
    return value;
}

inline const int8_t *eval77_fm_vector_quant_ptr(const int phase_idx, const int id) {
    return (const int8_t*)(eval77_fm_vector_payload + (size_t)phase_idx * eval77_fm_vector_phase_stride +
        (size_t)id * eval77_fm_vector_param_stride);
}

inline double eval77_fm_linear_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0.0;
    }
    const double linear_scale = eval77_fm_linear_scale[phase_idx];
    double linear = 0.0;
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        linear += (double)eval77_fm_linear_quant_at(phase_idx, id) * linear_scale;
    }
    return linear;
}

inline double eval77_fm_interaction_from_ids_quant(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx || eval77_fm_dim <= 0) {
        return 0.0;
    }
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum{};
    std::array<int32_t, EVAL77_FM_MAX_DIM> sum_sq{};
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        const int8_t *vector_quant = eval77_fm_vector_quant_ptr(phase_idx, id);
        for (int dim = 0; dim < eval77_fm_dim; ++dim) {
            const int32_t v = vector_quant[dim];
            sum[dim] += v;
            sum_sq[dim] += v * v;
        }
    }

    int64_t interaction_quant = 0;
    for (int dim = 0; dim < eval77_fm_dim; ++dim) {
        const int64_t s = sum[dim];
        interaction_quant += s * s - sum_sq[dim];
    }
    const double scale = (double)eval77_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}

#if USE_SIMD
struct Eval77FmSimdAccum {
    __m256i sum16;
    __m256i sum_sq_pair32;
};

inline void eval77_fm_clear_simd_accum(Eval77FmSimdAccum &accum) {
    const __m256i zero = _mm256_setzero_si256();
    accum.sum16 = zero;
    accum.sum_sq_pair32 = zero;
}

inline int64_t eval77_fm_reduce_epi32(const __m256i values) {
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256((__m256i*)tmp, values);
    int64_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += tmp[i];
    }
    return sum;
}

inline void eval77_fm_add_vector_simd16(
    const int phase_idx,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = eval77_fm_vector_quant_ptr(phase_idx, id);
    const __m128i q8 = _mm_loadu_si128((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline void eval77_fm_add_vector_simd8(
    const int phase_idx,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = eval77_fm_vector_quant_ptr(phase_idx, id);
    const __m128i q8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline void eval77_fm_add_vector_simd12(
    const int phase_idx,
    const int id,
    Eval77FmSimdAccum &accum
) {
    const int8_t *src_ptr = eval77_fm_vector_quant_ptr(phase_idx, id);
    const __m128i low8 = _mm_loadl_epi64((const __m128i*)src_ptr);
    int32_t tail4 = 0;
    std::memcpy(&tail4, src_ptr + 8, sizeof(tail4));
    const __m128i high4 = _mm_slli_si128(_mm_cvtsi32_si128(tail4), 8);
    const __m128i q8 = _mm_or_si128(low8, high4);
    const __m256i q16 = _mm256_cvtepi8_epi16(q8);
    accum.sum16 = _mm256_add_epi16(accum.sum16, q16);
    accum.sum_sq_pair32 = _mm256_add_epi32(accum.sum_sq_pair32, _mm256_madd_epi16(q16, q16));
}

inline double eval77_fm_interaction_from_ids_simd_padded16(const int phase_idx, const int active_ids[], const int n_active) {
    if (!eval77_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx ||
        (eval77_fm_dim != 8 && eval77_fm_dim != 12 && eval77_fm_dim != 16)) {
        return 0.0;
    }
    Eval77FmSimdAccum accum;
    eval77_fm_clear_simd_accum(accum);
    if (eval77_fm_dim == 16) {
        for (int i = 0; i < n_active; ++i) {
            const int id = active_ids[i];
            if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
                continue;
            }
            eval77_fm_add_vector_simd16(phase_idx, id, accum);
        }
    } else if (eval77_fm_dim == 12) {
        for (int i = 0; i < n_active; ++i) {
            const int id = active_ids[i];
            if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
                continue;
            }
            eval77_fm_add_vector_simd12(phase_idx, id, accum);
        }
    } else {
        for (int i = 0; i < n_active; ++i) {
            const int id = active_ids[i];
            if (id < 0 || EVAL77_FM_N_PARAMS_PER_PHASE <= id) {
                continue;
            }
            eval77_fm_add_vector_simd8(phase_idx, id, accum);
        }
    }

    const __m256i sum_sq_pair32 = _mm256_madd_epi16(accum.sum16, accum.sum16);
    const __m256i pair_term = _mm256_sub_epi32(sum_sq_pair32, accum.sum_sq_pair32);
    const int64_t interaction_quant = eval77_fm_reduce_epi32(pair_term);
    const double scale = (double)eval77_fm_vector_scale[phase_idx];
    return 0.5 * (double)interaction_quant * scale * scale;
}
#endif

inline double eval77_fm_interaction_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
#if USE_SIMD
    if (eval77_fm_dim == 8 || eval77_fm_dim == 12 || eval77_fm_dim == 16) {
        return eval77_fm_interaction_from_ids_simd_padded16(phase_idx, active_ids, n_active);
    }
#endif
    return eval77_fm_interaction_from_ids_quant(phase_idx, active_ids, n_active);
}

inline int eval77_fm_score_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    const double score = eval77_fm_linear_from_ids(phase_idx, active_ids, n_active) +
        eval77_fm_interaction_from_ids(phase_idx, active_ids, n_active);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline bool eval77_fm_subset_pattern_type_enabled(const int pattern_type) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
    if (pattern_type < 0 || N_PATTERNS <= pattern_type) {
        return false;
    }
    return ((eval77_fm_subset_pattern_mask >> pattern_type) & 1u) != 0;
#else
    (void)pattern_type;
    return true;
#endif
}

inline bool eval77_fm_subset_feature_enabled(const int id) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
    if (id >= EVAL77_FM_N_PATTERN_PARAMS_RAW) {
        return eval77_fm_subset_use_count;
    }
    if (id < 0) {
        return false;
    }
    return eval77_fm_subset_pattern_type_enabled(id / 59049);
#else
    (void)id;
    return true;
#endif
}

inline int eval77_fm_filter_subset_ids(const int active_ids[], const int n_active, int subset_ids[]) {
    int n_subset = 0;
    for (int i = 0; i < n_active; ++i) {
        if (eval77_fm_subset_feature_enabled(active_ids[i])) {
            subset_ids[n_subset++] = active_ids[i];
        }
    }
    return n_subset;
}

inline int eval77_fm_score_from_linear_and_fm_ids(
    const int phase_idx,
    const int linear_ids[],
    const int n_linear,
    const int fm_ids[],
    const int n_fm
) {
    const double score = eval77_fm_linear_from_ids(phase_idx, linear_ids, n_linear) +
        eval77_fm_interaction_from_ids(phase_idx, fm_ids, n_fm);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

inline int eval77_fm_score_from_ids_subset_filter(const int phase_idx, const int active_ids[], const int n_active) {
#if defined(EVALUATE_EXPERIMENT_7_7_FM_PATTERN_SUBSET)
    int subset_ids[N_PATTERN_FEATURES + 1];
    const int n_subset = eval77_fm_filter_subset_ids(active_ids, n_active, subset_ids);
    return eval77_fm_score_from_linear_and_fm_ids(phase_idx, active_ids, n_active, subset_ids, n_subset);
#else
    return eval77_fm_score_from_ids(phase_idx, active_ids, n_active);
#endif
}
